//! Actual-model metadata fallback and retained-output oracle. No serving hooks.
use super::{
    greedy::GreedySelector,
    load,
    model::{ForwardOutput, Input, Logits, Model, Sequence},
    runtime::{Blocks, KvCache, Workspace},
    sampling::{sample, SamplingConfig},
};
use crate::{axes::*, host_logit_bytes::bf16_bits_to_f32, ops, Chip};
use furiosa_opt_std::prelude::*;
use rand::{rngs::StdRng, SeedableRng};
use std::{collections::HashSet, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

async fn bytes<S: M>(d: &mut Device, x: &HbmTensor<bf16, Chip, S>) -> Result<Vec<u8>> {
    Ok(x.to_host::<S>(&mut d.pdma).await?.into_inner().into_buf())
}
pub async fn logits_bits(d: &mut Device, x: &Logits) -> Result<Vec<u16>> {
    let raw = bytes(d, x).await?;
    let out: Vec<_> = raw.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
    assert_eq!(out.len(), W::SIZE);
    assert!(out.iter().all(|x| x & 0x7f80 != 0x7f80), "nonfinite actual model logits");
    Ok(out)
}

/// Consume the borrow fully before another model call. Every requested vocabulary
/// row is read for this oracle; production only reads exceptional/stochastic rows.
/// `policy_case == 2` checks None (stochastic/unrequested) and all-banned fallback.
pub async fn collect_retained(
    d: &mut Device,
    selector: &mut GreedySelector,
    output: ForwardOutput<'_>,
    count: usize,
    policy_case: usize,
) -> Result<Vec<Vec<u16>>> {
    let mut bans: Vec<HashSet<usize>> = (0..count)
        .map(|r| [0, 1, 106, 107, 31 + r, W::SIZE + r].into_iter().collect())
        .collect();
    if policy_case == 2 && count > 1 {
        bans[0] = (0..W::SIZE).collect();
    }
    let policies: Vec<_> = bans.iter().enumerate()
        .map(|(r, b)| if policy_case == 2 && r == count - 1 { None } else { Some(b) })
        .collect();
    let mut result = Vec::with_capacity(count);
    match output {
        ForwardOutput::Rows(rows) => {
            assert_eq!(count, 1, "pure multi-row oracle must retain native logits");
            assert_eq!(rows.len(), count);
            for row in rows { result.push(logits_bits(d, row).await?); }
            println!("B16_METADATA_OUTPUT layout=Rows live={count} device_selected=0");
        }
        ForwardOutput::Batch16(logits, live) => {
            assert!((2..=B::SIZE).contains(&count));
            assert_eq!(live, count);
            let start = Instant::now();
            let selected = selector.select_b16(d, logits, &policies).await?;
            let select_ms = start.elapsed().as_secs_f64() * 1000.0;
            let mut device_selected = 0;
            for row in 0..count {
                let values = selector.read_fallback_b16(d, logits, row).await?;
                verify_selection(row, &values, selected[row], policies[row], &mut device_selected);
                result.push(f32_bits(&values));
            }
            assert!(device_selected > 0, "actual B16 oracle must exercise device selection");
            println!("B16_METADATA_OUTPUT layout=Batch16 live={live} policy_case={policy_case} device_selected={device_selected} select_ms={select_ms:.6}");
        }
        ForwardOutput::Blocked64(logits, live) => {
            assert!(count > B::SIZE);
            assert_eq!(live, count);
            let start = Instant::now();
            let selected = selector.select(d, logits, &policies).await?;
            let select_ms = start.elapsed().as_secs_f64() * 1000.0;
            let mut device_selected = 0;
            for row in 0..count {
                let values = selector.read_fallback(d, logits, row).await?;
                verify_selection(row, &values, selected[row], policies[row], &mut device_selected);
                result.push(f32_bits(&values));
            }
            assert!(device_selected > 0, "actual R64 oracle must exercise device selection");
            println!("B16_METADATA_OUTPUT layout=Blocked64 live={live} policy_case={policy_case} device_selected={device_selected} select_ms={select_ms:.6}");
        }
    }
    Ok(result)
}
fn f32_bits(values: &[f32]) -> Vec<u16> {
    assert_eq!(values.len(), W::SIZE);
    assert!(values.iter().all(|x| x.is_finite()));
    values.iter().map(|x| {
        assert_eq!(x.to_bits() & 0xffff, 0, "fallback must preserve BF16 values");
        (x.to_bits() >> 16) as u16
    }).collect()
}
fn verify_selection(row: usize, values: &[f32], token: Option<usize>, policy: Option<&HashSet<usize>>, selected: &mut usize) {
    match policy {
        Some(bans) => {
            let expected = sample(values, &SamplingConfig::DEFAULT, bans, &mut StdRng::seed_from_u64(61));
            if let Some(actual) = token { assert_eq!(actual, expected, "device greedy row={row}"); *selected += 1; }
            else { assert_eq!(sample(values, &SamplingConfig::DEFAULT, bans, &mut StdRng::seed_from_u64(91)), expected); }
        }
        None => assert_eq!(token, None, "unrequested/stochastic row must remain unselected"),
    }
}
fn compare_logits(case: usize, owner: usize, a: &[u16], b: &[u16]) {
    let mismatches = a.iter().zip(b).filter(|(x,y)| x != y).count();
    assert_eq!(a.len(), b.len());
    assert_eq!(mismatches, 0, "metadata logits case={case} owner={owner}");
    let a: Vec<_> = a.iter().map(|&x| bf16_bits_to_f32(x)).collect();
    let b: Vec<_> = b.iter().map(|&x| bf16_bits_to_f32(x)).collect();
    let banned = [0, 1, 106, 107].into_iter().collect();
    for config in [SamplingConfig::DEFAULT, SamplingConfig { temperature: 0.8, top_k: 37, top_p: 0.9 }] {
        let seed = case as u64 * 997 + owner as u64;
        assert_eq!(sample(&a, &config, &banned, &mut StdRng::seed_from_u64(seed)),
            sample(&b, &config, &banned, &mut StdRng::seed_from_u64(seed)));
    }
}
fn input(owner: usize, position: usize, logits: bool) -> Input<'static> {
    Input { token: 2 + (owner * 313 + position * 37) % 8000, soft: None, logits, image: None, audio: None }
}
async fn compare_legacy(d: &mut Device, ws: &Workspace, a: &Blocks, b: &Blocks) -> Result<usize> {
    assert!(!ws.is_canonical() && ws.request_pool.is_none());
    assert_eq!(a.full.len(), b.full.len());
    let mut checked = 0;
    for (layer, cache) in ws.kv_cache.iter().enumerate() {
        match cache {
            KvCache::Sliding { k, v } => {
                for (name, buffers) in [("K", k), ("V", v)] {
                    let left = bytes(d, &buffers[a.sliding]).await?;
                    let right = bytes(d, &buffers[b.sliding]).await?;
                    assert_eq!(left.len(), right.len());
                    assert_eq!(left.iter().zip(&right).filter(|(a,b)| a != b).count(), 0, "legacy sliding {layer} {name}"); checked += left.len();
                }
            }
            KvCache::Full { k, v } => {
                for (&pa, &pb) in a.full.iter().zip(&b.full) {
                    for (name, buffers) in [("K", k), ("V", v)] {
                        let left = bytes(d, &buffers[pa]).await?;
                        let right = bytes(d, &buffers[pb]).await?;
                        assert_eq!(left.len(), right.len());
                        assert_eq!(left.iter().zip(&right).filter(|(a,b)| a != b).count(), 0, "legacy full {layer} {name}"); checked += left.len();
                    }
                }
            }
            _ => return Err("reference workspace unexpectedly contains canonical KV".into()),
        }
    }
    Ok(checked)
}

pub async fn run_legacy() -> Result<()> {
    let begin = Instant::now();
    let mut d = Device::new(ops::embed_token.topology())?;
    let weights = load::load_weights(&mut d).await?;
    let mut ws = Workspace::new_reference(&mut d, &weights, 64, 8).await?;
    assert!(!ws.is_canonical() && ws.request_pool.is_none());
    let mut model = Model::new(&mut d, weights, &mut ws).await?;
    let mut selector = GreedySelector::new(&mut d).await?;
    let blocks: Vec<Vec<_>> = (0..2).map(|pass| (0..4).map(|owner| Blocks {
        sliding: if pass == 0 { owner } else { 7 - owner },
        full: vec![if pass == 0 { (owner * 3) % 4 } else { 7 - (owner * 3) % 4 }],
    }).collect()).collect();
    for pass in 0..2 {
        model.set_b16_compact_decode_metadata(pass == 1);
        for owner in 0..4 {
            let inputs: Vec<_> = (0..16).map(|p| input(owner, p, false)).collect();
            model.forward_sequences(&mut d, &[Sequence { inputs: &inputs, start_position: 0, blocks: &blocks[pass][owner] }], &mut ws).await?;
        }
    }
    let orders = [vec![0, 1, 2, 3], vec![3, 1], vec![2], vec![1, 3, 0], vec![3, 2, 1, 0]];
    let mut positions = [16; 4];
    let mut checked = 0;
    for (case, owners) in orders.iter().enumerate() {
        if case == 3 { positions[1] = 0; }
        let inputs: Vec<_> = owners.iter().map(|&r| [input(r, positions[r], true)]).collect();
        let mut snapshots = Vec::new();
        // Alternate physical arm order without sharing either arm's KV history.
        let order = if case % 2 == 0 { [1, 0] } else { [0, 1] };
        for pass in order {
            model.set_b16_compact_decode_metadata(pass == 1);
            let seq: Vec<_> = owners.iter().enumerate().map(|(i, &r)| Sequence { inputs: &inputs[i], start_position: positions[r], blocks: &blocks[pass][r] }).collect();
            let start = Instant::now();
            let values = if pass == 0 {
                let out = model.forward_sequences(&mut d, &seq, &mut ws).await?;
                let model_ms = start.elapsed().as_secs_f64() * 1000.0;
                let mut values = Vec::new(); for row in out { values.push(logits_bits(&mut d, row).await?); }
                println!("B16_METADATA_LEGACY case={case} pass={pass} live={} layout=materialized model_ms={model_ms:.6}", owners.len()); values
            } else {
                let out = model.forward_outputs(&mut d, &seq, &mut ws).await?;
                let model_ms = start.elapsed().as_secs_f64() * 1000.0;
                let values = collect_retained(&mut d, &mut selector, out, owners.len(), if case == 0 { 2 } else { 0 }).await?;
                println!("B16_METADATA_LEGACY case={case} pass={pass} live={} layout=retained model_ms={model_ms:.6}", owners.len()); values
            };
            snapshots.push((pass, values));
        }
        snapshots.sort_by_key(|x| x.0);
        for (row, &owner) in owners.iter().enumerate() {
            compare_logits(case, owner, &snapshots[0].1[row], &snapshots[1].1[row]); checked += W::SIZE; positions[owner] += 1;
        }
    }
    let mut cache_bytes = 0;
    for owner in 0..4 {
        let n = compare_legacy(&mut d, &ws, &blocks[0][owner], &blocks[1][owner]).await?;
        cache_bytes += n; println!("B16_METADATA_LEGACY_CACHE owner={owner} all_bytes={n} bitdiff=0");
    }
    println!("PASS B16 metadata actual unpooled fallback logits={checked} legacy_cache_bytes={cache_bytes};four owners,16-token real histories,width4/2/1/3/4,reset,permutation,retained-device-greedy/all-banned/stochastic;elapsed_s={:.3};different output APIs so timing is diagnostic only", begin.elapsed().as_secs_f64());
    Ok(())
}
