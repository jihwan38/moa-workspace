//! q019 MSE FFN in BOTH arms; only q018 B16 FP8 LM is toggled.
//! Teacher-forced all-logit drift, own selection, physical replay and KV isolation.
use furiosa_opt_gemma4::{
    LAYERS,
    axes::W,
    host::{
        b16_f4_weights,
        b16_metadata_diagnostic::{collect_retained, logits_bits},
        greedy::GreedySelector,
        load,
        model::{Input, Model, Sequence},
        runtime::{Blocks, Workspace},
        sampling::{SamplingConfig, sample},
    },
    host_logit_bytes::bf16_bits_to_f32,
    ops,
};
use furiosa_opt_std::prelude::*;
use rand::{SeedableRng, rngs::StdRng};
use std::{collections::HashSet, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
type Rows = Vec<Vec<u16>>;

fn input(owner: usize, p: usize, logits: bool) -> Input<'static> {
    Input {
        token: 2 + (owner * 313 + p * 37) % 8000,
        soft: None,
        logits,
        image: None,
        audio: None,
    }
}
fn blocks(owner: usize) -> Blocks {
    Blocks {
        sliding: (owner * 13 + 7) % 32,
        full: (0..4).map(|p| 255 - owner * 4 - p).collect(),
    }
}
#[cfg(backend = "npu")]
fn memory(d: &Device, label: &str) -> Result<usize> {
    let m = d.memory()?;
    println!(
        "Q021_LM_MEMORY label={label} capacity={} available={} largest={}",
        m.capacity, m.available, m.largest
    );
    Ok(m.available)
}
#[cfg(not(backend = "npu"))]
fn memory(_: &Device, _: &str) -> Result<usize> {
    Err("memory probe is NPU only".into())
}

async fn fill(
    d: &mut Device,
    m: &mut Model,
    w: &mut Workspace,
    allocation: &[Blocks],
    prefix: usize,
) -> Result<()> {
    let begin = Instant::now();
    for (owner, block) in allocation.iter().enumerate() {
        for p in (0..prefix).step_by(512) {
            let rows: Vec<_> = (p..p + 512).map(|p| input(owner, p, false)).collect();
            m.start_prefill512(
                d,
                &Sequence {
                    inputs: &rows,
                    start_position: p,
                    blocks: block,
                },
                w,
            )
            .await?;
            for layer in 0..LAYERS {
                assert_eq!(
                    m.advance_prefill(d, w).await?.is_some(),
                    layer + 1 == LAYERS
                );
            }
        }
    }
    println!(
        "Q021_LM_FILL owners={} actual_tokens={} elapsed_ms={:.6}",
        allocation.len(),
        allocation.len() * prefix,
        begin.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}

fn chosen(step: usize) -> Vec<usize> {
    match step {
        0 => vec![0],
        1 => vec![1, 2],
        2 => (0..9).rev().collect(),
        3 => (0..8).rev().collect(),
        4 | 7 => (0..16).map(|i| (i * 5 + 3) % 16).collect(),
        5 => (0..14).rev().collect(),
        6 => vec![0, 2],
        _ => (0..8).collect(),
    }
}
fn exact(label: &str, a: &[u16], b: &[u16]) {
    assert_eq!(a.len(), b.len());
    let count = a.iter().zip(b).filter(|(a, b)| a != b).count();
    assert_eq!(count, 0, "{label}: all-vocabulary BF16 mismatch");
    let af: Vec<_> = a.iter().map(|&x| bf16_bits_to_f32(x)).collect();
    let bf: Vec<_> = b.iter().map(|&x| bf16_bits_to_f32(x)).collect();
    let bans: HashSet<_> = [0, 1, 106, 107].into_iter().collect();
    for config in [
        SamplingConfig::DEFAULT,
        SamplingConfig {
            temperature: 0.8,
            top_k: 37,
            top_p: 0.9,
        },
    ] {
        assert_eq!(
            sample(&af, &config, &bans, &mut StdRng::seed_from_u64(71)),
            sample(&bf, &config, &bans, &mut StdRng::seed_from_u64(71))
        );
    }
}
fn drift(step: usize, owner: usize, a: &[u16], b: &[u16]) {
    let mut sq = 0.0f64;
    let mut norm = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut differences = 0usize;
    let af: Vec<_> = a.iter().map(|&x| bf16_bits_to_f32(x)).collect();
    let bf: Vec<_> = b.iter().map(|&x| bf16_bits_to_f32(x)).collect();
    for ((&a, &b), (&ab, &bb)) in af.iter().zip(&bf).zip(a.iter().zip(b)) {
        assert!(a.is_finite() && b.is_finite());
        sq += (a as f64 - b as f64).powi(2);
        norm += (a as f64).powi(2);
        max_abs = max_abs.max((a as f64 - b as f64).abs());
        differences += usize::from(ab != bb);
    }
    let bans = HashSet::new();
    let at = sample(
        &af,
        &SamplingConfig::DEFAULT,
        &bans,
        &mut StdRng::seed_from_u64(3),
    );
    let bt = sample(
        &bf,
        &SamplingConfig::DEFAULT,
        &bans,
        &mut StdRng::seed_from_u64(3),
    );
    println!(
        "Q021_LM_DRIFT step={step} owner={owner} values={} bitdiff={differences} relative_l2={} max_abs={max_abs} q019_bf16_lm_top1={at} candidate_top1={bt} same_top1={} teacher_forced=true quality_gate=false",
        a.len(),
        (sq / norm.max(1e-30)).sqrt(),
        at == bt
    );
}

async fn model_case(
    d: &mut Device,
    m: &mut Model,
    w: &mut Workspace,
    selector: &mut GreedySelector,
    prefix: usize,
) -> Result<()> {
    let allocations: Vec<Vec<_>> = (0..2)
        .map(|half| (0..16).map(|r| blocks(half * 16 + r)).collect())
        .collect();
    let mut original: Vec<Rows> = Vec::new();
    let mut candidate: Vec<Rows> = Vec::new();
    let mut prompt_reference = Vec::new();
    let mut exact_values = 0usize;
    let mut cache_values = 0usize;
    for pass in 0..3 {
        let allocation = &allocations[usize::from(pass == 1)];
        // Pass2 replays every touched coordinate of pass0. Untouched cells were
        // initialized equally; the union of writes is identical in all passes.
        fill(d, m, w, allocation, prefix).await?;
        let prompt: Vec<_> = (prefix..prefix + 128)
            .map(|p| input(15, p, p == prefix + 127))
            .collect();
        m.start_prefill(
            d,
            &Sequence {
                inputs: &prompt,
                start_position: prefix,
                blocks: &allocation[15],
            },
            w,
        )
        .await?;
        assert!(
            m.prepare_b16_lm_fp8(d).await.is_err(),
            "even idempotent preparation must reject an active checkpoint"
        );
        let mut positions = vec![prefix; 16];
        for step in 0..10 {
            let owners = chosen(step);
            if step == 6 {
                for &r in &owners {
                    positions[r] = 0;
                }
            }
            let rows: Vec<_> = owners
                .iter()
                .map(|&r| [input(r, positions[r], true)])
                .collect();
            let seq: Vec<_> = owners
                .iter()
                .enumerate()
                .map(|(i, &r)| Sequence {
                    inputs: &rows[i],
                    start_position: positions[r],
                    blocks: &allocation[r],
                })
                .collect();
            m.set_b16_epilogue_fused(step != 8);
            m.begin_stage_profile();
            let begin = Instant::now();
            let output = if pass == 0 {
                m.forward_outputs_original_b16_lm(d, &seq, w).await?
            } else if step == 1 {
                // Shadow existence must not alter ordinary Automatic calls.
                m.forward_outputs(d, &seq, w).await?
            } else {
                m.forward_outputs_b16_lm_fp8(d, &seq, w).await?
            };
            let model_ms = begin.elapsed().as_secs_f64() * 1000.0;
            let found = collect_retained(
                d,
                selector,
                output,
                owners.len(),
                if step == 3 { 2 } else { 0 },
            )
            .await?;
            for (i, bits) in found.iter().enumerate() {
                assert_eq!(bits.len(), W::SIZE);
                if pass == 1 {
                    drift(step, owners[i], &original[step][i], bits);
                    if step <= 1 {
                        exact("scalar/default-off fallback", &original[step][i], bits);
                        exact_values += bits.len();
                    }
                } else if pass == 2 {
                    exact("same-candidate physical replay", &candidate[step][i], bits);
                    exact_values += bits.len();
                }
            }
            if pass == 0 {
                original.push(found);
            } else if pass == 1 {
                candidate.push(found);
            }
            println!(
                "Q021_LM_STEP pass={pass} step={step} live={} candidate={} automatic={} fused_epilogue={} model_ms={model_ms:.6}",
                owners.len(),
                pass != 0,
                pass != 0 && step == 1,
                step != 8
            );
            for stage in m.take_stage_profile() {
                println!(
                    "Q021_LM_STAGE pass={pass} step={step} stage={} calls={} total_ms={:.6}",
                    stage.stage, stage.calls, stage.milliseconds
                );
            }
            for r in owners {
                positions[r] += 1;
            }
            if step < 4 {
                for quantum in 0..12 {
                    if let Some(out) = m.advance_prefill(d, w).await? {
                        assert_eq!((step, quantum), (3, 11));
                        let bits = logits_bits(d, out).await?;
                        if pass == 0 {
                            prompt_reference = bits;
                        } else {
                            exact("unchanged interleaved P128", &prompt_reference, &bits);
                            exact_values += bits.len();
                        }
                        positions[15] += 128;
                    } else {
                        assert_ne!((step, quantum), (3, 11));
                    }
                }
            }
        }
        m.set_b16_epilogue_fused(true);
        if pass > 0 {
            // Before replay overwrites control storage, verify LM-only candidate
            // changed no KV byte. Repeat after candidate physical replay.
            for owner in 0..16 {
                cache_values += m
                    .compare_canonical_kv(d, w, &allocations[0][owner], &allocations[1][owner])
                    .await?;
            }
            println!(
                "Q021_LM_CACHE pass={pass} owners=16 cumulative_bf16_values={cache_values} bitdiff=0 both_arms_q019_mse_ffn=true"
            );
        }
    }
    println!(
        "PASS Q021_LM_MODEL prefix={prefix} replay_exact_logits={exact_values} replay_exact_cache_bf16_values={cache_values} q019_bf16_lm_drift_reported=true own_output_greedy_verified=true seeded_stochastic_replay_exact=true quality_gate=false mse_ffn_default_enabled={}",
        b16_f4_weights::ENABLED
    );
    Ok(())
}

async fn guards_case(
    d: &mut Device,
    m: &mut Model,
    w: &mut Workspace,
    selector: &mut GreedySelector,
    prefix: usize,
) -> Result<()> {
    let allocation: Vec<_> = (0..17).map(blocks).collect();
    let mut expected: Vec<Rows> = Vec::new();
    let mut checked = 0;
    // Sequential passes retain exactly the same physical mapping and recompute
    // every history from zero. Reused future cells are masked by the same routes.
    for pass in 0..2 {
        fill(d, m, w, &allocation, prefix).await?;
        let mut positions = vec![prefix; 17];
        for step in 0..3 {
            let owners: Vec<_> = if step == 0 {
                vec![0]
            } else {
                (0..17).map(|i| (i * 5 + step) % 17).collect()
            };
            let rows: Vec<_> = owners
                .iter()
                .map(|&r| [input(r, positions[r], true)])
                .collect();
            let seq: Vec<_> = owners
                .iter()
                .enumerate()
                .map(|(i, &r)| Sequence {
                    inputs: &rows[i],
                    start_position: positions[r],
                    blocks: &allocation[r],
                })
                .collect();
            let output = if pass == 0 {
                m.forward_outputs_original_b16_lm(d, &seq, w).await?
            } else {
                m.forward_outputs_b16_lm_fp8(d, &seq, w).await?
            };
            let found = collect_retained(d, selector, output, owners.len(), 0).await?;
            if pass == 0 {
                expected.push(found);
            } else {
                for (a, b) in expected[step].iter().zip(&found) {
                    exact("scalar/R32 unaffected", a, b);
                    checked += b.len();
                }
            }
            for r in owners {
                positions[r] += 1;
            }
        }
    }
    println!(
        "PASS Q021_LM_GUARDS actual_history_prefix={prefix} owners=17 scalar_and_R32_exact_logits={checked} no_B16_LM_dispatch=true"
    );
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "model".into());
    let prefix = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "512".into())
        .parse::<usize>()?;
    assert!(["model", "guards"].contains(&mode.as_str()));
    assert!([512, 1024].contains(&prefix));
    assert!(
        b16_f4_weights::ENABLED,
        "both arms must retain q019 Automatic MSE FFN"
    );
    let begin = Instant::now();
    let mut d = Device::new(ops::embed_token.topology())?;
    let weights = load::load_weights(&mut d).await?;
    let mut w = Workspace::new_canonical64(&mut d, &weights, 2048, 32).await?;
    furiosa_opt_gemma4::host::kv_diagnostic::verify_physical32_identity(&w, 2048)?;
    let mut m = Model::new(&mut d, weights, &mut w).await?;
    m.prepare_group_scratch(&mut d).await?;
    m.prepare_projection_owned_ffn(&mut d).await?;
    let mut selector = GreedySelector::new(&mut d).await?;
    assert!(
        m.b16_f4_ready(),
        "q019 startup must publish MSE FFN before LM shadow"
    );
    let b = [blocks(0), blocks(1)];
    let rows = [[input(0, 0, true)], [input(1, 0, true)]];
    let seq: Vec<_> = (0..2)
        .map(|i| Sequence {
            inputs: &rows[i],
            start_position: 0,
            blocks: &b[i],
        })
        .collect();
    assert!(
        m.forward_outputs_b16_lm_fp8(&mut d, &seq, &mut w)
            .await
            .is_err(),
        "explicit LM call without LM shadow must fail before model work"
    );
    let timing = m
        .prepare_b16_f4_with_quantizer(&mut d, b16_f4_weights::F4Quantizer::MseDyadic)
        .await?;
    assert_eq!(timing.projections, LAYERS * 3);
    assert!(
        m.prepare_b16_f4_with_quantizer(&mut d, b16_f4_weights::F4Quantizer::PowerOfTwoRange)
            .await
            .is_err(),
        "q019 immutable MSE source cannot be silently replaced"
    );
    let before = memory(&d, "q019_mse_complete_before_lm_shadow")?;
    let startup = Instant::now();
    m.prepare_b16_lm_fp8(&mut d).await?;
    let startup_ms = startup.elapsed().as_secs_f64() * 1000.;
    let after = memory(&d, "q019_mse_plus_fp8_lm_shadow")?;
    assert!(
        before >= after && before - after <= 1_100_000_000,
        "LM shadow allocation cap"
    );
    println!(
        "Q021_LM_STARTUP ms={startup_ms:.6} allocator_delta={} original_bf16_embedding_retained=true ff_numerics=q019_MseDyadic startup_parallel=false",
        before - after
    );
    m.prepare_b16_lm_fp8(&mut d).await?;
    assert_eq!(memory(&d, "idempotent_lm_prepare")?, after);
    if mode == "model" {
        model_case(&mut d, &mut m, &mut w, &mut selector, prefix).await?;
    } else {
        guards_case(&mut d, &mut m, &mut w, &mut selector, prefix).await?;
    }
    memory(&d, "completed")?;
    println!(
        "Q021_LM_FINISHED mode={mode} prefix={prefix} both_arms_ffn=q019_MseDyadic elapsed_s={:.6} no_public_quality_claim=true no_activation=true",
        begin.elapsed().as_secs_f64()
    );
    Ok(())
}
