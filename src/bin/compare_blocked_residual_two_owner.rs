//! Same two-owner384+128 arithmetic with canonical versus persistent residual state.
//! Args: A start(default1664), B start(default1920), interleave0/1(default1).
use furiosa_opt_gemma4::{
    LAYERS,
    host::{
        generate::read_logits,
        load,
        model::{Input, Model, Sequence},
        runtime::{Blocks, Workspace},
    },
    ops,
};
use furiosa_opt_std::prelude::*;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn input(request: usize, position: usize, logits: bool) -> Input<'static> {
    Input {
        token: 2 + (request * 733 + position * 37) % 8000,
        soft: None,
        logits,
        image: None,
        audio: None,
    }
}
fn check(label: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    let mut differences = 0;
    let mut nonfinite = 0;
    for (a, b) in a.iter().zip(b) {
        differences += usize::from(a.to_bits() != b.to_bits());
        nonfinite += usize::from(!a.is_finite() || !b.is_finite());
    }
    println!(
        "PAIR384128_EXACT label={label} values={} bitdiff={differences} nonfinite={nonfinite}",
        a.len()
    );
    assert_eq!(nonfinite, 0);
    assert_eq!(differences, 0);
}
async fn fill(
    d: &mut Device,
    m: &mut Model,
    w: &mut Workspace,
    blocks: &[Blocks],
    starts: [usize; 2],
) -> Result<()> {
    for owner in 0..2 {
        for slot in [owner, owner + 2] {
            let mut position = 0;
            while position < starts[owner] {
                let count = if starts[owner] - position >= 512 {
                    512
                } else {
                    128
                };
                let rows: Vec<_> = (position..position + count)
                    .map(|p| input(owner, p, false))
                    .collect();
                let sequence = Sequence {
                    inputs: &rows,
                    start_position: position,
                    blocks: &blocks[slot],
                };
                if count == 512 {
                    m.start_prefill512(d, &sequence, w).await?;
                    for layer in 0..LAYERS {
                        assert_eq!(
                            m.advance_prefill(d, w).await?.is_some(),
                            layer + 1 == LAYERS
                        );
                    }
                } else {
                    m.forward_sequences_native_qkv(d, &[sequence], w).await?;
                }
                position += count;
            }
        }
    }
    Ok(())
}
async fn original(
    d: &mut Device,
    m: &mut Model,
    w: &mut Workspace,
    blocks: &[Blocks],
    starts: [usize; 2],
    prompts: &[Vec<Input<'static>>; 2],
) -> Result<(Vec<Vec<f32>>, f64)> {
    let sequences = [
        Sequence {
            inputs: &prompts[0],
            start_position: starts[0],
            blocks: &blocks[0],
        },
        Sequence {
            inputs: &prompts[1],
            start_position: starts[1],
            blocks: &blocks[1],
        },
    ];
    let mut outputs = Vec::new();
    let now = std::time::Instant::now();
    m.start_prefill384_128_original_blocked_residual(d, &sequences, w)
        .await?;
    let mut ms = now.elapsed().as_secs_f64() * 1000.0;
    for layer in 0..LAYERS {
        let now = std::time::Instant::now();
        let result = m.advance_prefill384_128(d, w).await?;
        ms += now.elapsed().as_secs_f64() * 1000.0;
        if let Some(rows) = result {
            assert_eq!(layer + 1, LAYERS);
            for row in rows {
                outputs.push(read_logits(d, row).await?);
            }
        } else {
            assert!(layer + 1 < LAYERS);
        }
    }
    assert_eq!(outputs.len(), 2);
    Ok((outputs, ms))
}
async fn pair(
    d: &mut Device,
    m: &mut Model,
    w: &mut Workspace,
    blocks: &[Blocks],
    starts: [usize; 2],
    prompts: &[Vec<Input<'static>>; 2],
    expected: &[Vec<f32>],
    interleave: bool,
) -> Result<f64> {
    let sequences = [
        Sequence {
            inputs: &prompts[0],
            start_position: starts[0],
            blocks: &blocks[2],
        },
        Sequence {
            inputs: &prompts[1],
            start_position: starts[1],
            blocks: &blocks[3],
        },
    ];
    let now = std::time::Instant::now();
    m.start_prefill384_128(d, &sequences, w).await?;
    let mut ms = now.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(
        m.active_prefill_rings(),
        vec![blocks[2].sliding, blocks[3].sliding]
    );
    assert_eq!(m.active_prefill_ring(), Some(blocks[2].sliding));
    // Both actual owners, and an independent ring sharing either full allocation,
    // remain protected throughout the checkpoint. Only two reservations are exposed.
    for owner in 0..2 {
        let one = [input(owner, starts[owner], true)];
        assert!(
            m.forward_sequences(
                d,
                &[Sequence {
                    inputs: &one,
                    start_position: starts[owner],
                    blocks: &blocks[owner + 2]
                }],
                w
            )
            .await
            .is_err()
        );
        let alias = Blocks {
            sliding: blocks[4].sliding,
            full: blocks[owner + 2].full.clone(),
        };
        assert!(
            m.forward_sequences(
                d,
                &[Sequence {
                    inputs: &one,
                    start_position: starts[owner],
                    blocks: &alias
                }],
                w
            )
            .await
            .is_err()
        );
    }
    assert!(m.start_prefill384_128(d, &sequences, w).await.is_err());
    assert!(m.advance_prefill_group(d, w).await.is_err());
    let other: Vec<_> = (0..128).map(|p| input(9, p, false)).collect();
    assert!(
        m.forward_sequences(
            d,
            &[Sequence {
                inputs: &other,
                start_position: 0,
                blocks: &blocks[4]
            }],
            w
        )
        .await
        .is_err()
    );
    for layer in 0..LAYERS {
        let now = std::time::Instant::now();
        let output = m.advance_prefill384_128(d, w).await?;
        ms += now.elapsed().as_secs_f64() * 1000.0;
        if let Some(outputs) = output {
            assert_eq!(layer + 1, LAYERS);
            assert_eq!(outputs.len(), 2);
            for owner in 0..2 {
                check(
                    &format!("pair_owner{owner}"),
                    &expected[owner],
                    &read_logits(d, &outputs[owner]).await?,
                );
            }
        } else {
            assert!(layer + 1 < LAYERS);
        }
        if interleave {
            let rows = [input(7, layer, true)];
            let mut reference = Vec::new();
            for slot in [4, 5] {
                let out = m
                    .forward_sequences_r64(
                        d,
                        &[Sequence {
                            inputs: &rows,
                            start_position: layer,
                            blocks: &blocks[slot],
                        }],
                        w,
                    )
                    .await?;
                let value = read_logits(d, &out[0]).await?;
                if slot == 4 {
                    reference = value;
                } else {
                    check(&format!("independent_R64_layer{layer}"), &reference, &value);
                }
            }
        }
    }
    assert!(m.active_prefill_rings().is_empty());
    assert!(m.active_prefill_ring().is_none());
    Ok(ms)
}
#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let starts = [
        args.get(1)
            .map_or("1664", String::as_str)
            .parse::<usize>()?,
        args.get(2)
            .map_or("1920", String::as_str)
            .parse::<usize>()?,
    ];
    let interleave = args.get(3).map_or("1", String::as_str) == "1";
    let ends = [
        starts[0].checked_add(384).ok_or("A overflow")?,
        starts[1].checked_add(128).ok_or("B overflow")?,
    ];
    let capacity = ends
        .iter()
        .max()
        .unwrap()
        .checked_add(64)
        .ok_or("capacity overflow")?;
    let pages = Workspace::blocks_for(capacity);
    if starts.iter().any(|p| p % 128 != 0) || 6 * pages > 256 {
        return Err("128-aligned starts and at most256 physical pages required".into());
    }
    let mut d = Device::new(ops::embed_token.topology())?;
    let weights = load::load_weights(&mut d).await?;
    let mut w = Workspace::new_canonical64(&mut d, &weights, capacity, 64).await?;
    let mut m = Model::new(&mut d, weights, &mut w).await?;
    m.prepare_group_scratch(&mut d).await?;
    let rings = [3, 27, 61, 52, 7, 49];
    let blocks: Vec<_> = rings
        .into_iter()
        .enumerate()
        .map(|(owner, sliding)| Blocks {
            sliding,
            full: (0..pages).map(|p| 255 - owner * pages - p).collect(),
        })
        .collect();
    fill(&mut d, &mut m, &mut w, &blocks, starts).await?;
    let prompts: [Vec<_>; 2] = std::array::from_fn(|owner| {
        (starts[owner]..ends[owner])
            .map(|p| input(owner, p, p + 1 == ends[owner]))
            .collect()
    });
    m.begin_stage_profile();
    let (expected, control_ms) =
        original(&mut d, &mut m, &mut w, &blocks, starts, &prompts).await?;
    for s in m.take_stage_profile() {
        println!(
            "PAIR384128_CONTROL_STAGE stage={} calls={} ms={:.6}",
            s.stage, s.calls, s.milliseconds
        );
    }
    m.begin_stage_profile();
    let pair_ms = pair(
        &mut d, &mut m, &mut w, &blocks, starts, &prompts, &expected, interleave,
    )
    .await?;
    for s in m.take_stage_profile() {
        println!(
            "PAIR384128_STAGE stage={} calls={} ms={:.6}",
            s.stage, s.calls, s.milliseconds
        );
    }
    // Includes the first full page after2048 in the default case.
    for step in 0..4 {
        let rows: [[_; 1]; 2] =
            std::array::from_fn(|owner| [input(owner, ends[owner] + step, true)]);
        let mut reference = Vec::new();
        for offset in [0, 2] {
            let seq: Vec<_> = (0..2)
                .map(|owner| Sequence {
                    inputs: &rows[owner],
                    start_position: ends[owner] + step,
                    blocks: &blocks[owner + offset],
                })
                .collect();
            let out = m.forward_sequences(&mut d, &seq, &mut w).await?;
            for owner in 0..2 {
                let value = read_logits(&mut d, &out[owner]).await?;
                if offset == 0 {
                    reference.push(value);
                } else {
                    check(
                        &format!("post_pair_B16_step{step}_owner{owner}"),
                        &reference[owner],
                        &value,
                    );
                }
            }
        }
    }
    // Reuse the same occupied ring/page reservations from position0. Old masks,
    // row identities and the completed pair state must not survive this reset.
    let reset: [Vec<_>; 2] = std::array::from_fn(|owner| {
        let count = if owner == 0 { 384 } else { 128 };
        (0..count)
            .map(|p| input(owner + 11, p, p + 1 == count))
            .collect()
    });
    let (expected, reset_control_ms) =
        original(&mut d, &mut m, &mut w, &blocks, [0, 0], &reset).await?;
    let reset_pair_ms = pair(
        &mut d,
        &mut m,
        &mut w,
        &blocks,
        [0, 0],
        &reset,
        &expected,
        false,
    )
    .await?;
    println!(
        "PASS persistent residual Prompt384128 starts={starts:?} interleave={interleave} owners=2 quarter_order=A0_A1_A2_B0 all_vocab_exact=true canonical_same_group_ms={control_ms:.3} paired_ms={pair_ms:.3} reset_control_ms={reset_control_ms:.3} reset_pair_ms={reset_pair_ms:.3}; timing excludes downloads/interleaved calls, not HTTP"
    );
    Ok(())
}
