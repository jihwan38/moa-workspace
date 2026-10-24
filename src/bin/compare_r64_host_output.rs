//! Full-model R64 ownership proof with one pool: replay identical independent
//! request histories under different physical allocations and row groupings.
//! The reference uses the same R64 arithmetic, avoiding a scalar-vs-FP8 false gate.
use furiosa_opt_gemma4::{
    host::{
        generate::read_logits,
        load,
        model::{ForwardOutput, Input, Model, Sequence},
        runtime::{Blocks, Workspace},
    },
    ops,
};
use furiosa_opt_std::prelude::*;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn input(r: usize, p: usize) -> Input<'static> {
    Input {
        token: 2 + (r * 313 + p * 37) % 8000,
        soft: None,
        logits: true,
        image: None,
        audio: None,
    }
}
fn selected(step: usize) -> Vec<usize> {
    match step {
        1 => (40..57).collect(),
        2 => vec![63],
        3 => (0..64).rev().collect(),
        4 => (17..48).collect(),
        _ => (0..64).collect(),
    }
}
fn check(step: usize, r: usize, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    let mut diff = 0;
    let mut max = 0.0f32;
    for (&a, &b) in a.iter().zip(b) {
        assert!(a.is_finite() && b.is_finite());
        diff += usize::from(a != b);
        max = max.max((a - b).abs());
    }
    println!(
        "step={step} request={r} mismatches={diff}/{} max_abs={max}",
        a.len()
    );
    assert_eq!(
        diff, 0,
        "row regrouping or physical allocation changed same-arithmetic R64 output"
    );
}
#[tokio::main]
async fn main() -> Result<()> {
    let mut device = Device::new(ops::embed_token.topology())?;
    let weights = load::load_weights(&mut device).await?;
    let mut ws = Workspace::new_canonical64(&mut device, &weights, 2048, 64).await?;
    let mut model = Model::new(&mut device, weights, &mut ws).await?;
    let mut expected: Vec<Vec<Vec<f32>>> = Vec::new();
    for pass in 0..2 {
        let blocks: Vec<_> = (0..64)
            .map(|r| {
                let slot = if pass == 0 { r } else { (r * 13 + 31) % 64 };
                Blocks {
                    sliding: slot,
                    full: (0..4)
                        .map(|page| {
                            if pass == 0 {
                                slot * 4 + page
                            } else {
                                255 - (slot * 4 + page)
                            }
                        })
                        .collect(),
                }
            })
            .collect();
        let mut positions = [0usize; 64];
        for step in 0..6 {
            let chosen = selected(step);
            let rows: Vec<_> = chosen.iter().map(|&r| [input(r, positions[r])]).collect();
            let mut collected = Vec::new();
            let width = if pass == 0 {
                64
            } else {
                match step {
                    0 => 17,
                    1 => 7,
                    2 => 1,
                    3 => 64,
                    4 => 16,
                    _ => 31,
                }
            };
            for start in (0..chosen.len()).step_by(width) {
                let end = (start + width).min(chosen.len());
                let sequences: Vec<_> = (start..end)
                    .map(|i| {
                        let r = chosen[i];
                        Sequence {
                            inputs: &rows[i],
                            start_position: positions[r],
                            blocks: &blocks[r],
                        }
                    })
                    .collect();
                model.begin_stage_profile();
                let now = std::time::Instant::now();
                let values = if pass == 1 && sequences.len() > 16 {
                    match model.forward_outputs(&mut device, &sequences, &mut ws).await? {
                        ForwardOutput::Blocked64(output, live) => {
                            let host = output.to_host::<furiosa_opt_gemma4::ops_r64_io_blocked::BlockedLogits>(&mut device.pdma).await?;
                            let bytes = host.into_inner().into_buf();
                            furiosa_opt_gemma4::host_logit_bytes::decode_blocked_r64(&bytes, live, 8)
                        }
                        ForwardOutput::Rows(_) => return Err("expected native blocked output".into()),
                    }
                } else {
                    let outputs = model.forward_sequences_r64(&mut device, &sequences, &mut ws).await?;
                    let mut values=Vec::new();
                    for out in outputs { values.push(read_logits(&mut device, out).await?); }
                    values
                };
                let elapsed = now.elapsed().as_secs_f64() * 1000.0;
                for (i, values) in values.into_iter().enumerate() {
                    if pass == 1 { check(step, chosen[start + i], &expected[step][start + i], &values); }
                    collected.push(values);
                }
                println!(
                    "pass={pass} step={step} rows={} model_and_read_ms={elapsed:.3}",
                    end - start
                );
                if pass == 0 && step == 3 {
                    for stage in model.take_stage_profile() {
                        println!(
                            "stage={} calls={} ms={:.3}",
                            stage.stage, stage.calls, stage.milliseconds
                        );
                    }
                }
            }
            if pass == 0 {
                expected.push(collected);
            }
            for r in chosen {
                positions[r] += 1;
            }
        }
    }
    println!(
        "PASS actual-model64 raw blocked host outputs: reordered pools, 64/17/1/31 live counts, regrouped row prefixes and position0 reuse"
    );
    Ok(())
}
