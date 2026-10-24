//! Four optimized P128 controls versus staged P512 FFN, with optional R64 interleave.
//! Args: four starts(default896/384/0/128), then interleave0/1(default0).
use furiosa_opt_gemma4::{
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
fn input(r: usize, p: usize, logits: bool) -> Input<'static> {
    Input {
        token: 2 + (r * 733 + p * 37) % 8000,
        soft: None,
        logits,
        image: None,
        audio: None,
    }
}
fn check(label: &str, a: &[f32], b: &[f32]) {
    let mut n = 0;
    let mut max = 0.0f32;
    assert_eq!(a.len(), b.len());
    for (&a, &b) in a.iter().zip(b) {
        assert!(a.is_finite() && b.is_finite());
        n += usize::from(a != b);
        max = max.max((a - b).abs());
    }
    println!("{label} mismatches={n}/{} max_abs={max}", a.len());
    assert_eq!(n, 0, "grouped prompt changed original P128 arithmetic");
}
#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let starts = [
        args.get(1).map_or("896", String::as_str).parse::<usize>()?,
        args.get(2).map_or("384", String::as_str).parse::<usize>()?,
        args.get(3).map_or("0", String::as_str).parse::<usize>()?,
        args.get(4).map_or("128", String::as_str).parse::<usize>()?,
    ];
    let interleave = args.get(5).map_or("0", String::as_str) == "1";
    let capacity = starts.iter().max().unwrap() + 192;
    let pages = Workspace::blocks_for(capacity);
    if pages > 4 || starts.iter().any(|p| p / 512 != (p + 127) / 512) {
        return Err("four page-local128 chunks and context<=2048 required".into());
    }
    let mut d = Device::new(ops::embed_token.topology())?;
    let weights = load::load_weights(&mut d).await?;
    let mut ws = Workspace::new_canonical64(&mut d, &weights, capacity, 64).await?;
    let mut model = Model::new(&mut d, weights, &mut ws).await?;
    let blocks: Vec<_> = (0..64)
        .map(|slot| Blocks {
            sliding: slot,
            full: (slot * pages..(slot + 1) * pages).collect(),
        })
        .collect();
    for r in 0..4 {
        let mut pos = 0;
        while pos < starts[r] {
            let left = (starts[r] - pos).min(512 - pos % 512);
            let n = if left >= 128 { 128 } else { left.min(16) };
            let rows: Vec<_> = (pos..pos + n).map(|p| input(r, p, false)).collect();
            for slot in [r, 60 + r] {
                model
                    .forward_sequences(
                        &mut d,
                        &[Sequence {
                            inputs: &rows,
                            start_position: pos,
                            blocks: &blocks[slot],
                        }],
                        &mut ws,
                    )
                    .await?;
            }
            pos += n;
        }
    }
    let prompts: Vec<Vec<_>> = (0..4)
        .map(|r| {
            (starts[r]..starts[r] + 128)
                .map(|p| input(r, p, p + 1 == starts[r] + 128))
                .collect()
        })
        .collect();
    let mut expected = Vec::new();
    let mut control_ms = 0.0;
    model.begin_stage_profile();
    for r in 0..4 {
        let now = std::time::Instant::now();
        let out = model
            .forward_sequences(
                &mut d,
                &[Sequence {
                    inputs: &prompts[r],
                    start_position: starts[r],
                    blocks: &blocks[r],
                }],
                &mut ws,
            )
            .await?;
        control_ms += now.elapsed().as_secs_f64() * 1000.0;
        expected.push(read_logits(&mut d, &out[0]).await?);
    }
    for stage in model.take_stage_profile() {
        println!(
            "control_stage={} calls={} ms={:.6}",
            stage.stage, stage.calls, stage.milliseconds
        );
    }
    let seq: Vec<_> = (0..4)
        .map(|r| Sequence {
            inputs: &prompts[r],
            start_position: starts[r],
            blocks: &blocks[60 + r],
        })
        .collect();
    let now = std::time::Instant::now();
    model.start_prefill_group(&mut d, &seq, &mut ws).await?;
    let prepare = now.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(model.active_prefill_rings(), vec![60, 61, 62, 63]);
    for sequence in &seq {
        assert!(
            model
                .forward_sequences(&mut d, std::slice::from_ref(sequence), &mut ws)
                .await
                .is_err()
        );
    }
    model.begin_stage_profile();
    let mut total = 0.0f64;
    let mut max = 0.0f64;
    for layer in 0..furiosa_opt_gemma4::LAYERS {
        let now = std::time::Instant::now();
        let out = model.advance_prefill_group(&mut d, &mut ws).await?;
        let elapsed = now.elapsed().as_secs_f64() * 1000.0;
        total += elapsed;
        max = max.max(elapsed);
        if let Some(outputs) = out {
            assert_eq!(layer + 1, furiosa_opt_gemma4::LAYERS);
            for (r, out) in outputs.iter().enumerate() {
                check(
                    &format!("prompt={r}"),
                    &expected[r],
                    &read_logits(&mut d, out).await?,
                );
            }
        }
        println!("layer={layer} grouped_quantum_ms={elapsed:.3}");
        if interleave {
            let row = [input(4, layer, true)];
            let out = model
                .forward_sequences_r64(
                    &mut d,
                    &[Sequence {
                        inputs: &row,
                        start_position: layer,
                        blocks: &blocks[4],
                    }],
                    &mut ws,
                )
                .await?;
            let expected = read_logits(&mut d, &out[0]).await?;
            let out = model
                .forward_sequences_r64(
                    &mut d,
                    &[Sequence {
                        inputs: &row,
                        start_position: layer,
                        blocks: &blocks[5],
                    }],
                    &mut ws,
                )
                .await?;
            check(
                &format!("interleaved={layer}"),
                &expected,
                &read_logits(&mut d, &out[0]).await?,
            );
        }
    }
    assert!(model.active_prefill_rings().is_empty());
    let profile = model.take_stage_profile();
    // A new token must observe the completed prompt KV in both physical layouts.
    let next: Vec<_> = (0..4).map(|r| [input(r, starts[r] + 128, true)]).collect();
    let control: Vec<_> = (0..4)
        .map(|r| Sequence {
            inputs: &next[r],
            start_position: starts[r] + 128,
            blocks: &blocks[r],
        })
        .collect();
    let candidate: Vec<_> = (0..4)
        .map(|r| Sequence {
            inputs: &next[r],
            start_position: starts[r] + 128,
            blocks: &blocks[60 + r],
        })
        .collect();
    let out = model.forward_sequences(&mut d, &control, &mut ws).await?;
    let mut after = Vec::new();
    for row in out {
        after.push(read_logits(&mut d, row).await?);
    }
    let out = model.forward_sequences(&mut d, &candidate, &mut ws).await?;
    for (r, row) in out.iter().enumerate() {
        check(
            &format!("post_group_decode={r}"),
            &after[r],
            &read_logits(&mut d, row).await?,
        );
    }
    for stage in profile {
        println!(
            "stage={} calls={} ms={:.3}",
            stage.stage, stage.calls, stage.milliseconds
        );
    }
    println!(
        "PASS grouped prompt starts={starts:?} interleave={interleave} control4_ms={control_ms:.3} cold_prepare_ms={prepare:.3} group_layers_ms={total:.3} max_quantum_ms={max:.3}"
    );
    Ok(())
}
