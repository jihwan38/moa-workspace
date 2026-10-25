//! Original grouped norm launches versus exact fused Prompt512 execution.
//! Args: start(default896), interleave0/1(default1). No HTTP dispatch changes.
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
fn check(label: &str, control: &[f32], candidate: &[f32]) {
    assert_eq!(control.len(), candidate.len());
    let mut mismatches = 0;
    let mut max_abs = 0.0f32;
    for (&a, &b) in control.iter().zip(candidate) {
        assert!(a.is_finite() && b.is_finite(), "{label}: nonfinite logit");
        mismatches += usize::from(a.to_bits() != b.to_bits());
        max_abs = max_abs.max((a - b).abs());
    }
    println!(
        "{label} mismatches={mismatches}/{} max_abs={max_abs}",
        control.len()
    );
    assert_eq!(
        mismatches, 0,
        "norm fusion changed original grouped arithmetic"
    );
}
#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let start = args.get(1).map_or("896", String::as_str).parse::<usize>()?;
    let interleave = args.get(2).map_or("1", String::as_str) == "1";
    let end = start.checked_add(512).ok_or("position overflow")?;
    let capacity = end.checked_add(64).ok_or("position overflow")?;
    let pages = Workspace::blocks_for(capacity);
    if start % 128 != 0 || pages > 4 {
        return Err(
            "need128-aligned start and context<=2048 for64 disjoint probe allocations".into(),
        );
    }
    let mut device = Device::new(ops::embed_token.topology())?;
    let weights = load::load_weights(&mut device).await?;
    let mut workspace = Workspace::new_canonical64(&mut device, &weights, capacity, 64).await?;
    let mut model = Model::new(&mut device, weights, &mut workspace).await?;
    let scratch_start = std::time::Instant::now();
    model.prepare_group_scratch(&mut device).await?;
    println!(
        "startup.group_scratch_ms={:.3}",
        scratch_start.elapsed().as_secs_f64() * 1000.0
    );
    // Reversed physical pages also exercise the last legal full page. There is
    // one canonical pool; control and candidate own disjoint ring/page pairs.
    let blocks: Vec<_> = (0..64)
        .map(|ring| Blocks {
            sliding: ring,
            full: (0..pages).map(|page| 255 - ring * pages - page).collect(),
        })
        .collect();
    let mut position = 0;
    while position < start {
        let left = (start - position).min(512 - position % 512);
        let count = if left >= 128 { 128 } else { left.min(16) };
        let inputs: Vec<_> = (position..position + count)
            .map(|p| input(0, p, false))
            .collect();
        for slot in [0, 63] {
            model
                .forward_sequences_native_qkv(
                    &mut device,
                    &[Sequence {
                        inputs: &inputs,
                        start_position: position,
                        blocks: &blocks[slot],
                    }],
                    &mut workspace,
                )
                .await?;
        }
        position += count;
    }
    let inputs: Vec<_> = (start..end).map(|p| input(0, p, p + 1 == end)).collect();
    let mut control_ms;
    let mut expected = Vec::new();
    model.begin_stage_profile();
    let started = std::time::Instant::now();
    model.start_prefill512_original_norms(&mut device, &Sequence {
        inputs: &inputs, start_position: start, blocks: &blocks[0],
    }, &mut workspace).await?;
    control_ms = started.elapsed().as_secs_f64() * 1000.0;
    for layer in 0..LAYERS {
        let started = std::time::Instant::now();
        let output = model.advance_prefill(&mut device, &mut workspace).await?;
        control_ms += started.elapsed().as_secs_f64() * 1000.0;
        if let Some(output) = output {
            assert_eq!(layer + 1, LAYERS);
            expected = read_logits(&mut device, output).await?;
        } else { assert!(layer + 1 < LAYERS); }
    }
    for stage in model.take_stage_profile() {
        println!(
            "control_stage={} calls={} ms={:.6}",
            stage.stage, stage.calls, stage.milliseconds
        );
    }
    let sequence = Sequence {
        inputs: &inputs,
        start_position: start,
        blocks: &blocks[63],
    };
    let started = std::time::Instant::now();
    model
        .start_prefill512(&mut device, &sequence, &mut workspace)
        .await?;
    let prepare_ms = started.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(model.active_prefill_rings(), vec![63]);
    assert_eq!(model.active_prefill_ring(), Some(63));
    assert!(
        model
            .forward_sequences(&mut device, std::slice::from_ref(&sequence), &mut workspace)
            .await
            .is_err()
    );
    let page_alias = Blocks {
        sliding: 59,
        full: blocks[63].full.clone(),
    };
    assert!(
        model
            .forward_sequences(
                &mut device,
                &[Sequence {
                    inputs: &inputs[..1],
                    start_position: start,
                    blocks: &page_alias,
                }],
                &mut workspace
            )
            .await
            .is_err()
    );
    model.begin_stage_profile();
    let mut layer_ms = 0.0f64;
    let mut max_quantum_ms = 0.0f64;
    for layer in 0..LAYERS {
        let started = std::time::Instant::now();
        let output = model.advance_prefill(&mut device, &mut workspace).await?;
        let elapsed = started.elapsed().as_secs_f64() * 1000.0;
        layer_ms += elapsed;
        max_quantum_ms = max_quantum_ms.max(elapsed);
        match output {
            Some(output) => {
                assert_eq!(layer + 1, LAYERS);
                check(
                    "prompt512",
                    &expected,
                    &read_logits(&mut device, output).await?,
                );
            }
            None => assert!(layer + 1 < LAYERS),
        }
        println!("layer={layer} prompt512_quantum_ms={elapsed:.3}");
        if interleave {
            let rows = [input(1, layer, true)];
            let a = model
                .forward_sequences_r64(
                    &mut device,
                    &[Sequence {
                        inputs: &rows,
                        start_position: layer,
                        blocks: &blocks[1],
                    }],
                    &mut workspace,
                )
                .await?;
            let expected = read_logits(&mut device, &a[0]).await?;
            let b = model
                .forward_sequences_r64(
                    &mut device,
                    &[Sequence {
                        inputs: &rows,
                        start_position: layer,
                        blocks: &blocks[62],
                    }],
                    &mut workspace,
                )
                .await?;
            check(
                &format!("interleaved={layer}"),
                &expected,
                &read_logits(&mut device, &b[0]).await?,
            );
        }
    }
    assert!(model.active_prefill_rings().is_empty());
    let profile = model.take_stage_profile();
    for step in 0..4 {
        let rows = [input(0, end + step, true)];
        let a = model
            .forward_sequences(
                &mut device,
                &[Sequence {
                    inputs: &rows,
                    start_position: end + step,
                    blocks: &blocks[0],
                }],
                &mut workspace,
            )
            .await?;
        let expected = read_logits(&mut device, &a[0]).await?;
        let b = model
            .forward_sequences(
                &mut device,
                &[Sequence {
                    inputs: &rows,
                    start_position: end + step,
                    blocks: &blocks[63],
                }],
                &mut workspace,
            )
            .await?;
        check(
            &format!("post_prompt_decode={step}"),
            &expected,
            &read_logits(&mut device, &b[0]).await?,
        );
    }
    for stage in profile {
        println!(
            "stage={} calls={} ms={:.6}",
            stage.stage, stage.calls, stage.milliseconds
        );
    }
    println!(
        "PASS norm fusion Prompt512 start={start} interleave={interleave} original_group_complete_ms={control_ms:.3} prepare_ms={prepare_ms:.3} layers_ms={layer_ms:.3} candidate_complete_ms={:.3} max_quantum_ms={max_quantum_ms:.3}",
        prepare_ms + layer_ms
    );
    Ok(())
}
