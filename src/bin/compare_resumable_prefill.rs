//! Compare uninterrupted wide prefill with one-layer quanta interleaved with
//! independent decode. Args: decode requests(default2), prompt start(default0),
//! chunk length(default128). All logits and a subsequent pooled step must match.
//! Decode controls use the same pooled arithmetic in an independent workspace.
//! The scalar E154 versus H960 batched path is a separate numerical comparison.
use furiosa_opt_gemma4::{
    axes::*,
    host::{
        generate::read_logits,
        load,
        model::{Input, Model, Sequence},
        runtime::{Blocks, Workspace},
    },
    ops,
};
use furiosa_opt_std::prelude::*;
fn input(request: usize, position: usize, logits: bool) -> Input<'static> {
    Input {
        token: 2 + (request * 733 + position * 37) % 8000,
        soft: None,
        logits,
        image: None,
        audio: None,
    }
}
fn check(label: &str, expected: &[f32], actual: &[f32]) {
    assert_eq!(expected.len(), actual.len());
    let mut mismatches = 0;
    let mut max = 0.0f32;
    for (&a, &b) in expected.iter().zip(actual) {
        assert!(a.is_finite() && b.is_finite());
        mismatches += usize::from(a != b);
        max = max.max((a - b).abs());
    }
    println!(
        "{label} mismatches={mismatches}/{} max_abs={max}",
        actual.len()
    );
    assert_eq!(mismatches, 0, "resuming prefill changed logits");
}
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let decoders: usize = args.get(1).map_or("2", String::as_str).parse()?;
    let start: usize = args.get(2).map_or("0", String::as_str).parse()?;
    let count: usize = args.get(3).map_or("128", String::as_str).parse()?;
    if !(1..B::SIZE).contains(&decoders)
        || !(2..=P128::SIZE).contains(&count)
        || start / Tf::SIZE != (start + count - 1) / Tf::SIZE
        || start + count + 64 > 8192
    {
        return Err("decoders1..15; page-local prompt2..128; end<=8128".into());
    }
    let requests = decoders + 1;
    let capacity = start + count + 64;
    let pages = Workspace::blocks_for(capacity);
    let mut device = Device::new(ops::embed_token.topology())?;
    let weights = load::load_weights(&mut device).await?;
    let layers = furiosa_opt_gemma4::LAYERS;
    let mut control_ws = Workspace::new(&mut device, &weights, capacity, requests).await?;
    let mut candidate_ws = Workspace::new(&mut device, &weights, capacity, requests).await?;
    // Original scalar prompt arithmetic is an independent numerical diagnostic.
    // Keep its KV separate: replaying into the same sliding ring could overwrite
    // history needed by an earlier query near a wrap. One request skips pool allocation.
    let mut scalar_ws = if count <= B::SIZE {
        Some(Workspace::new(&mut device, &weights, capacity, 1).await?)
    } else {
        None
    };
    let blocks: Vec<_> = (0..requests)
        .map(|r| Blocks {
            full: (r * pages..(r + 1) * pages).collect(),
            sliding: r,
        })
        .collect();
    let mut model = Model::new(&mut device, weights, &mut control_ws).await?;
    for r in 0..requests {
        let prefix = if r == 0 { start } else { 2 + r % 3 };
        let mut position = 0;
        while position < prefix {
            let remaining = (prefix - position).min(Tf::SIZE - position % Tf::SIZE);
            let len = if remaining >= P128::SIZE {
                P128::SIZE
            } else {
                remaining.min(B::SIZE)
            };
            let rows: Vec<_> = (position..position + len)
                .map(|pos| input(r, pos, false))
                .collect();
            let sequences = [Sequence {
                inputs: &rows,
                start_position: position,
                blocks: &blocks[r],
            }];
            model
                .forward_sequences(&mut device, &sequences, &mut control_ws)
                .await?;
            model
                .forward_sequences(&mut device, &sequences, &mut candidate_ws)
                .await?;
            if r == 0 {
                if let Some(scalar_ws) = &mut scalar_ws {
                    model
                        .forward_sequences(&mut device, &sequences, scalar_ws)
                        .await?;
                }
            }
            position += len;
        }
    }
    let prompt: Vec<_> = (start..start + count)
        .map(|position| input(0, position, position + 1 == start + count))
        .collect();
    let sequence = Sequence {
        inputs: &prompt,
        start_position: start,
        blocks: &blocks[0],
    };
    let scalar_prompt = if let Some(scalar_ws) = &mut scalar_ws {
        let outputs = model
            .forward_sequences_serial(&mut device, std::slice::from_ref(&sequence), scalar_ws)
            .await?;
        Some(read_logits(&mut device, &outputs[0]).await?)
    } else {
        None
    };
    let begin = std::time::Instant::now();
    let outputs = if count <= B::SIZE {
        model
            .forward_sequences(
                &mut device,
                std::slice::from_ref(&sequence),
                &mut control_ws,
            )
            .await?
    } else {
        model
            .forward_sequences_wide(
                &mut device,
                std::slice::from_ref(&sequence),
                &mut control_ws,
            )
            .await?
    };
    let control_ms = begin.elapsed().as_secs_f64() * 1000.0;
    let expected_prompt = read_logits(&mut device, &outputs[0]).await?;
    if let Some(scalar) = &scalar_prompt {
        assert!(scalar.iter().all(|value| value.is_finite()));
        let mismatches = scalar
            .iter()
            .zip(&expected_prompt)
            .filter(|(a, b)| a != b)
            .count();
        let max_abs = scalar
            .iter()
            .zip(&expected_prompt)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let diff_sq: f64 = scalar
            .iter()
            .zip(&expected_prompt)
            .map(|(a, b)| f64::from(a - b).powi(2))
            .sum();
        let ref_sq: f64 = scalar.iter().map(|a| f64::from(*a).powi(2)).sum();
        println!(
            "reported_only_scalar_vs_b16 mismatches={mismatches}/{} max_abs={max_abs} relative_l2={:.8} prefix=shared_b16",
            scalar.len(),
            (diff_sq / ref_sq.max(1e-30)).sqrt()
        );
    }
    let prepare = std::time::Instant::now();
    model
        .start_prefill(&mut device, &sequence, &mut candidate_ws)
        .await?;
    let prepare_ms = prepare.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(model.active_prefill_ring(), Some(0));
    assert!(
        model
            .start_prefill(&mut device, &sequence, &mut candidate_ws)
            .await
            .is_err()
    );
    assert!(
        model
            .forward_sequences_serial(
                &mut device,
                std::slice::from_ref(&sequence),
                &mut candidate_ws
            )
            .await
            .is_err()
    );
    let unrelated_prompt = [input(1, 3, false), input(1, 4, true)];
    let unrelated_sequence = [Sequence {
        inputs: &unrelated_prompt,
        start_position: 3,
        blocks: &blocks[1],
    }];
    assert!(
        model
            .forward_sequences(&mut device, &unrelated_sequence, &mut candidate_ws)
            .await
            .is_err(),
        "another batched prompt must not overwrite a pending checkpoint"
    );
    let mut positions: Vec<_> = (0..requests)
        .map(|r| if r == 0 { start + count } else { 2 + r % 3 })
        .collect();
    let mut layer_times = Vec::new();
    for layer in 0..layers {
        let now = std::time::Instant::now();
        let output = model
            .advance_prefill(&mut device, &mut candidate_ws)
            .await?;
        let layer_ms = now.elapsed().as_secs_f64() * 1000.0;
        layer_times.push(layer_ms);
        if let Some(output) = output {
            assert_eq!(layer + 1, layers);
            check(
                "prompt",
                &expected_prompt,
                &read_logits(&mut device, output).await?,
            );
        } else {
            assert!(layer + 1 < layers);
        }
        let rows: Vec<_> = (1..requests)
            .map(|r| [input(r, positions[r], true)])
            .collect();
        let sequences: Vec<_> = (1..requests)
            .map(|r| Sequence {
                inputs: &rows[r - 1],
                start_position: positions[r],
                blocks: &blocks[r],
            })
            .collect();
        let controls = model
            .forward_sequences(&mut device, &sequences, &mut control_ws)
            .await?;
        let mut expected = Vec::new();
        for control in controls {
            expected.push(read_logits(&mut device, control).await?);
        }
        let now = std::time::Instant::now();
        let outputs = model
            .forward_sequences(&mut device, &sequences, &mut candidate_ws)
            .await?;
        let decode_ms = now.elapsed().as_secs_f64() * 1000.0;
        for (row, output) in outputs.iter().enumerate() {
            check(
                &format!("layer={layer} request={}", row + 1),
                &expected[row],
                &read_logits(&mut device, output).await?,
            );
        }
        for position in positions.iter_mut().skip(1) {
            *position += 1;
        }
        println!("layer={layer} quantum_ms={layer_ms:.3} decode_ms={decode_ms:.3}");
    }
    assert_eq!(model.active_prefill_ring(), None);
    assert!(
        model
            .advance_prefill(&mut device, &mut candidate_ws)
            .await
            .is_err()
    );
    let rows: Vec<_> = (0..requests)
        .map(|r| [input(r, positions[r], true)])
        .collect();
    let sequences: Vec<_> = (0..requests)
        .map(|r| Sequence {
            inputs: &rows[r],
            start_position: positions[r],
            blocks: &blocks[r],
        })
        .collect();
    let outputs = model
        .forward_sequences(&mut device, &sequences, &mut control_ws)
        .await?;
    let mut expected = Vec::new();
    for output in outputs {
        expected.push(read_logits(&mut device, output).await?);
    }
    let outputs = model
        .forward_sequences(&mut device, &sequences, &mut candidate_ws)
        .await?;
    for (row, output) in outputs.iter().enumerate() {
        check(
            &format!("after-prefill request={row}"),
            &expected[row],
            &read_logits(&mut device, output).await?,
        );
    }
    println!(
        "decoders={decoders} start={start} count={count} control_ms={control_ms:.3} prepare_ms={prepare_ms:.3} quantum_total_ms={:.3} quantum_max_ms={:.3}",
        layer_times.iter().sum::<f64>(),
        layer_times.iter().copied().fold(0.0f64, f64::max)
    );
    Ok(())
}
