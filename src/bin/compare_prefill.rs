//! Actual48-layer paired diagnostic with independent KV pools.
//! `compare_prefill TOKENS` compares B16 with scalar. `TOKENS wide` compares
//! P128 with B16 chunks, isolating wide matrix and causal-segment ordering.
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let count: usize = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "16".into())
        .parse()?;
    let wide = std::env::args().nth(2).is_some_and(|mode| mode == "wide");
    if !(1..=8192).contains(&count) {
        return Err("expected 1..8192 tokens".into());
    }
    let chunk_size = if wide { P128::SIZE } else { B::SIZE };
    let mut device = Device::new(ops::embed_token.topology())?;
    let weights = load::load_weights(&mut device).await?;
    let mut control_ws = Workspace::new(&mut device, &weights, count, 1).await?;
    let mut candidate_ws = Workspace::new(&mut device, &weights, count, 1).await?;
    let blocks = Blocks {
        full: (0..Workspace::blocks_for(count)).collect(),
        sliding: 0,
    };
    let mut model = Model::new(&mut device, weights, &mut control_ws).await?;
    let mut control = Vec::new();
    let mut candidate = Vec::new();
    let mut control_time = std::time::Duration::ZERO;
    let mut candidate_time = std::time::Duration::ZERO;
    let mut stage_totals = std::collections::BTreeMap::<&'static str, (usize, f64)>::new();
    let mut start = 0;
    while start < count {
        let len = (count - start)
            .min(chunk_size)
            .min(Tf::SIZE - start % Tf::SIZE);
        let inputs: Vec<_> = (0..len)
            .map(|row| Input {
                token: 2 + ((start + row) * 37 % 8000),
                soft: None,
                logits: start + row + 1 == count,
                image: None,
                audio: None,
            })
            .collect();
        let sequences = [Sequence {
            inputs: &inputs,
            start_position: start,
            blocks: &blocks,
        }];
        if wide {
            for offset in (0..len).step_by(B::SIZE) {
                let end = (offset + B::SIZE).min(len);
                let rows = [Sequence {
                    inputs: &inputs[offset..end],
                    start_position: start + offset,
                    blocks: &blocks,
                }];
                let now = std::time::Instant::now();
                let logits = model
                    .forward_sequences(&mut device, &rows, &mut control_ws)
                    .await?;
                control_time += now.elapsed();
                if start + end == count {
                    control = read_logits(&mut device, &logits[0]).await?;
                }
            }
        } else {
            let now = std::time::Instant::now();
            let logits = model
                .forward_sequences_serial(&mut device, &sequences, &mut control_ws)
                .await?;
            control_time += now.elapsed();
            if start + len == count {
                control = read_logits(&mut device, &logits[0]).await?;
            }
        }
        model.begin_stage_profile();
        let now = std::time::Instant::now();
        let logits = if wide {
            model
                .forward_sequences_wide(&mut device, &sequences, &mut candidate_ws)
                .await?
        } else {
            model
                .forward_sequences(&mut device, &sequences, &mut candidate_ws)
                .await?
        };
        candidate_time += now.elapsed();
        if start + len == count {
            candidate = read_logits(&mut device, &logits[0]).await?;
        }
        for stage in model.take_stage_profile() {
            let entry = stage_totals.entry(stage.stage).or_default();
            entry.0 += stage.calls;
            entry.1 += stage.milliseconds;
        }
        start += len;
    }
    let top = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    let mut max_error = 0.0f32;
    let mut squared_error = 0.0f64;
    let mut squared_reference = 0.0f64;
    let mut exact = 0;
    for (&a, &b) in control.iter().zip(&candidate) {
        assert!(a.is_finite() && b.is_finite());
        max_error = max_error.max((a - b).abs());
        squared_error += (a as f64 - b as f64).powi(2);
        squared_reference += (a as f64).powi(2);
        exact += usize::from(a == b);
    }
    println!(
        "tokens={count} control={} control_ms={:.3} candidate_ms={:.3} exact_logits={exact}/{} max_abs_error={max_error} relative_l2={} control_top={} candidate_top={}",
        if wide { "b16" } else { "scalar" },
        control_time.as_secs_f64() * 1000.0,
        candidate_time.as_secs_f64() * 1000.0,
        control.len(),
        (squared_error / squared_reference).sqrt(),
        top(&control),
        top(&candidate)
    );
    for (stage, (calls, milliseconds)) in stage_totals {
        println!("stage={stage} calls={calls} total_ms={milliseconds:.3}");
    }
    Ok(())
}
