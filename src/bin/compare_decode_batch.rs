//! Actual-model independent-request batch versus serial execution. Each request
//! gets a distinct prompt, position, full-page allocation and sliding-ring slot.
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

fn inputs(request: usize, start: usize, count: usize, logits: bool) -> Vec<Input<'static>> {
    (0..count)
        .map(|r| Input {
            token: 2 + ((request * 313 + (start + r) * 37) % 8000),
            soft: None,
            logits: logits && r + 1 == count,
            image: None,
            audio: None,
        })
        .collect()
}
fn metrics(reference: &[f32], candidate: &[f32]) -> (usize, usize, f32, f64) {
    let top = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    let mut max = 0.0f32;
    let mut error = 0.0f64;
    let mut norm = 0.0f64;
    for (&a, &b) in reference.iter().zip(candidate) {
        assert!(a.is_finite() && b.is_finite());
        max = max.max((a - b).abs());
        error += (a as f64 - b as f64).powi(2);
        norm += (a as f64).powi(2);
    }
    (top(reference), top(candidate), max, (error / norm).sqrt())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let requests: usize = args.get(1).map_or("4", String::as_str).parse()?;
    let prefix: usize = args.get(2).map_or("16", String::as_str).parse()?;
    let steps: usize = args.get(3).map_or("3", String::as_str).parse()?;
    if !(2..=B::SIZE).contains(&requests) || prefix + steps + 2 > 8192 || steps == 0 {
        return Err("usage: requests2..16 prefix steps>0, total<=8192".into());
    }
    let capacity = prefix + steps + 2;
    let pages = Workspace::blocks_for(capacity);
    let mut device = Device::new(ops::embed_token.topology())?;
    let weights = load::load_weights(&mut device).await?;
    let mut serial = Workspace::new(&mut device, &weights, capacity, requests).await?;
    let mut batched = Workspace::new(&mut device, &weights, capacity, requests).await?;
    let blocks: Vec<_> = (0..requests)
        .map(|r| Blocks {
            full: (r * pages..(r + 1) * pages).collect(),
            sliding: r,
        })
        .collect();
    let mut model = Model::new(&mut device, weights, &mut serial).await?;
    for r in 0..requests {
        let length = prefix + r % 3;
        let mut start = 0;
        while start < length {
            let count = (length - start)
                .min(B::SIZE)
                .min(Tf::SIZE - start % Tf::SIZE);
            let rows = inputs(r, start, count, false);
            let sequence = [Sequence {
                inputs: &rows,
                start_position: start,
                blocks: &blocks[r],
            }];
            // Identical serial prefixes isolate the independent-request decode path.
            model
                .forward_sequences_serial(&mut device, &sequence, &mut serial)
                .await?;
            model
                .forward_sequences_serial(&mut device, &sequence, &mut batched)
                .await?;
            start += count;
        }
    }
    for step in 0..steps {
        let rows: Vec<_> = (0..requests)
            .map(|r| inputs(r, prefix + r % 3 + step, 1, true))
            .collect();
        let sequences: Vec<_> = (0..requests)
            .map(|r| Sequence {
                inputs: &rows[r],
                start_position: prefix + r % 3 + step,
                blocks: &blocks[r],
            })
            .collect();
        let now = std::time::Instant::now();
        let logits = model
            .forward_sequences_serial(&mut device, &sequences, &mut serial)
            .await?;
        let serial_ms = now.elapsed().as_secs_f64() * 1000.0;
        let mut reference = Vec::with_capacity(requests);
        for value in logits {
            reference.push(read_logits(&mut device, value).await?);
        }
        model.begin_stage_profile();
        let now = std::time::Instant::now();
        let logits = model
            .forward_sequences(&mut device, &sequences, &mut batched)
            .await?;
        let batch_ms = now.elapsed().as_secs_f64() * 1000.0;
        for (row, value) in logits.iter().enumerate() {
            let candidate = read_logits(&mut device, value).await?;
            let (a, b, max, l2) = metrics(&reference[row], &candidate);
            println!(
                "step={step} row={row} position={} control_top={a} candidate_top={b} max_abs={max} relative_l2={l2}",
                sequences[row].start_position
            );
        }
        for stage in model.take_stage_profile() {
            println!(
                "step={step} stage={} calls={} total_ms={:.3}",
                stage.stage, stage.calls, stage.milliseconds
            );
        }
        println!(
            "step={step} requests={requests} serial_ms={serial_ms:.3} batch_ms={batch_ms:.3} generated_tokens_per_second={:.3}",
            requests as f64 * 1000.0 / batch_ms
        );
    }
    Ok(())
}
