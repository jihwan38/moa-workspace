//! Actual-model KV authority checks across many->one->many requests and ring reuse.
//! Optional arguments: request count (2..16), prefix length (default2).
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
fn input(request: usize, epoch: usize, position: usize, logits: bool) -> Input<'static> {
    Input {
        token: 2 + ((request * 313 + epoch * 1777 + position * 37) % 8000),
        soft: None,
        logits,
        image: None,
        audio: None,
    }
}
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let requests: usize = args.get(1).map_or("4", String::as_str).parse()?;
    let prefix: usize = args.get(2).map_or("2", String::as_str).parse()?;
    if !(2..=B::SIZE).contains(&requests) || prefix + 10 > 8192 {
        return Err("requests2..16,prefix<=8182".into());
    }
    let capacity = prefix + 10;
    let pages = Workspace::blocks_for(capacity);
    let mut device = Device::new(ops::embed_token.topology())?;
    let weights = load::load_weights(&mut device).await?;
    let mut control_ws = Workspace::new(&mut device, &weights, capacity, requests).await?;
    let mut candidate_ws = Workspace::new(&mut device, &weights, capacity, requests).await?;
    let blocks: Vec<_> = (0..requests)
        .map(|r| Blocks {
            full: (r * pages..(r + 1) * pages).collect(),
            sliding: r,
        })
        .collect();
    let mut model = Model::new(&mut device, weights, &mut control_ws).await?;
    let mut positions: Vec<_> = (0..requests).map(|r| prefix + r % 3).collect();
    let mut epochs = vec![0; requests];
    for r in 0..requests {
        for position in 0..positions[r] {
            let rows = [input(r, 0, position, false)];
            let sequence = [Sequence {
                inputs: &rows,
                start_position: position,
                blocks: &blocks[r],
            }];
            model
                .forward_sequences_serial(&mut device, &sequence, &mut control_ws)
                .await?;
            model
                .forward_sequences_serial(&mut device, &sequence, &mut candidate_ws)
                .await?;
        }
    }
    for step in 0..6 {
        if step == 3 {
            positions[0] = 0;
            epochs[0] += 1;
        }
        let selected: Vec<_> = if step == 1 {
            vec![0]
        } else {
            (0..requests).collect()
        };
        let rows: Vec<_> = selected
            .iter()
            .map(|&r| [input(r, epochs[r], positions[r], true)])
            .collect();
        let sequences: Vec<_> = selected
            .iter()
            .enumerate()
            .map(|(row, &r)| Sequence {
                inputs: &rows[row],
                start_position: positions[r],
                blocks: &blocks[r],
            })
            .collect();
        let outputs = model
            .forward_sequences_serial(&mut device, &sequences, &mut control_ws)
            .await?;
        let mut expected = Vec::new();
        for logits in outputs {
            expected.push(read_logits(&mut device, logits).await?);
        }
        model.begin_stage_profile();
        let now = std::time::Instant::now();
        let outputs = if step == 5 {
            model
                .forward_sequences_serial(&mut device, &sequences, &mut candidate_ws)
                .await?
        } else {
            model
                .forward_sequences(&mut device, &sequences, &mut candidate_ws)
                .await?
        };
        let elapsed = now.elapsed().as_secs_f64() * 1000.0;
        for (row, logits) in outputs.iter().enumerate() {
            let actual = read_logits(&mut device, logits).await?;
            let mut mismatches = 0;
            let mut max = 0.0f32;
            for (&a, &b) in expected[row].iter().zip(&actual) {
                assert!(a.is_finite() && b.is_finite());
                mismatches += usize::from(a != b);
                max = max.max((a - b).abs());
            }
            println!(
                "step={step} request={} position={} epoch={} mismatches={mismatches}/{} max_abs={max}",
                selected[row],
                positions[selected[row]],
                epochs[selected[row]],
                actual.len()
            );
            assert_eq!(mismatches, 0, "pool transition changed actual-model logits");
        }
        for stage in model.take_stage_profile() {
            println!(
                "step={step} stage={} calls={} total_ms={:.3}",
                stage.stage, stage.calls, stage.milliseconds
            );
        }
        println!(
            "step={step} active={} candidate_ms={elapsed:.3} mode={}",
            selected.len(),
            if step == 5 {
                "forced-legacy"
            } else {
                "automatic"
            }
        );
        for &r in &selected {
            positions[r] += 1;
        }
    }
    Ok(())
}
