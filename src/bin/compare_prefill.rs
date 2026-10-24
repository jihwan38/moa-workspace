//! Paired whole-model diagnostic with actual model weights and separate KV pools.
//! Optional positional argument is the number of deterministic input tokens.
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
    if !(1..=8192).contains(&count) {
        return Err("expected 1..8192 tokens".into());
    }
    let mut device = Device::new(ops::embed_token.topology())?;
    let weights = load::load_weights(&mut device).await?;
    let mut serial = Workspace::new(&mut device, &weights, count, 1).await?;
    let mut batched = Workspace::new(&mut device, &weights, count, 1).await?;
    let blocks = Blocks {
        full: (0..Workspace::blocks_for(count)).collect(),
        sliding: 0,
    };
    let mut model = Model::new(&mut device, weights, &mut serial).await?;
    let mut control = Vec::new();
    let mut candidate = Vec::new();
    let mut serial_time = std::time::Duration::ZERO;
    let mut batch_time = std::time::Duration::ZERO;
    let mut start = 0;
    while start < count {
        let len = (count - start)
            .min(B::SIZE)
            .min(Tf::SIZE - start % Tf::SIZE);
        let inputs: Vec<_> = (0..len)
            .map(|r| Input {
                token: 2 + ((start + r) * 37 % 8000),
                soft: None,
                logits: start + r + 1 == count,
                image: None,
                audio: None,
            })
            .collect();
        let sequences = [Sequence {
            inputs: &inputs,
            start_position: start,
            blocks: &blocks,
        }];
        let now = std::time::Instant::now();
        let logits = model
            .forward_sequences_serial(&mut device, &sequences, &mut serial)
            .await?;
        serial_time += now.elapsed();
        if start + len == count {
            control = read_logits(&mut device, &logits[0]).await?;
        }
        let now = std::time::Instant::now();
        let logits = model
            .forward_sequences(&mut device, &sequences, &mut batched)
            .await?;
        batch_time += now.elapsed();
        if start + len == count {
            candidate = read_logits(&mut device, &logits[0]).await?;
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
        "tokens={count} serial_ms={:.3} batch_ms={:.3} exact_logits={exact}/{} max_abs_error={max_error} relative_l2={} control_top={} candidate_top={}",
        serial_time.as_secs_f64() * 1000.0,
        batch_time.as_secs_f64() * 1000.0,
        control.len(),
        (squared_error / squared_reference).sqrt(),
        top(&control),
        top(&candidate)
    );
    Ok(())
}
