//! Actual-weight native P128 attribution after a separate-request warmup.
//! Args: starting absolute position(default0), must be divisible by128.
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
fn inputs(start: usize, logits: bool) -> Vec<Input<'static>> {
    (0..P128::SIZE)
        .map(|r| Input {
            token: 2 + ((start + r) * 37 % 8000),
            soft: None,
            logits: logits && r + 1 == P128::SIZE,
            image: None,
            audio: None,
        })
        .collect()
}
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "0".into())
        .parse::<usize>()?;
    if start % P128::SIZE != 0 || start + P128::SIZE > E::SIZE {
        return Err("need a page-local P128 start within context".into());
    }
    let capacity = start + P128::SIZE;
    let pages = Workspace::blocks_for(capacity);
    let mut d = Device::new(ops::embed_token.topology())?;
    let weights = load::load_weights(&mut d).await?;
    let mut ws = Workspace::new_canonical64(&mut d, &weights, capacity, 64).await?;
    let mut model = Model::new(&mut d, weights, &mut ws).await?;
    let warm = Blocks {
        sliding: 0,
        full: vec![0],
    };
    let current = Blocks {
        sliding: 63,
        full: (256 - pages..256).rev().collect(),
    };
    let rows = inputs(0, false);
    model
        .forward_sequences(
            &mut d,
            &[Sequence {
                inputs: &rows,
                start_position: 0,
                blocks: &warm,
            }],
            &mut ws,
        )
        .await?;
    for p in (0..start).step_by(P128::SIZE) {
        let rows = inputs(p, false);
        model
            .forward_sequences(
                &mut d,
                &[Sequence {
                    inputs: &rows,
                    start_position: p,
                    blocks: &current,
                }],
                &mut ws,
            )
            .await?;
    }
    let rows = inputs(start, true);
    model.begin_stage_profile();
    let now = std::time::Instant::now();
    let output = model
        .forward_sequences(
            &mut d,
            &[Sequence {
                inputs: &rows,
                start_position: start,
                blocks: &current,
            }],
            &mut ws,
        )
        .await?;
    let total = now.elapsed().as_secs_f64() * 1000.0;
    let logits = read_logits(&mut d, &output[0]).await?;
    assert!(logits.iter().all(|x| x.is_finite()));
    let mut sum = 0.0;
    for stage in model.take_stage_profile() {
        sum += stage.milliseconds;
        println!(
            "stage={} calls={} ms={:.6}",
            stage.stage, stage.calls, stage.milliseconds
        );
    }
    println!(
        "P128 start={start} model_ms={total:.6} attributed_ms={sum:.6} outside_stages_ms={:.6}",
        total - sum
    );
    Ok(())
}
