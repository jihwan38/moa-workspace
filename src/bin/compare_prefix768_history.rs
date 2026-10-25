//! Same-R64-arithmetic original attention versus causal768 on real independent histories.
//! Sequential passes reuse one Model/pool and retain all control BF16 logits on CPU.
use furiosa_opt_gemma4::{
    LAYERS,
    axes::*,
    host::{
        load,
        model::{Input, Logits, Model, Sequence},
        runtime::{Blocks, Workspace},
    },
    ops,
};
use furiosa_opt_std::prelude::*;
use std::time::Instant;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const N: usize = 64;
fn input(request: usize, position: usize, logits: bool) -> Input<'static> {
    Input {
        token: 2 + (request * 313 + position * 37) % 8000,
        soft: None,
        logits,
        image: None,
        audio: None,
    }
}
fn prefix(mode: &str) -> Vec<usize> {
    match mode {
        "uniform" => vec![512; N],
        "mixed-low" => (0..32).map(|r| [511, 512, 767, 639][r / 8]).collect(),
        "mixed-high" => (0..32).map(|r| [768, 1023, 1024, 0][r / 8]).collect(),
        "mixed-edge" => {
            let mut p = vec![512; N];
            for (r, v) in [
                (0, 511),
                (8, 767),
                (16, 768),
                (24, 1023),
                (32, 1024),
                (40, 0),
                (48, 639),
            ] {
                p[r] = v;
            }
            p
        }
        _ => unreachable!(),
    }
}
fn selected(mode: &str, step: usize) -> Vec<usize> {
    match mode {
        "uniform" => match step {
            0 => (0..N).collect(),
            1 => (0..N).rev().collect(),
            2 => vec![63, 7, 8, 31, 32, 0, 16],
            3 => (0..N).map(|r| (r * 17 + 5) % N).collect(),
            _ => unreachable!(),
        },
        "mixed-low" => match step {
            0 => (0..32).collect(),
            1 => (8..16).chain(24..32).collect(),
            2 => (0..8).collect(),
            3 => (16..24).collect(),
            4 => (0..32).map(|r| (r * 17 + 5) % 32).collect(),
            _ => unreachable!(),
        },
        "mixed-high" => match step {
            0 => (0..32).collect(),
            1 => (0..8).collect(),
            2 => (8..16).collect(),
            3 => (24..32).collect(),
            4 => (0..32).map(|r| (r * 17 + 5) % 32).collect(),
            _ => unreachable!(),
        },
        "mixed-edge" => match step {
            0 => (0..N).filter(|r| ![16, 24, 32].contains(r)).collect(),
            1 => (0..N).collect(),
            2 => (0..N).filter(|r| ![8, 16, 24, 32].contains(r)).collect(),
            3 => vec![63, 7, 8, 31, 32, 0, 16],
            4 => (0..N).map(|r| (r * 17 + 5) % N).collect(),
            _ => unreachable!(),
        },
        _ => unreachable!(),
    }
}
fn eligible(positions: &[usize]) -> bool {
    positions.len() > 56
        && positions.iter().all(|&p| p < 768)
        && positions.chunks(8).all(|g| g.iter().any(|&p| p >= 512))
}
async fn fill(
    device: &mut Device,
    model: &mut Model,
    ws: &mut Workspace,
    blocks: &[Blocks],
    prefix: &[usize],
    phase: usize,
) -> Result<()> {
    let begin = Instant::now();
    for request in 0..prefix.len() {
        let mut position = 0;
        while position < prefix[request] {
            let remain = prefix[request] - position;
            let count = if remain >= 512 {
                512
            } else if remain >= 128 {
                128
            } else {
                remain.min(16)
            };
            let inputs: Vec<_> = (position..position + count)
                .map(|p| input(request, p, false))
                .collect();
            let sequence = Sequence {
                inputs: &inputs,
                start_position: position,
                blocks: &blocks[request],
            };
            if count == 512 {
                model.start_prefill512(device, &sequence, ws).await?;
                for layer in 0..LAYERS {
                    let done = model.advance_prefill(device, ws).await?.is_some();
                    assert_eq!(done, layer + 1 == LAYERS);
                }
            } else {
                model.forward_sequences(device, &[sequence], ws).await?;
            }
            position += count;
        }
    }
    println!(
        "PREFIX768_HISTORY phase={phase} independent_tokens={} fill_ms={:.3}",
        prefix.iter().sum::<usize>(),
        begin.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}
async fn read(device: &mut Device, output: &Logits) -> Result<Vec<u8>> {
    Ok(output
        .to_host::<m![W]>(&mut device.pdma)
        .await?
        .into_inner()
        .into_buf())
}
fn check(actual: &[u8], expected: &[u8], step: usize, request: usize) {
    assert_eq!(actual.len(), W::SIZE * 2);
    assert_eq!(actual.len(), expected.len());
    let mut differences = 0;
    for (token, (a, b)) in actual
        .chunks_exact(2)
        .zip(expected.chunks_exact(2))
        .enumerate()
    {
        let a = u16::from_le_bytes([a[0], a[1]]);
        let b = u16::from_le_bytes([b[0], b[1]]);
        assert_ne!(
            a & 0x7f80,
            0x7f80,
            "nonfinite step={step} request={request} token={token}"
        );
        if a != b {
            if differences < 4 {
                eprintln!(
                    "PREFIX768_DIFF step={step} request={request} token={token} actual={a:04x} expected={b:04x}"
                );
            }
            differences += 1;
        }
    }
    println!(
        "PREFIX768_MODEL_EXACT step={step} request={request} checked={} differences={differences}",
        W::SIZE
    );
    assert_eq!(
        differences, 0,
        "same-schedule original/prefix768 logits changed"
    );
}
#[tokio::main]
async fn main() -> Result<()> {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "uniform".into());
    assert!(["uniform", "mixed-low", "mixed-high", "mixed-edge"].contains(&mode.as_str()));

    let total = Instant::now();
    let mut device = Device::new(ops::embed_token.topology())?;
    let weights = load::load_weights(&mut device).await?;
    let mut ws = Workspace::new_canonical64(&mut device, &weights, 2048, N).await?;
    let mut model = Model::new(&mut device, weights, &mut ws).await?;
    println!(
        "PREFIX768_STARTUP mode={mode} ms={:.3}",
        total.elapsed().as_secs_f64() * 1000.0
    );
    let prefix = prefix(&mode);
    let steps = if mode == "uniform" { 4 } else { 5 };
    let mut expected: Vec<Vec<Vec<u8>>> = Vec::new();
    let mut checked = 0;
    for phase in 0..2 {
        // Reuse the same physical pool with a different bijective ring/page map.
        // Every request recomputes its own entire causal prefix from position0.
        let blocks: Vec<_> = (0..prefix.len())
            .map(|r| {
                let slot = if phase == 0 {
                    (r * 13 + 31) % N
                } else {
                    (r * 17 + 5) % N
                };
                Blocks {
                    sliding: slot,
                    full: (0..3).map(|p| 255 - slot * 3 - p).collect(),
                }
            })
            .collect();
        fill(&mut device, &mut model, &mut ws, &blocks, &prefix, phase).await?;
        let mut positions = prefix.clone();
        for step in 0..steps {
            let chosen = selected(&mode, step);
            let rows: Vec<_> = chosen
                .iter()
                .map(|&r| [input(r, positions[r], true)])
                .collect();
            let sequences: Vec<_> = chosen
                .iter()
                .enumerate()
                .map(|(row, &r)| Sequence {
                    inputs: &rows[row],
                    start_position: positions[r],
                    blocks: &blocks[r],
                })
                .collect();
            let pos: Vec<_> = chosen.iter().map(|&r| positions[r]).collect();
            let use768 = eligible(&pos);
            let expected_policy = match mode.as_str() {
                "uniform" => step != 2,
                "mixed-edge" => matches!(step, 0 | 2),
                _ => false,
            };
            assert_eq!(
                use768, expected_policy,
                "production eligibility step={step}"
            );
            model.begin_stage_profile();
            let begin = Instant::now();
            let outputs = if phase == 0 {
                model
                    .forward_sequences_r64_original_sliding(&mut device, &sequences, &mut ws)
                    .await?
            } else {
                model
                    .forward_sequences_r64(&mut device, &sequences, &mut ws)
                    .await?
            };
            let elapsed = begin.elapsed().as_secs_f64() * 1000.0;
            let mut control = Vec::new();
            for (row, output) in outputs.iter().enumerate() {
                let bytes = read(&mut device, output).await?;
                if phase == 0 {
                    control.push(bytes);
                } else {
                    check(&bytes, &expected[step][row], step, chosen[row]);
                    checked += W::SIZE;
                }
            }
            if phase == 0 {
                expected.push(control);
            }
            println!(
                "PREFIX768_MODEL_TIMING mode={mode} phase={phase} step={step} live={} eligible768={use768} model_ms={elapsed:.3} positions={pos:?}",
                chosen.len()
            );
            for stage in model.take_stage_profile() {
                if stage.stage == "r64.sliding.attention" {
                    println!(
                        "PREFIX768_MODEL_STAGE phase={phase} step={step} name={} calls={} ms={:.6}",
                        stage.stage, stage.calls, stage.milliseconds
                    );
                }
            }
            for r in chosen {
                positions[r] += 1;
            }
        }
    }
    let expected_rows = match mode.as_str() {
        "uniform" => 199,
        "mixed-low" => 96,
        "mixed-high" => 88,
        "mixed-edge" => 256,
        _ => unreachable!(),
    };
    assert_eq!(checked, expected_rows * W::SIZE);
    println!(
        "PASS prefix768 same-R64 history mode={mode} all_logits={checked} independent_refill_and_slot_reset=true total_ms={:.3}",
        total.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}
