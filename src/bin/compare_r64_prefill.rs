//! One canonical pool, paired17+17 decoder slots and prompt slots62/63.
//! Same R64 arithmetic on each pair while a prompt yields after every layer.
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
    assert_eq!(a.len(), b.len());
    let mut n = 0;
    let mut max = 0.0f32;
    for (&a, &b) in a.iter().zip(b) {
        assert!(a.is_finite() && b.is_finite());
        n += usize::from(a != b);
        max = max.max((a - b).abs());
    }
    println!("{label} mismatches={n}/{} max_abs={max}", a.len());
    assert_eq!(n, 0, "interleaving changed same-arithmetic output");
}
#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let start: usize = args.get(1).map_or("896", String::as_str).parse()?;
    let count: usize = args.get(2).map_or("128", String::as_str).parse()?;
    let capacity = (start + count + 64).max(128);
    if !(2..=128).contains(&count) || start / 512 != (start + count - 1) / 512 || capacity > 2048 {
        return Err("page-local prompt2..128, end+64<=2048".into());
    }
    let pages = Workspace::blocks_for(capacity);
    let mut device = Device::new(ops::embed_token.topology())?;
    let weights = load::load_weights(&mut device).await?;
    let mut ws = Workspace::new_canonical64(&mut device, &weights, capacity, 64).await?;
    let mut model = Model::new(&mut device, weights, &mut ws).await?;
    let blocks: Vec<_> = (0..64)
        .map(|slot| Blocks {
            sliding: slot,
            full: (slot * pages..(slot + 1) * pages).collect(),
        })
        .collect();
    // Prompt reference and candidate get identical prefix arithmetic on distinct slots.
    let mut position = 0;
    while position < start {
        let remain = (start - position).min(Tf::SIZE - position % Tf::SIZE);
        let n = if remain >= P128::SIZE {
            P128::SIZE
        } else {
            remain.min(B::SIZE)
        };
        let rows: Vec<_> = (position..position + n)
            .map(|p| input(0, p, false))
            .collect();
        for slot in [62, 63] {
            model
                .forward_sequences(
                    &mut device,
                    &[Sequence {
                        inputs: &rows,
                        start_position: position,
                        blocks: &blocks[slot],
                    }],
                    &mut ws,
                )
                .await?;
        }
        position += n;
    }
    // Both decoder groups start with identical independent histories;17 rows forces R64.
    for pos in 0..2 {
        let rows: Vec<_> = (0..17).map(|r| [input(r + 1, pos, false)]).collect();
        for base in [0, 17] {
            let sequences: Vec<_> = (0..17)
                .map(|r| Sequence {
                    inputs: &rows[r],
                    start_position: pos,
                    blocks: &blocks[base + r],
                })
                .collect();
            model
                .forward_sequences_r64(&mut device, &sequences, &mut ws)
                .await?;
        }
    }
    let prompt: Vec<_> = (start..start + count)
        .map(|p| input(0, p, p + 1 == start + count))
        .collect();
    let control = [Sequence {
        inputs: &prompt,
        start_position: start,
        blocks: &blocks[62],
    }];
    let outputs = if count > 16 && count < 128 {
        model
            .forward_sequences_wide(&mut device, &control, &mut ws)
            .await?
    } else {
        model
            .forward_sequences(&mut device, &control, &mut ws)
            .await?
    };
    let expected_prompt = read_logits(&mut device, &outputs[0]).await?;
    model
        .start_prefill(
            &mut device,
            &Sequence {
                inputs: &prompt,
                start_position: start,
                blocks: &blocks[63],
            },
            &mut ws,
        )
        .await?;
    let mut total = 0.0f64;
    let mut max = 0.0f64;
    for layer in 0..furiosa_opt_gemma4::LAYERS {
        let begin = std::time::Instant::now();
        let output = model.advance_prefill(&mut device, &mut ws).await?;
        let quantum = begin.elapsed().as_secs_f64() * 1000.0;
        total += quantum;
        max = max.max(quantum);
        if let Some(output) = output {
            assert_eq!(layer + 1, furiosa_opt_gemma4::LAYERS);
            check(
                "prompt",
                &expected_prompt,
                &read_logits(&mut device, output).await?,
            );
        }
        let rows: Vec<_> = (0..17).map(|r| [input(r + 1, layer + 2, true)]).collect();
        let reference: Vec<_> = (0..17)
            .map(|r| Sequence {
                inputs: &rows[r],
                start_position: layer + 2,
                blocks: &blocks[r],
            })
            .collect();
        let outputs = model
            .forward_sequences_r64(&mut device, &reference, &mut ws)
            .await?;
        let mut expected = Vec::new();
        for out in outputs {
            expected.push(read_logits(&mut device, out).await?);
        }
        let candidate: Vec<_> = (0..17)
            .map(|r| Sequence {
                inputs: &rows[r],
                start_position: layer + 2,
                blocks: &blocks[r + 17],
            })
            .collect();
        let begin = std::time::Instant::now();
        let outputs = model
            .forward_sequences(&mut device, &candidate, &mut ws)
            .await?;
        let decode = begin.elapsed().as_secs_f64() * 1000.0;
        for (r, out) in outputs.iter().enumerate() {
            check(
                &format!("layer={layer} request={r}"),
                &expected[r],
                &read_logits(&mut device, out).await?,
            );
        }
        println!("layer={layer} quantum_ms={quantum:.3} r64_auto17_ms={decode:.3}");
    }
    assert_eq!(model.active_prefill_ring(), None);
    println!(
        "PASS paired17 R64 automatic decode with resumable prompt start={start} count={count}; quantum_total_ms={total:.3} quantum_max_ms={max:.3}"
    );
    Ok(())
}
