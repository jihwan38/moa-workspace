//! Same-arithmetic model proof: legacy KV versus one canonical Rc64/Fc256 pool.
//! Args: prompt start (default509), chunk length(default3). Two logical requests
//! use remote physical rings63/31 and pages255 downward in the candidate.
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
fn input(r: usize, p: usize, epoch: usize, logits: bool) -> Input<'static> {
    Input {
        token: 2 + (r * 733 + p * 37 + epoch * 1777) % 8000,
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
    assert_eq!(n, 0, "canonical KV changed model logits");
}
async fn collect(
    model: &mut Model,
    device: &mut Device,
    ws: &mut Workspace,
    seq: &[Sequence<'_>],
    serial: bool,
) -> Result<Vec<Vec<f32>>> {
    let out = if serial {
        model.forward_sequences_serial(device, seq, ws).await?
    } else {
        model.forward_sequences(device, seq, ws).await?
    };
    let mut v = Vec::new();
    for x in out {
        v.push(read_logits(device, x).await?);
    }
    Ok(v)
}
#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let start: usize = args.get(1).map_or("509", String::as_str).parse()?;
    let count: usize = args.get(2).map_or("3", String::as_str).parse()?;
    if !(2..=P128::SIZE).contains(&count)
        || start / Tf::SIZE != (start + count - 1) / Tf::SIZE
        || start + count + 64 > 8192
    {
        return Err("page-local prompt 2..128 ending <=8128 required".into());
    }
    let capacity = (start + count + 64).max(128);
    let pages = Workspace::blocks_for(capacity);
    let mut device = Device::new(ops::embed_token.topology())?;
    let weights = load::load_weights(&mut device).await?;
    let mut control = Workspace::new_reference(&mut device, &weights, capacity, 2).await?;
    let mut candidate = Workspace::new_canonical64(&mut device, &weights, capacity, 64).await?;
    let mut model = Model::new(&mut device, weights, &mut candidate).await?;
    let cb: Vec<_> = (0..2)
        .map(|r| Blocks {
            sliding: r,
            full: (r * pages..(r + 1) * pages).collect(),
        })
        .collect();
    let pb: Vec<_> = [63, 31]
        .into_iter()
        .enumerate()
        .map(|(r, sliding)| Blocks {
            sliding,
            full: (r * pages..(r + 1) * pages).map(|i| 255 - i).collect(),
        })
        .collect();
    for r in 0..2 {
        let end = if r == 0 { start } else { 16 };
        let mut p = 0;
        while p < end {
            let remain = (end - p).min(Tf::SIZE - p % Tf::SIZE);
            let n = if remain >= P128::SIZE {
                P128::SIZE
            } else {
                remain.min(B::SIZE)
            };
            let rows: Vec<_> = (p..p + n).map(|p| input(r, p, 0, false)).collect();
            for (ws, blocks) in [(&mut control, &cb), (&mut candidate, &pb)] {
                model
                    .forward_sequences(
                        &mut device,
                        &[Sequence {
                            inputs: &rows,
                            start_position: p,
                            blocks: &blocks[r],
                        }],
                        ws,
                    )
                    .await?;
            }
            p += n;
        }
    }
    let prompt: Vec<_> = (start..start + count)
        .map(|p| input(0, p, 0, p + 1 == start + count))
        .collect();
    let controls = [Sequence {
        inputs: &prompt,
        start_position: start,
        blocks: &cb[0],
    }];
    let expected_prompt = if count > B::SIZE && count < P128::SIZE {
        let out = model
            .forward_sequences_wide(&mut device, &controls, &mut control)
            .await?;
        read_logits(&mut device, &out[0]).await?
    } else {
        collect(&mut model, &mut device, &mut control, &controls, false)
            .await?
            .remove(0)
    };
    model
        .start_prefill(
            &mut device,
            &Sequence {
                inputs: &prompt,
                start_position: start,
                blocks: &pb[0],
            },
            &mut candidate,
        )
        .await?;
    assert_eq!(model.active_prefill_ring(), Some(63));
    let mut second = 16;
    for layer in 0..furiosa_opt_gemma4::LAYERS {
        let now = std::time::Instant::now();
        let out = model.advance_prefill(&mut device, &mut candidate).await?;
        println!(
            "layer={layer} quantum_ms={:.3}",
            now.elapsed().as_secs_f64() * 1000.0
        );
        if let Some(out) = out {
            assert_eq!(layer + 1, furiosa_opt_gemma4::LAYERS);
            check(
                "prompt",
                &expected_prompt,
                &read_logits(&mut device, out).await?,
            );
        }
        let row = [input(1, second, 0, true)];
        let a = collect(
            &mut model,
            &mut device,
            &mut control,
            &[Sequence {
                inputs: &row,
                start_position: second,
                blocks: &cb[1],
            }],
            true,
        )
        .await?;
        let b = collect(
            &mut model,
            &mut device,
            &mut candidate,
            &[Sequence {
                inputs: &row,
                start_position: second,
                blocks: &pb[1],
            }],
            false,
        )
        .await?;
        check(&format!("interleaved={layer}"), &a[0], &b[0]);
        second += 1;
    }
    assert_eq!(model.active_prefill_ring(), None);
    let mut positions = [start + count, second];
    for step in 0..5 {
        if step == 3 {
            positions[0] = 0;
        }
        let selected: Vec<usize> = if step == 1 { vec![0] } else { vec![0, 1] };
        let rows: Vec<_> = selected
            .iter()
            .map(|&r| {
                [input(
                    r,
                    positions[r],
                    usize::from(step >= 3 && r == 0),
                    true,
                )]
            })
            .collect();
        let aseq: Vec<_> = selected
            .iter()
            .enumerate()
            .map(|(i, &r)| Sequence {
                inputs: &rows[i],
                start_position: positions[r],
                blocks: &cb[r],
            })
            .collect();
        let bseq: Vec<_> = selected
            .iter()
            .enumerate()
            .map(|(i, &r)| Sequence {
                inputs: &rows[i],
                start_position: positions[r],
                blocks: &pb[r],
            })
            .collect();
        let a = collect(&mut model, &mut device, &mut control, &aseq, true).await?;
        let b = collect(&mut model, &mut device, &mut candidate, &bseq, step == 4).await?;
        for (i, &r) in selected.iter().enumerate() {
            check(
                &format!("transition={step} request={r} ring={}", pb[r].sliding),
                &a[i],
                &b[i],
            );
            positions[r] += 1;
        }
    }
    println!(
        "PASS canonical whole-model start={start} count={count}, scalar/prompt/resumable/joint/reset, no cache migrations"
    );
    Ok(())
}
