//! Isolated actual-model numerical comparison: P128 BF16 QKV versus shared native QKV.
//! One canonical pool, disjoint request caches, identical other kernels and teacher-forced tokens.
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
fn input(p: usize, salt: usize, logits: bool) -> Input<'static> {
    Input {
        token: 2 + (p * 37 + salt * 733) % 8000,
        soft: None,
        logits,
        image: None,
        audio: None,
    }
}
fn report(label: &str, control: &[f32], candidate: &[f32]) {
    assert_eq!(control.len(), candidate.len());
    let (mut max_abs, mut e2, mut r2, mut exact) = (0.0f32, 0.0f64, 0.0f64, 0usize);
    for (&a, &b) in control.iter().zip(candidate) {
        assert!(a.is_finite() && b.is_finite());
        max_abs = max_abs.max((a - b).abs());
        e2 += (a as f64 - b as f64).powi(2);
        r2 += (a as f64).powi(2);
        exact += usize::from(a.to_bits() == b.to_bits());
    }
    let top = |v: &[f32]| {
        let mut ids: Vec<_> = (0..v.len()).collect();
        ids.sort_unstable_by(|&a, &b| v[b].total_cmp(&v[a]).then(a.cmp(&b)));
        ids.truncate(10);
        ids
    };
    let a = top(control);
    let b = top(candidate);
    let overlap = a.iter().filter(|i| b.contains(i)).count();
    println!(
        "{label} logits={} exact={exact} max_abs={max_abs} relative_l2={} control_top={} candidate_top={} top10_overlap={overlap}",
        control.len(),
        (e2 / r2.max(1e-300)).sqrt(),
        a[0],
        b[0]
    );
}
#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let count: usize = args.get(1).map_or("513", String::as_str).parse()?;
    let salt: usize = args.get(2).map_or("0", String::as_str).parse()?;
    if !(128..=4096).contains(&count) {
        return Err("expected128..4096 prompt tokens".into());
    }
    let mut d = Device::new(ops::embed_token.topology())?;
    let weights = load::load_weights(&mut d).await?;
    let mut ws = Workspace::new_canonical64(&mut d, &weights, count + 4, 64).await?;
    let pages = Workspace::blocks_for(count + 4);
    let blocks: [Blocks; 2] = std::array::from_fn(|r| Blocks {
        sliding: 62 + r,
        full: (0..pages).map(|p| 255 - (r * pages + p)).collect(),
    });
    let mut model = Model::new(&mut d, weights, &mut ws).await?;
    let mut elapsed = [std::time::Duration::ZERO; 2];
    let mut final_logits = [Vec::new(), Vec::new()];
    let mut p = 0;
    while p < count {
        let remain = (count - p).min(Tf::SIZE - p % Tf::SIZE);
        let n = if remain >= P128::SIZE {
            P128::SIZE
        } else {
            remain.min(B::SIZE)
        };
        let inputs: Vec<_> = (p..p + n).map(|i| input(i, salt, i + 1 == count)).collect();
        for r in 0..2 {
            let seq = [Sequence {
                inputs: &inputs,
                start_position: p,
                blocks: &blocks[r],
            }];
            let now = std::time::Instant::now();
            let logits = if r == 1 && n == P128::SIZE {
                model
                    .forward_sequences_native_qkv(&mut d, &seq, &mut ws)
                    .await?
            } else if n == P128::SIZE {
                model.forward_sequences_bf16_qkv(&mut d, &seq, &mut ws).await?
            } else {
                model.forward_sequences(&mut d, &seq, &mut ws).await?
            };
            elapsed[r] += now.elapsed();
            if p + n == count {
                final_logits[r] = read_logits(&mut d, &logits[0]).await?;
            }
        }
        p += n;
    }
    report("prompt", &final_logits[0], &final_logits[1]);
    println!(
        "tokens={count} salt={salt} bf16_control_ms={:.3} native_qkv_ms={:.3}",
        elapsed[0].as_secs_f64() * 1000.0,
        elapsed[1].as_secs_f64() * 1000.0
    );
    // Same scalar decode math and identical supplied tokens expose the downstream KV delta.
    for step in 0..4 {
        let inputs = [input(count + step, salt, true)];
        for r in 0..2 {
            let logits = model
                .forward_sequences(
                    &mut d,
                    &[Sequence {
                        inputs: &inputs,
                        start_position: count + step,
                        blocks: &blocks[r],
                    }],
                    &mut ws,
                )
                .await?;
            final_logits[r] = read_logits(&mut d, &logits[0]).await?;
        }
        report(
            &format!("teacher_forced_decode{step}"),
            &final_logits[0],
            &final_logits[1],
        );
    }
    Ok(())
}
