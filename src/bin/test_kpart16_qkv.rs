//! Explicit independent and differential projection gates for every H960 wrapper.
use furiosa_opt_gemma4::{
    axes::{H, B, Pf, Ps, Qf, Qs},
    ops_batch_kpart16, ops_batch,
};
use furiosa_opt_std::prelude::*;
fn mix(i: usize, seed: u64) -> u64 {
    let mut x = (i as u64).wrapping_add(seed);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
fn dx(b: usize, k: usize) -> f32 {
    ((mix((b % 41) * H::SIZE + k, 71) % 63) as i32 - 31) as f32 / 32.0
}
fn dw(n: usize, k: usize) -> f32 {
    ((mix((n % 37) * H::SIZE + k, 73) % 33) as i32 - 16) as f32 / 16.0
}
fn varied(i: usize, seed: u64, top: f32) -> f32 {
    let bits = mix(i, seed);
    let signed = ((bits >> 40) as f32 / 16777216.0) * 2.0 - 1.0;
    signed * top * 2f32.powi(-((bits % 9) as i32))
}
fn compare(label: &str, reference: &[bf16], candidate: &[bf16], abs_tol: f32) -> usize {
    let (mut differences, mut failures, mut max_abs, mut max_rel, mut error_sq, mut reference_sq) =
        (0, 0, 0f32, 0f32, 0f64, 0f64);
    for (i, (e, a)) in reference.iter().zip(candidate).enumerate() {
        let (ef, af) = (e.to_f32(), a.to_f32());
        let error = (ef - af).abs();
        differences += usize::from(e.to_bits() != a.to_bits());
        max_abs = max_abs.max(error);
        max_rel = max_rel.max(error / ef.abs().max(0.001));
        error_sq += (error as f64).powi(2);
        reference_sq += (ef as f64).powi(2);
        if !af.is_finite() || error > abs_tol + 0.01 * ef.abs() {
            if failures < 4 {
                eprintln!("{label} i={i} got={af} expected={ef}");
            }
            failures += 1;
        }
    }
    println!(
        "{label} checked={} bitdiff={differences} failures={failures} max_abs={max_abs} max_rel={max_rel} relative_l2={}",
        reference.len(),
        (error_sq / reference_sq.max(1e-30)).sqrt()
    );
    failures
}
macro_rules! probe {
    ($name:ident,$axis:ident,$function:ident) => {
        async fn $name(device: &mut Device) {
            let xs: Vec<bf16> = (0..B::SIZE * H::SIZE)
                .map(|i| bf16::from_f32(dx(i / H::SIZE, i % H::SIZE)))
                .collect();
            let ws: Vec<f8e4m3> = (0..$axis::SIZE * H::SIZE)
                .map(|i| f8e4m3::from_f32(dw(i / H::SIZE, i % H::SIZE)))
                .collect();
            let scales: Vec<bf16> = (0..$axis::SIZE)
                .map(|i| bf16::from_f32(0.25 + (mix(i, 79) % 31) as f32 / 32.0))
                .collect();
            let x = HostTensor::<bf16, m![B, H]>::from_vec(xs.iter().copied())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let w = HostTensor::<f8e4m3, m![$axis, H]>::from_vec(ws.iter().copied())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let s = HostTensor::<bf16, m![$axis]>::from_vec(scales.iter().copied())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let mut out = HostTensor::<bf16, m![B, $axis]>::from_vec(
                (0..B::SIZE * $axis::SIZE).map(|_| bf16::from_f32(f32::NAN)),
            )
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
            let mut dots = vec![0f32; 41 * 37];
            for b in 0..41 {
                for n in 0..37 {
                    dots[b * 37 + n] = (0..H::SIZE).map(|k| dx(b, k) * dw(n, k)).sum();
                }
            }
            let oracle: Vec<bf16> = (0..B::SIZE * $axis::SIZE)
                .map(|i| {
                    bf16::from_f32(
                        bf16::from_f32(dots[(i / $axis::SIZE % 41) * 37 + i % $axis::SIZE % 37])
                            .to_f32()
                            * scales[i % $axis::SIZE].to_f32(),
                    )
                })
                .collect();
            for candidate in [false, true] {
                if candidate {
                    launch(ops_batch_kpart16::$function, (device, &x, &w, &s, &mut out))
                        .await
                        .unwrap();
                } else {
                    launch(ops_batch::$function, (device, &x, &w, &s, &mut out))
                        .await
                        .unwrap();
                }
                let got: HostTensor<bf16, m![B, $axis]> =
                    out.to_host(&mut device.pdma).await.unwrap();
                let got = got.into_vec();
                let failures = oracle
                    .iter()
                    .zip(&got)
                    .filter(|(e, a)| e.to_bits() != a.to_bits())
                    .count();
                println!(
                    "{} dyadic variant={} checked={} exact_failures={failures}",
                    stringify!($axis),
                    if candidate { "H960" } else { "fullH" },
                    oracle.len()
                );
                assert_eq!(failures, 0);
            }
            drop(x);
            drop(w);
            let xs: Vec<bf16> = (0..B::SIZE * H::SIZE)
                .map(|i| bf16::from_f32(varied(i, 83, 4.0)))
                .collect();
            let ws: Vec<f8e4m3> = (0..$axis::SIZE * H::SIZE)
                .map(|i| f8e4m3::from_f32(varied(i, 89, 1.0)))
                .collect();
            let x = HostTensor::<bf16, m![B, H]>::from_vec(xs.iter().copied())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let w = HostTensor::<f8e4m3, m![$axis, H]>::from_vec(ws.iter().copied())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            launch(ops_batch::$function, (device, &x, &w, &s, &mut out))
                .await
                .unwrap();
            let original: HostTensor<bf16, m![B, $axis]> =
                out.to_host(&mut device.pdma).await.unwrap();
            let original = original.into_vec();
            launch(ops_batch_kpart16::$function, (device, &x, &w, &s, &mut out))
                .await
                .unwrap();
            let candidate: HostTensor<bf16, m![B, $axis]> =
                out.to_host(&mut device.pdma).await.unwrap();
            let candidate = candidate.into_vec();
            assert_eq!(
                compare(
                    concat!(stringify!($axis), " random H960-vs-fullH"),
                    &original,
                    &candidate,
                    0.001
                ),
                0
            );
            let (mut expected, mut got0, mut got1) = (Vec::new(), Vec::new(), Vec::new());
            for j in 0..512 {
                let b = (mix(j, 97) as usize) % B::SIZE;
                let n = (mix(j, 101) as usize) % $axis::SIZE;
                let dot: f64 = (0..H::SIZE)
                    .map(|k| {
                        xs[b * H::SIZE + k].to_f32() as f64 * ws[n * H::SIZE + k].to_f32() as f64
                    })
                    .sum();
                expected.push(bf16::from_f32(
                    bf16::from_f32(dot as f32).to_f32() * scales[n].to_f32(),
                ));
                got0.push(original[b * $axis::SIZE + n]);
                got1.push(candidate[b * $axis::SIZE + n]);
            }
            assert_eq!(
                compare(
                    concat!(stringify!($axis), " random fullH-vs-F64scalar"),
                    &expected,
                    &got0,
                    0.001
                ),
                0
            );
            assert_eq!(
                compare(
                    concat!(stringify!($axis), " random H960-vs-F64scalar"),
                    &expected,
                    &got1,
                    0.001
                ),
                0
            );
            let mut times=[Vec::<f64>::new(),Vec::<f64>::new()];
            for variant in [0,1,1,0] {
                let start=std::time::Instant::now();
                for _ in 0..5 {
                    if variant==0 {launch(ops_batch::$function,(device,&x,&w,&s,&mut out)).await.unwrap();}
                    else {launch(ops_batch_kpart16::$function,(device,&x,&w,&s,&mut out)).await.unwrap();}
                }
                let _:HostTensor<bf16,m![B,$axis]>=out.to_host(&mut device.pdma).await.unwrap();
                times[variant].push(start.elapsed().as_secs_f64()*1000.0/5.0);
            }
            println!("{} timing original_replicated_B16_ms={:?} H960_split_cluster_ms={:?} scope=5launches+1read_divided5",stringify!($axis),times[0],times[1]);
        }
    };
}
probe!(query, Qs, project_query_batch);
probe!(sliding_kv, Ps, project_sliding_kv_batch);
probe!(full_query, Qf, project_full_query_batch);
probe!(full_key, Pf, project_full_key_batch);
#[tokio::main]
async fn main() {
    let mut device = Device::new(ops_batch_kpart16::project_query_batch.topology()).unwrap();
    query(&mut device).await;
    sliding_kv(&mut device).await;
    full_query(&mut device).await;
    full_key(&mut device).await;
}
