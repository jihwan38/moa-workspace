//! Persistent Q4/H480 residual byte oracle, feedback tests and complete boundary cost.
#[path = "support/npu_profile.rs"]
mod npu_profile;
use furiosa_opt_gemma4::{
    Chip, EPS, axes::*, ops_native_fp8_p512_downblocked::Output,
    ops_norm_blocked_residual as blocked, ops_norm_quarter_post as control,
};
use furiosa_opt_std::prelude::*;
use std::time::Instant;
type Hidden = HbmTensor<bf16, Chip, m![P128, H]>;
fn bf(x: f32) -> f32 {
    f32::from_bits(x.to_bits().wrapping_add(0x7fff + ((x.to_bits() >> 16) & 1)) & 0xffff0000)
}
fn values(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut x = (i as u64)
                .wrapping_add(seed)
                .wrapping_add(0x9e3779b97f4a7c15);
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            x ^= x >> 31;
            bf(((x >> 40) as f32 / 16777216.0 * 2.0 - 1.0) * scale)
        })
        .collect()
}
fn raw(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}
fn residual_index(p: usize, h: usize) -> usize {
    (((((p / 64 % 2) * 32 + p / 2 % 32) * 8 + h / 480) * 4 + p / 128) * 2 + p % 2) * 480 + h % 480
}
fn pack_residual(values: &[Vec<f32>; 4]) -> Vec<u8> {
    let mut out = vec![0xa5; 512 * 3840 * 2];
    let mut seen = vec![false; 512 * 3840];
    for p in 0..512 {
        for h in 0..3840 {
            let i = residual_index(p, h);
            assert!(!seen[i]);
            seen[i] = true;
            out[2 * i..2 * i + 2].copy_from_slice(
                &((values[p / 128][p % 128 * 3840 + h].to_bits() >> 16) as u16).to_le_bytes(),
            );
        }
    }
    assert!(seen.iter().all(|x| *x));
    out
}
fn unpack_cpu(bytes: &[u8], quarter: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(128 * 3840 * 2);
    for p in quarter * 128..(quarter + 1) * 128 {
        for h in 0..3840 {
            let i = residual_index(p, h);
            out.extend_from_slice(&bytes[2 * i..2 * i + 2]);
        }
    }
    out
}
fn pack_output(values: &[Vec<f32>; 4]) -> Vec<u8> {
    let mut out = vec![0xa5; 512 * 3840 * 2];
    let mut seen = vec![false; 512 * 3840];
    for p in 0..512 {
        for h in 0..3840 {
            let i = (((p / 256 * 16 + h / 240) * 256 + p % 256) * 240) + h % 240;
            assert!(!seen[i]);
            seen[i] = true;
            out[2 * i..2 * i + 2].copy_from_slice(
                &((values[p / 128][p % 128 * 3840 + h].to_bits() >> 16) as u16).to_le_bytes(),
            );
        }
    }
    assert!(seen.iter().all(|x| *x));
    out
}
async fn upload<Mp: M>(d: &mut Device, v: Vec<u8>) -> HbmTensor<bf16, Chip, Mp> {
    assert_eq!(v.len(), Mp::SIZE * 2);
    HostTensor::<bf16, Mp>::from_buf(v)
        .to_hbm(&mut d.pdma)
        .await
        .unwrap()
}
async fn bytes<Mp: M>(d: &mut Device, x: &HbmTensor<bf16, Chip, Mp>) -> Vec<u8> {
    let h: HostTensor<bf16, Mp> = x.to_host(&mut d.pdma).await.unwrap();
    h.into_inner().into_buf()
}
fn exact(label: &str, a: &[u8], b: &[u8]) {
    assert_eq!(a.len(), b.len());
    let n = a.iter().zip(b).filter(|(x, y)| x != y).count();
    println!("{label} bytes={} mismatches={n}", a.len());
    assert_eq!(n, 0, "{label}");
}
async fn canonical(d: &mut Device, v: &[Vec<f32>; 4]) -> [Hidden; 4] {
    [
        upload(d, raw(&v[0])).await,
        upload(d, raw(&v[1])).await,
        upload(d, raw(&v[2])).await,
        upload(d, raw(&v[3])).await,
    ]
}
async fn pack(
    d: &mut Device,
    rows: &[Hidden; 4],
    out: &mut HbmTensor<bf16, Chip, blocked::Residual>,
) {
    launch(
        blocked::pack_four,
        (d, &rows[0], &rows[1], &rows[2], &rows[3], out),
    )
    .await
    .unwrap();
}
async fn unpack(
    d: &mut Device,
    input: &HbmTensor<bf16, Chip, blocked::Residual>,
    rows: &mut [Hidden; 4],
) {
    let [a, b, c, e] = rows;
    launch(blocked::unpack_four, (d, input, a, b, c, e))
        .await
        .unwrap();
}
async fn old(
    d: &mut Device,
    input: &HbmTensor<bf16, Chip, Output>,
    weight: &HbmTensor<bf16, Chip, m![H]>,
    gate: &HbmTensor<bf16, Chip, m![1#8]>,
    rows: &mut [Hidden; 4],
) {
    let [a, b, c, e] = rows;
    launch(
        control::post_feedforward_four,
        (d, input, weight, gate, a, b, c, e),
    )
    .await
    .unwrap();
}
async fn new(
    d: &mut Device,
    input: &HbmTensor<bf16, Chip, Output>,
    weight: &HbmTensor<bf16, Chip, m![H]>,
    gate: &HbmTensor<bf16, Chip, m![1#8]>,
    residual: &mut HbmTensor<bf16, Chip, blocked::Residual>,
) {
    launch(
        blocked::post_feedforward,
        (d, input, weight, gate, residual),
    )
    .await
    .unwrap();
}
async fn test(d: &mut Device, seed: u64, order: [usize; 4], gate_value: f32) {
    let pv: [Vec<f32>; 4] = std::array::from_fn(|g| {
        let mut v = values(128 * 3840, seed + order[g] as u64 * 111, 3.0);
        for row in 0..128 {
            if row % 23 == 0 {
                v[row * 3840..(row + 1) * 3840].fill(0.0);
            }
            if row % 23 == 1 {
                v[row * 3840..(row + 1) * 3840].fill(bf(1e-5));
            }
        }
        v
    });
    let rv: [Vec<f32>; 4] =
        std::array::from_fn(|g| values(128 * 3840, seed + order[g] as u64 * 137 + 17, 0.75));
    let producer = pack_output(&pv);
    let residual_bytes = pack_residual(&rv);
    let input = upload(d, producer.clone()).await;
    let wv = values(3840, seed + 1500, 1.75);
    let weight = upload(d, raw(&wv)).await;
    let gate = upload(d, raw(&[bf(gate_value); 8])).await;
    let original = canonical(d, &rv).await;
    let mut packed = upload::<blocked::Residual>(d, vec![0x7f; 512 * 3840 * 2]).await;
    pack(d, &original, &mut packed).await;
    exact(
        "pack exact CPU physical mapping",
        &residual_bytes,
        &bytes(d, &packed).await,
    );
    let mut unpacked: [Hidden; 4] = [
        upload(d, vec![0x7f; 128 * 3840 * 2]).await,
        upload(d, vec![0x7f; 128 * 3840 * 2]).await,
        upload(d, vec![0x7f; 128 * 3840 * 2]).await,
        upload(d, vec![0x7f; 128 * 3840 * 2]).await,
    ];
    unpack(d, &packed, &mut unpacked).await;
    for g in 0..4 {
        exact(
            &format!("unpack all poison overwritten group{g}"),
            &raw(&rv[g]),
            &bytes(d, &unpacked[g]).await,
        );
    }
    let mut rows = canonical(d, &rv).await;
    let mut residual = upload::<blocked::Residual>(d, residual_bytes.clone()).await;
    for step in 0..4 {
        new(d, &input, &weight, &gate, &mut residual).await;
        old(d, &input, &weight, &gate, &mut rows).await;
        let got = bytes(d, &residual).await;
        let (mut error, mut energy) = (0f64, 0f64);
        for g in 0..4 {
            let a = bytes(d, &rows[g]).await;
            let b = unpack_cpu(&got, g);
            exact(&format!("feedback seed{seed} step{step} group{g}"), &a, &b);
            if step == 0 {
                for p in [0, 1, 2, 31, 63, 64, 127] {
                    let rms = (pv[g][p * 3840..(p + 1) * 3840]
                        .iter()
                        .map(|x| (*x as f64).powi(2))
                        .sum::<f64>()
                        / 3840.0
                        + EPS as f64)
                        .sqrt();
                    for h in 0..3840 {
                        let i = p * 3840 + h;
                        let expected = bf(bf(
                            bf((pv[g][i] as f64 / rms * wv[h] as f64) as f32) + rv[g][i]
                        ) * bf(gate_value));
                        let value = f32::from_bits(
                            (u16::from_le_bytes([b[2 * i], b[2 * i + 1]]) as u32) << 16,
                        );
                        assert!(value.is_finite());
                        error += (value as f64 - expected as f64).powi(2);
                        energy += (expected as f64).powi(2);
                    }
                }
            }
        }
        if step == 0 {
            let e = (error / energy.max(1e-30)).sqrt();
            println!("CPU F64 seed={seed} gate={gate_value} nrmse={e}");
            assert!(e < 0.003);
        }
    }
    exact(
        "immutable original FFN output",
        &producer,
        &bytes(d, &input).await,
    );
    unpack(d, &residual, &mut unpacked).await;
    for g in 0..4 {
        exact(
            &format!("final device unpack group{g}"),
            &bytes(d, &rows[g]).await,
            &bytes(d, &unpacked[g]).await,
        );
    }
    let mut timing = [Vec::new(), Vec::new()];
    for variant in [0usize, 1, 1, 0, 0, 1, 1, 0] {
        let elapsed = if variant == 0 {
            rows = canonical(d, &rv).await;
            let now = Instant::now();
            old(d, &input, &weight, &gate, &mut rows).await;
            now.elapsed()
        } else {
            residual = upload(d, residual_bytes.clone()).await;
            let now = Instant::now();
            new(d, &input, &weight, &gate, &mut residual).await;
            now.elapsed()
        };
        timing[variant].push(elapsed.as_secs_f64() * 1e6);
    }
    println!(
        "BLOCKED_RESIDUAL_TIMING seed={seed} gate={gate_value} canonical_us={:?} blocked_us={:?}",
        timing[0], timing[1]
    );
    let (mut packs, mut unpacks, mut totals) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..4 {
        let now = Instant::now();
        pack(d, &original, &mut packed).await;
        let p = now.elapsed().as_secs_f64() * 1e6;
        let middle = Instant::now();
        unpack(d, &packed, &mut unpacked).await;
        let u = middle.elapsed().as_secs_f64() * 1e6;
        packs.push(p);
        unpacks.push(u);
        totals.push(now.elapsed().as_secs_f64() * 1e6);
    }
    println!(
        "BLOCKED_RESIDUAL_BOUNDARY seed={seed} pack_us={packs:?} unpack_us={unpacks:?} total_us={totals:?}"
    );
    if seed == 713 {
        if std::env::var_os("FURIOSA_OPT_PROFILE").is_some() {
            for variant in [0, 1] {
                rows = canonical(d, &rv).await;
                residual = upload(d, residual_bytes.clone()).await;
                println!(
                    "BLOCKED_RESIDUAL_PROFILE_BEGIN variant={}",
                    if variant == 0 { "canonical" } else { "blocked" }
                );
                PROFILE.with(|p| p.enable(true));
                if variant == 0 {
                    old(d, &input, &weight, &gate, &mut rows).await;
                } else {
                    new(d, &input, &weight, &gate, &mut residual).await;
                }
                PROFILE.with(|p| p.enable(false));
            }
        }
        for steps in [1, 4, 48] {
            rows = canonical(d, &rv).await;
            let now = Instant::now();
            for _ in 0..steps {
                old(d, &input, &weight, &gate, &mut rows).await;
            }
            let old_us = now.elapsed().as_secs_f64() * 1e6;
            let mut boundary_rows = canonical(d, &rv).await;
            let now = Instant::now();
            pack(d, &boundary_rows, &mut residual).await;
            for _ in 0..steps {
                new(d, &input, &weight, &gate, &mut residual).await;
            }
            unpack(d, &residual, &mut boundary_rows).await;
            let new_us = now.elapsed().as_secs_f64() * 1e6;
            for g in 0..4 {
                exact(
                    &format!("inclusive chain steps{steps} group{g}"),
                    &bytes(d, &rows[g]).await,
                    &bytes(d, &boundary_rows[g]).await,
                );
            }
            println!(
                "BLOCKED_RESIDUAL_CHAIN steps={steps} canonical_us={old_us} blocked_with_pack_unpack_us={new_us}"
            );
        }
    }
}
thread_local! {static PROFILE:npu_profile::Collector=npu_profile::Collector::default();}
#[tokio::main]
async fn main() {
    PROFILE.with(|p| tracing::subscriber::set_global_default(p.clone()).unwrap());
    let mut d = Device::new(blocked::post_feedforward.topology()).unwrap();
    test(&mut d, 713, [0, 1, 2, 3], 1.25).await;
    test(&mut d, 1771, [3, 1, 0, 2], -0.75).await;
    test(&mut d, 5001, [2, 0, 3, 1], 0.0).await;
    test(&mut d, 9123, [1, 3, 2, 0], 1.0).await;
    println!(
        "PASS persistent H480 residual all CPUphysicalbytes/poison/feedback/BF16/F64/boundary chain; no host adoption"
    );
}
