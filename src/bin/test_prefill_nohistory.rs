//! Deterministic NPU/CPU attention comparisons, page/ring boundaries, and future-key invariance.
use furiosa_opt_gemma4::ops_sliding_wide::Pk128;
use furiosa_opt_gemma4::{
    Chip, axes::*, ops_pool_views::Rc64, ops_request_sliding_r64 as bridge,
    ops_sliding_wide_head_rows as control, ops_sliding_wide_head_rows as candidate,
};
use furiosa_opt_std::prelude::*;
#[path = "support/npu_profile.rs"]
mod npu_profile;
use std::time::Instant;

fn bf(x: f32) -> f32 {
    f32::from_bits((x.to_bits().wrapping_add(0x7fff + ((x.to_bits() >> 16) & 1))) & 0xffff0000)
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
            bf((((x >> 40) as f32) / 16777216.0 * 2.0 - 1.0) * scale)
        })
        .collect()
}
async fn bf_upload<Mp: M>(d: &mut Device, x: &[f32]) -> HbmTensor<bf16, Chip, Mp> {
    assert_eq!(x.len(), Mp::SIZE);
    let bytes: Vec<u8> = x
        .iter()
        .flat_map(|x| ((bf(*x).to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    HostTensor::<bf16, Mp>::from_buf(bytes)
        .to_hbm(&mut d.pdma)
        .await
        .unwrap()
}
async fn f_upload<Mp: M>(d: &mut Device, x: &[f32]) -> HbmTensor<f32, Chip, Mp> {
    assert_eq!(x.len(), Mp::SIZE);
    HostTensor::<f32, Mp>::from_buf(x.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
        .to_hbm(&mut d.pdma)
        .await
        .unwrap()
}
async fn zero<D: ScalarBytes + MaterializableScalar, Mp: M>(
    d: &mut Device,
) -> HbmTensor<D, Chip, Mp> {
    HostTensor::<D, Mp>::from_buf(vec![0u8; Mp::SIZE * D::BITS / 8])
        .to_hbm(&mut d.pdma)
        .await
        .unwrap()
}
async fn read<Mp: M>(d: &mut Device, x: &HbmTensor<bf16, Chip, Mp>) -> Vec<f32> {
    let h: HostTensor<bf16, Mp> = x.to_host(&mut d.pdma).await.unwrap();
    h.into_vec().into_iter().map(bf16::to_f32).collect()
}

fn old_position(start: usize, slot: usize) -> Option<usize> {
    if start == 0 {
        return None;
    }
    let last = start - 1;
    let p = last as isize - ((last + 1024 - slot) % 1024) as isize;
    (p >= 0).then_some(p as usize)
}
fn masks(start: usize, valid: usize, rows: usize, keys: usize) -> (Vec<f32>, Vec<f32>) {
    let old = (0..rows)
        .flat_map(|r| {
            (0..1024).map(move |t| {
                if r < valid && old_position(start, t).is_some_and(|p| p + 1024 > start + r) {
                    0.
                } else {
                    -1e30
                }
            })
        })
        .collect();
    let new = (0..rows)
        .flat_map(|r| {
            (0..keys).map(move |t| {
                if r < valid && t <= r && t < valid {
                    0.
                } else {
                    -1e30
                }
            })
        })
        .collect();
    (old, new)
}
fn metric(label: &str, a: &[f32], b: &[f32], bound: f64) {
    assert_eq!(a.len(), b.len());
    let mut error = 0f64;
    let mut norm = 0f64;
    let mut max = 0f32;
    let mut diffs = 0usize;
    for (&x, &y) in a.iter().zip(b) {
        assert!(x.is_finite() && y.is_finite());
        error += (x as f64 - y as f64).powi(2);
        norm += (y as f64).powi(2);
        max = max.max((x - y).abs());
        diffs += usize::from(x.to_bits() != y.to_bits());
    }
    let nrms = (error / norm.max(1e-30)).sqrt();
    println!(
        "{label} n={} nrms={nrms:.8} max_abs={max} bit_differences={diffs}",
        a.len()
    );
    assert!(nrms < bound, "{label} exceeds numerical gate");
}
fn cpu_samples(
    q: &[f32],
    ok: &[f32],
    ov: &[f32],
    nk: &[f32],
    nv: &[f32],
    start: usize,
    valid: usize,
    got: &[f32],
) {
    let mut rows = vec![0, 1, 15, 16, 63, 64, valid - 1];
    rows.retain(|r| *r < valid);
    rows.sort();
    rows.dedup();
    let mut want = Vec::new();
    let mut selected = Vec::new();
    for r in rows {
        for h in [0usize, 3, 7] {
            for g in 0..2 {
                let qi = (r * 16 + h * 2 + g) * 256;
                let mut entries = Vec::new();
                for t in 0..1024 {
                    if old_position(start, t).is_some_and(|p| p + 1024 > start + r) {
                        let base = (t * 8 + h) * 256;
                        let score = bf((0..256)
                            .map(|j| q[qi + j] as f64 * ok[base + j] as f64)
                            .sum::<f64>() as f32);
                        entries.push((score, &ov[base..base + 256]));
                    }
                }
                for t in 0..=r {
                    let base = (t * 8 + h) * 256;
                    let score = bf((0..256)
                        .map(|j| q[qi + j] as f64 * nk[base + j] as f64)
                        .sum::<f64>() as f32);
                    entries.push((score, &nv[base..base + 256]));
                }
                let max = entries
                    .iter()
                    .map(|(s, _)| *s)
                    .fold(f32::NEG_INFINITY, f32::max);
                let denom = entries
                    .iter()
                    .map(|(s, _)| ((*s - max) as f64).exp())
                    .sum::<f64>()
                    + 1e-6;
                let mut sum = vec![0f64; 256];
                for (s, v) in entries {
                    let w = bf(((s - max) as f64).exp() as f32) as f64;
                    for j in 0..256 {
                        sum[j] += w * v[j] as f64;
                    }
                }
                want.extend(sum.into_iter().map(|x| bf((x / denom) as f32)));
                selected.extend_from_slice(&got[qi..qi + 256]);
            }
        }
    }
    metric(
        &format!("nativeCPU start={start} valid={valid}"),
        &selected,
        &want,
        0.008,
    );
}

async fn idx<Mp: M>(d: &mut Device, v: Vec<i32>) -> HbmTensor<i32, Chip, Mp> {
    HostTensor::<i32, Mp>::from_vec(v)
        .to_hbm(&mut d.pdma)
        .await
        .unwrap()
}
fn exact(label: &str, a: &[f32], b: &[f32]) {
    let count = a
        .iter()
        .zip(b)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    println!("{label}: n={} differences={count}", a.len());
    assert_eq!(count, 0, "{label}");
}
async fn test(
    d: &mut Device,
    pk: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    pv: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    start: usize,
    valid: usize,
    slot: usize,
    sharp: bool,
    profile: &npu_profile::Collector,
) {
    let qv = values(
        128 * 4096,
        100 + slot as u64 * 37,
        if sharp { 1.5 } else { 0.2 },
    );
    let ok = values(1024 * 2048, 200, if sharp { 1.0 } else { 0.2 });
    let ov = values(1024 * 2048, 300, 1.);
    let nkv = values(
        128 * 2048,
        400 + slot as u64 * 53,
        if sharp { 1.0 } else { 0.2 },
    );
    let nvv = values(128 * 2048, 500 + slot as u64 * 73, 1.);
    let q = bf_upload::<m![P128, Ns, Gs, Ds]>(d, &qv).await;
    let k = bf_upload::<m![Ts, Ns, Ds]>(d, &ok).await;
    let v = bf_upload::<m![Ts, Ns, Ds]>(d, &ov).await;
    let nk = bf_upload::<m![P128, Ns, Ds]>(d, &nkv).await;
    let nv = bf_upload::<m![P128, Ns, Ds]>(d, &nvv).await;
    let offset = (slot * 1024 * 2048 * 2) as i32;
    let base = idx::<m![1]>(d, vec![offset]).await;
    let bases = idx::<m![P128 / 64]>(d, vec![offset; 2]).await;
    launch(
        bridge::copy_sliding_cache_to_pool_rc64,
        (d, &k, &v, &base, pk, pv),
    )
    .await
    .unwrap();
    let (om, cm) = masks(start, valid, 128, 128);
    let om = f_upload::<m![P128, Ts]>(d, &om).await;
    let cm = f_upload::<m![P128, Pk128]>(d, &cm).await;
    let mut a = bf_upload::<m![P128,Ns,Gs,Ds]>(d, &vec![f32::NAN;128*4096]).await;
    let mut b = bf_upload::<m![P128,Ns,Gs,Ds]>(d, &vec![f32::NAN;128*4096]).await;
    launch(
        candidate::sliding_attention_first_p128,
        (d, &q, &nk, &nv, &cm, &mut b),
    )
    .await
    .unwrap();
    launch(
        control::sliding_attention_p128,
        (d, &q, pk, pv, &bases, &nk, &nv, &om, &cm, &mut a),
    )
    .await
    .unwrap();
    let want = read(d, &a).await;
    let got = read(d, &b).await;
    exact(
        &format!("first128_vs_full_history slot={slot} start={start} valid={valid} sharp={sharp}"),
        &got[..valid * 4096],
        &want[..valid * 4096],
    );
    // Existing independent F64 sample oracle omits original BF16 value-partial rounding.
    // Use it only for the established flat-distribution regime; sharp is gated against control.
    if !sharp {
        cpu_samples(&qv, &ok, &ov, &nkv, &nvv, start, valid, &got);
    }
    let mut future = nkv.clone();
    let mut futurev = nvv.clone();
    for x in &mut future[(valid - 1) * 2048..] {
        *x = bf(*x * 128. + 16.);
    }
    for x in &mut futurev[(valid - 1) * 2048..] {
        *x = bf(*x * 128. + 1000.);
    }
    let fk = bf_upload::<m![P128, Ns, Ds]>(d, &future).await;
    let fv = bf_upload::<m![P128, Ns, Ds]>(d, &futurev).await;
    launch(
        candidate::sliding_attention_first_p128,
        (d, &q, &fk, &fv, &cm, &mut b),
    )
    .await
    .unwrap();
    exact(
        "future and padded K/V invariant",
        &got[..(valid - 1) * 4096],
        &read(d, &b).await[..(valid - 1) * 4096],
    );
    if start >= 1024 {
        let t = start % 1024;
        let mut deadk = ok.clone();
        let mut deadv = ov.clone();
        deadk[t * 2048..(t + 1) * 2048].fill(128.);
        deadv[t * 2048..(t + 1) * 2048].fill(1024.);
        let dk = bf_upload::<m![Ts, Ns, Ds]>(d, &deadk).await;
        let dv = bf_upload::<m![Ts, Ns, Ds]>(d, &deadv).await;
        launch(
            bridge::copy_sliding_cache_to_pool_rc64,
            (d, &dk, &dv, &base, pk, pv),
        )
        .await
        .unwrap();
        launch(
            candidate::sliding_attention_first_p128,
            (d, &q, &nk, &nv, &cm, &mut b),
        )
        .await
        .unwrap();
        exact(
            "expired old K/V invariant",
            &got[..valid * 4096],
            &read(d, &b).await[..valid * 4096],
        );
        launch(
            bridge::copy_sliding_cache_to_pool_rc64,
            (d, &k, &v, &base, pk, pv),
        )
        .await
        .unwrap();
    }
    let mut ta = Vec::new();
    let mut tb = Vec::new();
    for _ in 0..3 {
        for choose in [false, true, true, false] {
            let begin = Instant::now();
            if choose {
                launch(
                    candidate::sliding_attention_first_p128,
                    (d, &q, &nk, &nv, &cm, &mut b),
                )
                .await
                .unwrap();
                tb.push(begin.elapsed().as_secs_f64() * 1e6);
            } else {
                launch(
                    control::sliding_attention_p128,
                    (d, &q, pk, pv, &bases, &nk, &nv, &om, &cm, &mut a),
                )
                .await
                .unwrap();
                ta.push(begin.elapsed().as_secs_f64() * 1e6);
            }
        }
    }
    println!(
        "TIMING slot={slot} start={start} valid={valid} sharp={sharp} control_us={ta:?} exact_us={tb:?}"
    );
    if slot == 63 && start == 0 && std::env::args().any(|a| a == "--profile") {
        println!("PROFILE CONTROL");
        profile.enable(true);
        launch(
            control::sliding_attention_p128,
            (d, &q, pk, pv, &bases, &nk, &nv, &om, &cm, &mut a),
        )
        .await
        .unwrap();
        profile.enable(false);
        println!("PROFILE FIRST128");
        profile.enable(true);
        launch(
            candidate::sliding_attention_first_p128,
            (d, &q, &nk, &nv, &cm, &mut b),
        )
        .await
        .unwrap();
        profile.enable(false);
    }
    let mut rk = zero::<bf16, m![Ts, Ns, Ds]>(d).await;
    let mut rv = zero::<bf16, m![Ts, Ns, Ds]>(d).await;
    launch(
        furiosa_opt_gemma4::ops_commit_wide_pool::copy_sliding_pool_to_cache_rc64,
        (d, pk, pv, &base, &mut rk, &mut rv),
    )
    .await
    .unwrap();
    exact("old K unchanged", &read(d, &rk).await, &ok);
    exact("old V unchanged", &read(d, &rv).await, &ov);
}
#[tokio::main]
async fn main() {
    let profile = npu_profile::Collector::default();
    tracing::subscriber::set_global_default(profile.clone()).unwrap();
    let mut d = Device::new(candidate::sliding_attention_first_p128.topology()).unwrap();
    let mut pk = zero::<bf16, m![Rc64, Ts, Ns, Ds]>(&mut d).await;
    let mut pv = zero::<bf16, m![Rc64, Ts, Ns, Ds]>(&mut d).await;
    for (start, valid, slot, sharp) in [
        (0, 128, 0, false),
        (0, 128, 31, false),
        (0, 128, 63, false),
        (0, 128, 0, true),
        (0, 128, 31, true),
        (0, 128, 63, true),
    ] {
        test(
            &mut d, &mut pk, &mut pv, start, valid, slot, sharp, &profile,
        )
        .await;
    }
    println!(
        "PASS first128 causal attention all BF16 bits exact, six finite cases with staleKV, flat independent CPU samples, future invariance, immutable rings; model gate pending"
    );
}
