//! Direct E4 old-ring read: exact owned-BF16 oracle, CPU samples, causal invariance, ABBA.
#![feature(register_tool)]
#![register_tool(furiosa_opt)]
use furiosa_opt_gemma4::{
    Chip, axes::*, ops_kv_e4_pool as pool, ops_pool_views::Rc64,
    ops_request_sliding_r64_fp8_scales::Scale8, ops_sliding_wide as owned,
    ops_sliding_wide_e4_pool as candidate,
};
use furiosa_opt_std::prelude::*;
use owned::Pk128;
use std::time::Instant;
const RING: usize = 1024 * 8 * 256;
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

fn word(i: usize, seed: u64) -> u64 {
    let mut x = (i as u64)
        .wrapping_add(seed)
        .wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
fn decode(c: u8) -> f32 {
    let e = (c >> 3) & 15;
    let m = c & 7;
    let a = if e == 0 {
        m as f32 / 512.
    } else {
        (1. + m as f32 / 8.) * 2f32.powi(e as i32 - 7)
    };
    if c & 128 != 0 { -a } else { a }
}
fn encoded(slot: usize, is_v: bool) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let seed = 5000 + slot as u64 + if is_v { 10000 } else { 0 };
    let mut codes = vec![0u8; RING];
    let mut scales = vec![0f32; 8 * 1024 * 8];
    let mut decoded = vec![0f32; RING];
    for t in 0..1024 {
        for h in 0..8 {
            let scale = 2f32.powi((word(t * 8 + h, seed) % 4) as i32 - if is_v { 6 } else { 9 });
            scales[(h * 1024 + t) * 8..(h * 1024 + t + 1) * 8].fill(scale);
            for j in 0..256 {
                let i = (t * 8 + h) * 256 + j;
                let w = word(i, seed);
                let c = (w % 88) as u8 | ((w >> 32) as u8 & 128);
                codes[i] = c;
                decoded[i] = bf(decode(c) * scale);
            }
        }
    }
    (codes, scales, decoded)
}
#[device(chip = 1)]
fn put_ring(
    device: &mut Device,
    k: &HbmTensor<f8e4m3, Chip, m![Ts, Ns, Ds]>,
    s: &HbmTensor<f32, Chip, m![Ns, Ts, Scale8]>,
    data_base: &HbmTensor<i32, Chip, m![1]>,
    scale_base: &HbmTensor<i32, Chip, m![1]>,
    out: &mut HbmTensor<f8e4m3, Chip, pool::DataPool>,
    os: &mut HbmTensor<f32, Chip, pool::ScalePool>,
) {
    type Cl = m![1#2];
    type Rows = m![Ns, Ts / 32];
    let k: DmTensor<f8e4m3, Chip, Cl, Rows, m![Ts % 32, Ds]> = k.to_dm(&mut device.tdma);
    let s: DmTensor<f32, Chip, Cl, Rows, m![Ts % 32, Scale8]> = s.to_dm(&mut device.tdma);
    out.view_mut()
        .scatter::<m![Rc64], m![Ts, Ns, Ds]>()
        .by_byte_offsets(data_base.view())
        .from_dm(&mut device.tdma, k);
    os.view_mut()
        .scatter::<m![Rc64], m![Ns, Ts, Scale8]>()
        .by_byte_offsets(scale_base.view())
        .from_dm(&mut device.tdma, s);
}
async fn indices(d: &mut Device, x: i32) -> HbmTensor<i32, Chip, m![1]> {
    HostTensor::<i32, m![1]>::from_vec(vec![x])
        .to_hbm(&mut d.pdma)
        .await
        .unwrap()
}
async fn put(
    d: &mut Device,
    codes: &[u8],
    scales: &[f32],
    db: &HbmTensor<i32, Chip, m![1]>,
    sb: &HbmTensor<i32, Chip, m![1]>,
    p: &mut HbmTensor<f8e4m3, Chip, pool::DataPool>,
    s: &mut HbmTensor<f32, Chip, pool::ScalePool>,
) {
    let q = HostTensor::<f8e4m3, m![Ts, Ns, Ds]>::from_buf(codes.to_vec())
        .to_hbm(&mut d.pdma)
        .await
        .unwrap();
    let scale = f_upload::<m![Ns, Ts, Scale8]>(d, scales).await;
    launch(put_ring, (d, &q, &scale, db, sb, p, s))
        .await
        .unwrap();
}
fn exact(label: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    let count = a
        .iter()
        .zip(b)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    if count != 0 {
        for (i, (a, b)) in a
            .iter()
            .zip(b)
            .enumerate()
            .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
            .take(5)
        {
            println!("DIFF {label} i={i} actual={a} expected={b}");
        }
    }
    assert_eq!(count, 0, "{label}");
    println!("EXACT {label} values={} differences=0", a.len());
}
async fn test(
    d: &mut Device,
    slot: usize,
    start: usize,
    valid: usize,
    pk: &mut HbmTensor<f8e4m3, Chip, pool::DataPool>,
    pv: &mut HbmTensor<f8e4m3, Chip, pool::DataPool>,
    pks: &mut HbmTensor<f32, Chip, pool::ScalePool>,
    pvs: &mut HbmTensor<f32, Chip, pool::ScalePool>,
) {
    let db = indices(d, (slot * RING) as i32).await;
    let sb = indices(d, (slot * 8 * 1024 * 8 * 4) as i32).await;
    let (mut ck, mut sk, ok) = encoded(slot, false);
    let (mut cv, mut sv, ov) = encoded(slot, true);
    put(d, &ck, &sk, &db, &sb, pk, pks).await;
    put(d, &cv, &sv, &db, &sb, pv, pvs).await;
    let qv = values(128 * 4096, 100 + start as u64, 0.2);
    let nkv = values(128 * 2048, 400 + start as u64, 0.2);
    let nvv = values(128 * 2048, 500 + start as u64, 1.);
    let q = bf_upload::<m![P128, Ns, Gs, Ds]>(d, &qv).await;
    let nk = bf_upload::<m![P128, Ns, Ds]>(d, &nkv).await;
    let nv = bf_upload::<m![P128, Ns, Ds]>(d, &nvv).await;
    let mut rk = zero::<bf16, m![Ts, Ns, Ds]>(d).await;
    let mut rv = zero::<bf16, m![Ts, Ns, Ds]>(d).await;
    launch(
        pool::dequant_ring,
        (d, &*pk, &*pks, &*pv, &*pvs, &db, &sb, &mut rk, &mut rv),
    )
    .await
    .unwrap();
    exact("CPU reconstructed K", &read(d, &rk).await, &ok);
    exact("CPU reconstructed V", &read(d, &rv).await, &ov);
    let (om, nm) = masks(start, valid, 128, 128);
    let om = f_upload::<m![P128, Ts]>(d, &om).await;
    let nm = f_upload::<m![P128, Pk128]>(d, &nm).await;
    let mut want = zero::<bf16, m![P128, Ns, Gs, Ds]>(d).await;
    let mut got = zero::<bf16, m![P128, Ns, Gs, Ds]>(d).await;
    launch(
        owned::sliding_attention_p128,
        (d, &q, &rk, &rv, &nk, &nv, &om, &nm, &mut want),
    )
    .await
    .unwrap();
    launch(
        candidate::sliding_attention_p128,
        (
            d, &q, &*pk, &*pks, &*pv, &*pvs, &db, &sb, &nk, &nv, &om, &nm, &mut got,
        ),
    )
    .await
    .unwrap();
    let expected = read(d, &want).await;
    let actual = read(d, &got).await;
    exact(
        &format!("direct slot={slot} start={start} valid={valid}"),
        &actual[..valid * 4096],
        &expected[..valid * 4096],
    );
    cpu_samples(&qv, &ok, &ov, &nkv, &nvv, start, valid, &actual);
    let mut a = vec![];
    let mut b = vec![];
    for choice in [0, 1, 1, 0, 1, 0, 0, 1] {
        let begin = Instant::now();
        if choice == 0 {
            launch(
                pool::dequant_ring,
                (d, &*pk, &*pks, &*pv, &*pvs, &db, &sb, &mut rk, &mut rv),
            )
            .await
            .unwrap();
            launch(
                owned::sliding_attention_p128,
                (d, &q, &rk, &rv, &nk, &nv, &om, &nm, &mut want),
            )
            .await
            .unwrap();
            a.push(begin.elapsed().as_secs_f64() * 1e6);
        } else {
            launch(
                candidate::sliding_attention_p128,
                (
                    d, &q, &*pk, &*pks, &*pv, &*pvs, &db, &sb, &nk, &nv, &om, &nm, &mut got,
                ),
            )
            .await
            .unwrap();
            b.push(begin.elapsed().as_secs_f64() * 1e6);
        }
    }
    println!(
        "TIMING slot={slot} start={start} valid={valid} reconstruct_plus_owned_us={a:?} direct_us={b:?}"
    );
    let mut nk_poison = nkv.clone();
    let mut nv_poison = nvv.clone();
    nk_poison[(valid - 1) * 2048..].fill(128.);
    nv_poison[(valid - 1) * 2048..].fill(1024.);
    let nkp = bf_upload::<m![P128, Ns, Ds]>(d, &nk_poison).await;
    let nvp = bf_upload::<m![P128, Ns, Ds]>(d, &nv_poison).await;
    launch(
        candidate::sliding_attention_p128,
        (
            d, &q, &*pk, &*pks, &*pv, &*pvs, &db, &sb, &nkp, &nvp, &om, &nm, &mut got,
        ),
    )
    .await
    .unwrap();
    exact(
        "future current chunk invariance",
        &read(d, &got).await[..(valid - 1) * 4096],
        &expected[..(valid - 1) * 4096],
    );
    {
        let mut changed = 0;
        for t in 0..1024 {
            if !old_position(start, t).is_some_and(|p| p + 1024 > start) {
                ck[t * 2048..(t + 1) * 2048].fill(0x7e);
                cv[t * 2048..(t + 1) * 2048].fill(0xfe);
                for h in 0..8 {
                    sk[(h * 1024 + t) * 8..(h * 1024 + t + 1) * 8].fill(1.);
                    sv[(h * 1024 + t) * 8..(h * 1024 + t + 1) * 8].fill(1.);
                }
                changed += 1;
            }
        }
        if changed > 0 {
            put(d, &ck, &sk, &db, &sb, pk, pks).await;
            put(d, &cv, &sv, &db, &sb, pv, pvs).await;
            launch(
                candidate::sliding_attention_p128,
                (
                    d, &q, &*pk, &*pks, &*pv, &*pvs, &db, &sb, &nk, &nv, &om, &nm, &mut got,
                ),
            )
            .await
            .unwrap();
            exact(
                "unwritten or expired old history invariance",
                &read(d, &got).await[..valid * 4096],
                &expected[..valid * 4096],
            );
        }
    }
    // Reader must leave all selected cache bytes untouched. Reconstruct and compare current poison state.
    launch(
        pool::dequant_ring,
        (d, &*pk, &*pks, &*pv, &*pvs, &db, &sb, &mut rk, &mut rv),
    )
    .await
    .unwrap();
    let dk: Vec<_> = (0..RING)
        .map(|i| {
            let t = i / 2048;
            let h = i / 256 % 8;
            bf(decode(ck[i]) * sk[(h * 1024 + t) * 8])
        })
        .collect();
    let dv: Vec<_> = (0..RING)
        .map(|i| {
            let t = i / 2048;
            let h = i / 256 % 8;
            bf(decode(cv[i]) * sv[(h * 1024 + t) * 8])
        })
        .collect();
    exact("old K unchanged", &read(d, &rk).await, &dk);
    exact("old V unchanged", &read(d, &rv).await, &dv);
}
#[tokio::main]
async fn main() {
    let all = Instant::now();
    let mut d = Device::new(candidate::sliding_attention_p128.topology()).unwrap();
    let mut pk = launch(pool::zero_data_pool, &mut d).await.unwrap();
    let mut pv = launch(pool::zero_data_pool, &mut d).await.unwrap();
    let mut pks = launch(pool::zero_scale_pool, &mut d).await.unwrap();
    let mut pvs = launch(pool::zero_scale_pool, &mut d).await.unwrap();
    for (slot, start, valid) in [
        (0, 0, 128),
        (31, 384, 127),
        (63, 896, 128),
        (0, 1017, 17),
        (31, 1024, 128),
        (63, 2048, 128),
    ] {
        test(
            &mut d, slot, start, valid, &mut pk, &mut pv, &mut pks, &mut pvs,
        )
        .await;
    }
    println!(
        "PASS direct E4 P128 reader same arithmetic, CPU, causal/future/expired invariance and immutable ring; elapsed={:.3}s",
        all.elapsed().as_secs_f64()
    );
}
