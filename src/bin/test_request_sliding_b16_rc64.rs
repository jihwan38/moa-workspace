//! Pooled sliding attention: distinct request rings, sparse updates and migration.
use furiosa_opt_gemma4::{Chip, axes::*, ops, ops_request_attention as pool};
use furiosa_opt_std::prelude::*;
use furiosa_opt_gemma4::ops_pool_views::Rc64;
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

const POS: [usize; 16] = [
    0, 1, 255, 511, 512, 1023, 1024, 1535, 31, 510, 700, 2048, 15, 256, 800, 8191,
];
fn slot(r: usize) -> usize {
    [0,31,63,1,32,62,2,33,61,3,34,60,4,35,59,7][r]
}
async fn indices<Mp: M>(d: &mut Device, x: Vec<i32>) -> HbmTensor<i32, Chip, Mp> {
    HostTensor::<i32, Mp>::from_vec(x)
        .to_hbm(&mut d.pdma)
        .await
        .unwrap()
}
async fn run(
    d: &mut Device,
    q: &HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    k: &HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    count: usize,
) -> Vec<f32> {
    let index = indices::<m![B]>(
        d,
        (0..16).map(|r| (slot(r) * 1024 * 2048 * 2) as i32).collect(),
    )
    .await;
    let mask: Vec<f32> = (0..16)
        .flat_map(|r| {
            (0..1024).map(move |t| {
                if r < count && (POS[r] >= 1023 || t <= POS[r]) {
                    1.
                } else {
                    0.
                }
            })
        })
        .collect();
    let mask = f_upload::<m![B, Ts]>(d, &mask).await;
    let mut out = zero::<bf16, m![B, Ns, Gs, Ds]>(d).await;
    launch(
        furiosa_opt_gemma4::ops_request_sliding_b16_rc64::sliding_attention_b16,
        (d, q, k, v, &index, &mask, &mut out),
    )
    .await
    .unwrap();
    let start = Instant::now();
    launch(
        furiosa_opt_gemma4::ops_request_sliding_b16_rc64::sliding_attention_b16,
        (d, q, k, v, &index, &mask, &mut out),
    )
    .await
    .unwrap();
    println!(
        "pooled sliding count={count} warm launch us={:.3}",
        start.elapsed().as_secs_f64() * 1e6
    );
    read(d, &out).await
}
async fn compare(
    d: &mut Device,
    q: &[f32],
    data: &[(Vec<f32>, Vec<f32>)],
    got: &[f32],
    count: usize,
) {
    for r in 0..count {
        let qr = bf_upload::<m![Ns, Gs, Ds]>(d, &q[r * 4096..(r + 1) * 4096]).await;
        let kr = bf_upload::<m![Ts, Ns, Ds]>(d, &data[r].0).await;
        let vr = bf_upload::<m![Ts, Ns, Ds]>(d, &data[r].1).await;
        let mask: Vec<f32> = (0..1024)
            .map(|t| {
                if POS[r] >= 1023 || t <= POS[r] {
                    1.
                } else {
                    0.
                }
            })
            .collect();
        let mask = f_upload::<m![Ts]>(d, &mask).await;
        let mut reference = zero::<bf16, m![Ns, Gs, Ds]>(d).await;
        launch(
            ops::sliding_attention,
            (d, &qr, &kr, &vr, &mask, &mut reference),
        )
        .await
        .unwrap();
        let reference = read(d, &reference).await;
        let n = got[r * 4096..(r + 1) * 4096]
            .iter()
            .zip(&reference)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let max = got[r * 4096..(r + 1) * 4096]
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!(
            "pooled sliding row={r} pos={} bit_mismatches={n} max_abs={max}",
            POS[r]
        );
        assert_eq!(n, 0, "pooled sliding vs scalar");
    }
    assert!(
        got[count * 4096..].iter().all(|x| *x == 0.),
        "inactive rows must be finite zero"
    );
}
async fn copy(
    d: &mut Device,
    k: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    v: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    r: usize,
    data: &(Vec<f32>, Vec<f32>),
) {
    let kr = bf_upload::<m![Ts, Ns, Ds]>(d, &data.0).await;
    let vr = bf_upload::<m![Ts, Ns, Ds]>(d, &data.1).await;
    let index = indices::<m![1]>(d, vec![(slot(r) * 1024 * 2048 * 2) as i32]).await;
    launch(
        furiosa_opt_gemma4::ops_request_sliding_r64::copy_sliding_cache_to_pool_rc64,
        (d, &kr, &vr, &index, k, v),
    )
    .await
    .unwrap();
}
#[tokio::main]
async fn main() {
    let mut d = Device::new(
        furiosa_opt_gemma4::ops_request_sliding_b16_rc64::sliding_attention_b16.topology(),
    )
    .unwrap();
    let mut k = zero::<bf16, m![Rc64, Ts, Ns, Ds]>(&mut d).await;
    let mut v = zero::<bf16, m![Rc64, Ts, Ns, Ds]>(&mut d).await;
    let qv = values(16 * 4096, 100, 0.2);
    let q = bf_upload::<m![B, Ns, Gs, Ds]>(&mut d, &qv).await;
    let mut data = Vec::new();
    for r in 0..16 {
        data.push((
            values(1024 * 2048, 200 + r as u64, 0.2),
            values(1024 * 2048, 300 + r as u64, 1.),
        ));
        copy(&mut d, &mut k, &mut v, r, &data[r]).await;
    }
    let before = run(&mut d, &q, &k, &v, 16).await;
    compare(&mut d, &qv, &data, &before, 16).await;
    for t in 1..1024 {
        for h in 0..8 {
            for j in 0..256 {
                let i = (t * 8 + h) * 256 + j;
                data[0].0[i] = bf(qv[(h * 2) * 256 + j] * 128.);
                data[0].1[i] = bf(1000. + j as f32);
            }
        }
    }
    copy(&mut d, &mut k, &mut v, 0, &data[0]).await;
    let after = run(&mut d, &q, &k, &v, 16).await;
    assert_eq!(
        before, after,
        "future keys/values must not affect any independent output"
    );
    let newk = values(16 * 2048, 400, 0.2);
    let newv = values(16 * 2048, 500, 1.);
    let nk = bf_upload::<m![B, Ns, Ds]>(&mut d, &newk).await;
    let nv = bf_upload::<m![B, Ns, Ds]>(&mut d, &newv).await;
    let offsets = indices::<m![B = 7]>(
        &mut d,
        (0..7)
            .map(|r| ((slot(r) * 1024 + POS[r] % 1024) * 2048 * 2) as i32)
            .collect(),
    )
    .await;
    launch(
        furiosa_opt_gemma4::ops_commit_r64::commit_sliding_pool_7,
        (&mut d, &nk, &nv, &offsets, &mut k, &mut v),
    )
    .await
    .unwrap();
    for r in 0..7 {
        let p = POS[r] % 1024;
        data[r].0[p * 2048..(p + 1) * 2048].copy_from_slice(&newk[r * 2048..(r + 1) * 2048]);
        data[r].1[p * 2048..(p + 1) * 2048].copy_from_slice(&newv[r * 2048..(r + 1) * 2048]);
        let mut kr = zero::<bf16, m![Ts, Ns, Ds]>(&mut d).await;
        let mut vr = zero::<bf16, m![Ts, Ns, Ds]>(&mut d).await;
        let index = indices::<m![1]>(&mut d, vec![(slot(r) * 1024 * 2048 * 2) as i32]).await;
        launch(
            furiosa_opt_gemma4::ops_commit_wide_pool::copy_sliding_pool_to_cache_rc64,
            (&mut d, &k, &v, &index, &mut kr, &mut vr),
        )
        .await
        .unwrap();
        assert_eq!(read(&mut d, &kr).await, data[r].0);
        assert_eq!(read(&mut d, &vr).await, data[r].1);
    }
    let got = run(&mut d, &q, &k, &v, 7).await;
    compare(&mut d, &qv, &data, &got, 7).await;
    let got = run(&mut d, &q, &k, &v, 16).await;
    compare(&mut d, &qv, &data, &got, 16).await;
    println!(
        "PASS: pooled sliding16 independent requests inRc64 slots0/31/63, future invariance,7-row update and migration"
    );
}
