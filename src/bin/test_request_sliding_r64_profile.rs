#[path="support/npu_profile.rs"]
mod npu_profile;
// Pooled sliding attention: distinct request rings, sparse updates and migration.
use furiosa_opt_gemma4::{Chip, axes::*, ops, ops_request_attention as pool};
use furiosa_opt_std::prelude::*;
use furiosa_opt_gemma4::ops_pool_views::Rc64;
use std::time::Instant;
static COLLECTOR:std::sync::OnceLock<npu_profile::Collector>=std::sync::OnceLock::new();
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

const POS:[usize;64]=[0,1,255,511,512,1023,1024,1535,31,510,700,2048,15,256,800,8191,0,1,255,511,512,1023,1024,1535,31,510,700,2048,15,256,800,8191,0,1,255,511,512,1023,1024,1535,31,510,700,2048,15,256,800,8191,0,1,255,511,512,1023,1024,1535,31,510,700,2048,15,256,800,8191];
fn slot(r: usize) -> usize {
    (r * 5 + 3) % 64
}
async fn indices<Mp: M>(d: &mut Device, x: Vec<i32>) -> HbmTensor<i32, Chip, Mp> {
    HostTensor::<i32, Mp>::from_vec(x)
        .to_hbm(&mut d.pdma)
        .await
        .unwrap()
}
async fn run(
    d: &mut Device,
    q: &HbmTensor<bf16, Chip, m![R64, Ns, Gs, Ds]>,
    k: &HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    count: usize,
) -> Vec<f32> {
    let index = indices::<m![R64]>(
        d,
        (0..64).map(|r| (slot(r) * 1024 * 2048 * 2) as i32).collect(),
    )
    .await;
    let mask: Vec<f32> = (0..64)
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
    let mask = f_upload::<m![R64, Ts]>(d, &mask).await;
    let mut out = zero::<bf16, m![R64, Ns, Gs, Ds]>(d).await;
    launch(
        furiosa_opt_gemma4::ops_request_sliding_r64::sliding_attention_r64,
        (d, q, k, v, &index, &mask, &mut out),
    )
    .await
    .unwrap();
    let start = Instant::now();
    launch(
        furiosa_opt_gemma4::ops_request_sliding_r64::sliding_attention_r64,
        (d, q, k, v, &index, &mask, &mut out),
    )
    .await
    .unwrap();
    println!(
        "pooled sliding count={count} warm launch us={:.3}",
        start.elapsed().as_secs_f64() * 1e6
    );
    if std::env::var_os("FURIOSA_OPT_PROFILE").is_some() {
      COLLECTOR.get().unwrap().enable(true);
      launch(furiosa_opt_gemma4::ops_request_sliding_r64::sliding_attention_r64,(d,q,k,v,&index,&mask,&mut out)).await.unwrap();
      COLLECTOR.get().unwrap().enable(false);
    }
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
    let collector=npu_profile::Collector::default();
    tracing::subscriber::set_global_default(collector.clone()).unwrap();
    COLLECTOR.set(collector).ok().unwrap();
    let mut d = Device::new(
        furiosa_opt_gemma4::ops_request_sliding_r64::sliding_attention_r64.topology(),
    )
    .unwrap();
    let mut k = zero::<bf16, m![Rc64, Ts, Ns, Ds]>(&mut d).await;
    let mut v = zero::<bf16, m![Rc64, Ts, Ns, Ds]>(&mut d).await;
    let qv = values(64 * 4096, 100, 0.2);
    let q = bf_upload::<m![R64, Ns, Gs, Ds]>(&mut d, &qv).await;
    let mut data = Vec::new();
    for r in 0..64 {
        data.push((
            values(1024 * 2048, 200 + r as u64, 0.2),
            values(1024 * 2048, 300 + r as u64, 1.),
        ));
        copy(&mut d, &mut k, &mut v, r, &data[r]).await;
    }
    let before = run(&mut d, &q, &k, &v, 64).await;
    compare(&mut d, &qv, &data, &before, 64).await;
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
    let after = run(&mut d, &q, &k, &v, 64).await;
    assert_eq!(
        before, after,
        "future keys/values must not affect any independent output"
    );
    for r in 0..7 {let pos=POS[r]%1024;let newk=values(2048,400+r as u64,0.2);let newv=values(2048,500+r as u64,1.);data[r].0[pos*2048..(pos+1)*2048].copy_from_slice(&newk);data[r].1[pos*2048..(pos+1)*2048].copy_from_slice(&newv);copy(&mut d,&mut k,&mut v,r,&data[r]).await;}
    let got = run(&mut d, &q, &k, &v, 7).await;
    compare(&mut d, &qv, &data, &got, 7).await;
    let got = run(&mut d, &q, &k, &v, 64).await;
    compare(&mut d, &qv, &data, &got, 64).await;
    println!(
        "PASS: pooled sliding64 independent requests, future invariance,7-row CPU-authoritative cache updates,64->7->64 active transitions"
    );
}
