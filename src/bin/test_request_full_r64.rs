//! Independent full attention over permuted pooled pages, ragged lengths and sparse updates.
use furiosa_opt_gemma4::{Chip, axes::*, ops, ops_request_attention as pool};
use furiosa_opt_std::prelude::*;
use pool::Fc;
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
async fn full_serial(d: &mut Device, qv: &[f32], kv: &[f32], vv: &[f32], pos: usize) -> Vec<f32> {
    let q = bf_upload::<m![Gf, Df]>(d, qv).await;
    let mut max = zero::<f32, m![Gf]>(d).await;
    let mut sum = zero::<f32, m![Gf]>(d).await;
    let mut out = zero::<bf16, m![Gf, Df]>(d).await;
    for page in 0..=pos / 512 {
        let a = page * 512 * 512;
        let k = bf_upload::<m![Tf, Df]>(d, &kv[a..a + 512 * 512]).await;
        let v = bf_upload::<m![Tf, Df]>(d, &vv[a..a + 512 * 512]).await;
        let mask: Vec<f32> = (0..512)
            .map(|t| if page * 512 + t <= pos { 1.0 } else { 0. })
            .collect();
        let mask = f_upload::<m![Tf]>(d, &mask).await;
        if page == 0 {
            launch(
                ops::full_attention_first_page,
                (d, &q, &k, &v, &mask, &mut max, &mut sum, &mut out),
            )
            .await
            .unwrap();
        } else {
            launch(
                ops::full_attention_page,
                (d, &q, &k, &v, &mask, &mut max, &mut sum, &mut out),
            )
            .await
            .unwrap();
        }
    }
    let sum: HostTensor<f32, m![Gf]> = sum.to_host(&mut d.pdma).await.unwrap();
    let sum = sum.into_vec();
    read(d, &out)
        .await
        .into_iter()
        .enumerate()
        .map(|(i, x)| bf(x / sum[i / 512]))
        .collect()
}

const POS:[usize;64]=[0,1,255,511,512,513,1023,1024,15,500,700,1030,31,256,800,1040,0,1,255,511,512,513,1023,1024,15,500,700,1030,31,256,800,1040,0,1,255,511,512,513,1023,1024,15,500,700,1030,31,256,800,1040,0,1,255,511,512,513,1023,1024,15,500,700,1030,31,256,800,1040];
fn block(row: usize, page: usize) -> usize {
    ((row * 3 + page) * 5 + 7) % 256
}
async fn indices<Mp: M>(d: &mut Device, x: Vec<i32>) -> HbmTensor<i32, Chip, Mp> {
    HostTensor::<i32, Mp>::from_vec(x)
        .to_hbm(&mut d.pdma)
        .await
        .unwrap()
}
async fn run(
    d: &mut Device,
    q: &HbmTensor<bf16, Chip, m![R64, Gf, Df]>,
    k: &HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    v: &HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    count: usize,
) -> Vec<f32> {
    let mut max = zero::<f32, m![R64,Gf,1 # 8]>(d).await;
    let mut sum = zero::<f32, m![R64,Gf,1 # 8]>(d).await;
    let mut out = zero::<bf16, m![R64, Gf, Df]>(d).await;
    let mut previous = vec![0.; 64 * 8192];
    for page in 0..3 {
        let offset = indices::<m![R64]>(
            d,
            (0..64)
                .map(|r| (block(r, page.min(POS[r] / 512)) * 512 * 512 * 2) as i32)
                .collect(),
        )
        .await;
        let active: Vec<f32> = (0..64)
            .flat_map(|r| {
                vec![
                    if r < count && page <= POS[r] / 512 {
                        1.
                    } else {
                        0.
                    };
                    8
                ]
            })
            .collect();
        let active = f_upload::<m![R64,1 # 8]>(d, &active).await;
        let mask: Vec<f32> = (0..64)
            .flat_map(|r| {
                (0..512).map(move |t| {
                    if r < count && page * 512 + t <= POS[r] {
                        1.
                    } else {
                        0.
                    }
                })
            })
            .collect();
        let mask = f_upload::<m![R64, Tf]>(d, &mask).await;
        if page == 0 {
            launch(furiosa_opt_gemma4::ops_request_full_r64::full_attention_first_page_r64,
                (d,q,k,v,&offset,&mask,&active,&mut max,&mut sum,&mut out)).await.unwrap();
        }
        let start = Instant::now();
        if page == 0 {
            launch(
                furiosa_opt_gemma4::ops_request_full_r64::full_attention_first_page_r64,
                (
                    d, q, k, v, &offset, &mask, &active, &mut max, &mut sum, &mut out,
                ),
            )
            .await
            .unwrap();
        } else {
            launch(
                furiosa_opt_gemma4::ops_request_full_r64::full_attention_next_page_r64,
                (
                    d, q, k, v, &offset, &mask, &active, &mut max, &mut sum, &mut out,
                ),
            )
            .await
            .unwrap();
        }
        println!(
            "pool full count={count} page={page} launch us={:.3}",
            start.elapsed().as_secs_f64() * 1e6
        );
        let now = read(d, &out).await;
        if page > 0 {
            for r in 0..count {
                if page > POS[r] / 512 {
                    assert_eq!(
                        &now[r * 8192..(r + 1) * 8192],
                        &previous[r * 8192..(r + 1) * 8192],
                        "inactive page changed numerator row={r}"
                    );
                }
            }
        }
        previous = now;
    }
    let mut divided = zero::<bf16, m![R64, Gf, Df]>(d).await;
    launch(
        furiosa_opt_gemma4::ops_request_full_r64::full_attention_divide_r64,
        (d, &out, &sum, &mut divided),
    )
    .await
    .unwrap();
    read(d, &divided).await
}
async fn compare(
    d: &mut Device,
    qv: &[f32],
    data: &[(Vec<f32>, Vec<f32>)],
    got: &[f32],
    count: usize,
) {
    for r in 0..count {
        let reference = full_serial(
            d,
            &qv[r * 8192..(r + 1) * 8192],
            &data[r].0,
            &data[r].1,
            POS[r],
        )
        .await;
        let n = got[r * 8192..(r + 1) * 8192]
            .iter()
            .zip(&reference)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let max = got[r * 8192..(r + 1) * 8192]
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!(
            "pool full row={r} pos={} bit_mismatches={n} max_abs={max}",
            POS[r]
        );
        assert_eq!(n, 0, "pooled vs scalar");
    }
    assert!(
        got[count * 8192..].iter().all(|x| *x == 0.),
        "inactive batch rows must produce finite zero"
    );
}
#[tokio::main]
async fn main() {
    let mut d = Device::new(furiosa_opt_gemma4::ops_request_full_r64::full_attention_first_page_r64.topology()).unwrap();
    let mut k = zero::<bf16, m![Fc, Tf, Df]>(&mut d).await;
    let mut v = zero::<bf16, m![Fc, Tf, Df]>(&mut d).await;
    let qv = values(64 * 8192, 100, 0.2);
    let q = bf_upload::<m![R64, Gf, Df]>(&mut d, &qv).await;
    let mut data = Vec::new();
    for r in 0..64 {
        let len = (POS[r] / 512 + 1) * 512 * 512;
        let kv = values(len, 200 + r as u64, 0.2);
        let vv = values(len, 300 + r as u64, 1.);
        for page in 0..=POS[r] / 512 {
            let a = page * 512 * 512;
            let kr = bf_upload::<m![Tf, Df]>(&mut d, &kv[a..a + 512 * 512]).await;
            let vr = bf_upload::<m![Tf, Df]>(&mut d, &vv[a..a + 512 * 512]).await;
            let index =
                indices::<m![1]>(&mut d, vec![(block(r, page) * 512 * 512 * 2) as i32]).await;
            launch(
                pool::copy_full_page_to_pool,
                (&mut d, &kr, &vr, &index, &mut k, &mut v),
            )
            .await
            .unwrap();
        }
        data.push((kv, vv));
    }
    let got = run(&mut d, &q, &k, &v, 64).await;
    compare(&mut d, &qv, &data, &got, 64).await;
    let baseline=got.clone();for t in 1..512{data[0].0[t*512..(t+1)*512].fill(128.);data[0].1[t*512..(t+1)*512].fill(1024.);}let kr=bf_upload::<m![Tf,Df]>(&mut d,&data[0].0).await;let vr=bf_upload::<m![Tf,Df]>(&mut d,&data[0].1).await;let index=indices::<m![1]>(&mut d,vec![(block(0,0)*512*512*2) as i32]).await;launch(pool::copy_full_page_to_pool,(&mut d,&kr,&vr,&index,&mut k,&mut v)).await.unwrap();let changed=run(&mut d,&q,&k,&v,64).await;assert_eq!(baseline,changed,"invalidfuture K/V changed independent outputs");
    let newk = values(16 * 512, 400, 0.2);
    let newv = values(16 * 512, 500, 1.);
    let nk = bf_upload::<m![B, Df]>(&mut d, &newk).await;
    let nv = bf_upload::<m![B, Df]>(&mut d, &newv).await;
    let offsets = indices::<m![B = 3]>(
        &mut d,
        (0..3)
            .map(|r| ((block(r, 0) * 512 + POS[r]) * 512 * 2) as i32)
            .collect(),
    )
    .await;
    launch(
        pool::commit_full_pool_3,
        (&mut d, &nk, &nv, &offsets, &mut k, &mut v),
    )
    .await
    .unwrap();
    for r in 0..3 {
        data[r].0[POS[r] * 512..(POS[r] + 1) * 512].copy_from_slice(&newk[r * 512..(r + 1) * 512]);
        data[r].1[POS[r] * 512..(POS[r] + 1) * 512].copy_from_slice(&newv[r * 512..(r + 1) * 512]);
        let mut kr = zero::<bf16, m![Tf, Df]>(&mut d).await;
        let mut vr = zero::<bf16, m![Tf, Df]>(&mut d).await;
        let index = indices::<m![1]>(&mut d, vec![(block(r, 0) * 512 * 512 * 2) as i32]).await;
        launch(
            pool::copy_full_pool_to_page,
            (&mut d, &k, &v, &index, &mut kr, &mut vr),
        )
        .await
        .unwrap();
        assert_eq!(read(&mut d, &kr).await, data[r].0[..512 * 512]);
        assert_eq!(read(&mut d, &vr).await, data[r].1[..512 * 512]);
    }
    let got = run(&mut d, &q, &k, &v, 3).await;
    compare(&mut d, &qv, &data, &got, 3).await;
    let got = run(&mut d, &q, &k, &v, 64).await;
    compare(&mut d, &qv, &data, &got, 64).await;
    println!(
        "PASS: pooled full attention exact64 requests, ragged pages,futureK/V invariance,3-row update and pool migration,64->3->64 active transitions"
    );
}
