//! Scalar causal softmax must ignore finite but stale future cache bytes.
use furiosa_opt_gemma4::{Chip, axes::*, ops};
use furiosa_opt_std::prelude::*;
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

fn exact(label: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    let differences = a
        .iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    let energy = a.iter().map(|x| x * x).sum::<f32>();
    println!("{label}: bit_mismatches={differences} output_energy={energy:.6}");
    assert!(energy > 0.1, "trivial all-zero output");
    assert_eq!(
        differences, 0,
        "masked future cache changed visible attention"
    );
}
async fn test_full(d: &mut Device, pos: usize) {
    let size = (pos / 512 + 1) * 512 * 512;
    let q = values(8192, 100 + pos as u64, 0.2);
    let k = values(size, 200, 0.2);
    let v = values(size, 300, 1.);
    let before = full_serial(d, &q, &k, &v, pos).await;
    let mut future_k = k.clone();
    let mut future_v = v.clone();
    for token in pos + 1..(pos / 512 + 1) * 512 {
        for dim in 0..512 {
            future_k[token * 512 + dim] = bf(q[dim] * 128.);
            future_v[token * 512 + dim] = bf(1000. + dim as f32);
        }
    }
    let after = full_serial(d, &q, &future_k, &future_v, pos).await;
    exact(
        &format!("full position={pos} actual1/0 mask"),
        &before,
        &after,
    );
}
async fn sliding(
    d: &mut Device,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    pos: usize,
    visible: f32,
) -> Vec<f32> {
    let q = bf_upload::<m![Ns, Gs, Ds]>(d, q).await;
    let k = bf_upload::<m![Ts, Ns, Ds]>(d, k).await;
    let v = bf_upload::<m![Ts, Ns, Ds]>(d, v).await;
    let mask: Vec<f32> = (0..1024)
        .map(|t| if t <= pos { visible } else { 0. })
        .collect();
    let mask = f_upload::<m![Ts]>(d, &mask).await;
    let mut out = zero::<bf16, m![Ns, Gs, Ds]>(d).await;
    launch(ops::sliding_attention, (d, &q, &k, &v, &mask, &mut out))
        .await
        .unwrap();
    read(d, &out).await
}
async fn test_sliding(d: &mut Device, pos: usize) {
    let q = values(4096, 400 + pos as u64, 0.2);
    let k = values(1024 * 2048, 500, 0.2);
    let v = values(1024 * 2048, 600, 1.);
    let before = sliding(d, &q, &k, &v, pos, 1.).await;
    let mut future_k = k.clone();
    let mut future_v = v.clone();
    for token in pos + 1..1024 {
        for h in 0..8 {
            for dim in 0..256 {
                let index = (token * 8 + h) * 256 + dim;
                future_k[index] = bf(q[(h * 2) * 256 + dim] * 128.);
                future_v[index] = bf(1000. + dim as f32);
            }
        }
    }
    let after = sliding(d, &q, &future_k, &future_v, pos, 1.).await;
    exact(
        &format!("sliding position={pos} actual1/0 mask"),
        &before,
        &after,
    );
    let infinity = sliding(d, &q, &future_k, &future_v, pos, f32::INFINITY).await;
    exact(
        &format!("sliding position={pos} legacyinfinity mask"),
        &after,
        &infinity,
    );
}
#[tokio::main]
async fn main() {
    let mut d = Device::new(ops::sliding_attention.topology()).unwrap();
    for p in [0, 31, 255, 512, 1024] {
        test_full(&mut d, p).await;
    }
    for p in [0, 255, 511, 1022] {
        test_sliding(&mut d, p).await;
    }
    println!("PASS: scalar full/sliding output independent of masked future K and V");
}
