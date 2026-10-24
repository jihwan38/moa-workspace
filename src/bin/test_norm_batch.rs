//! Nonzero original scalar oracle for B16 hidden RMSNorm/residual/gate.
use furiosa_opt_gemma4::{Chip, axes::*, ops_norm_batch as norm};
use furiosa_opt_std::prelude::*;
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

fn exact(label: &str, a: &[f32], b: &[f32]) {
    let n = a
        .iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    let max = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    println!("{label}: n={} bit_mismatches={n} max_abs={max}", a.len());
    assert_eq!(n, 0, "{label} original-head mismatch");
}

async fn test(d: &mut Device, seed: u64, gate: f32) {
    let mut xv = values(16 * 3840, seed, 7.);
    xv[..3840].fill(0.);
    xv[3840..7680].fill(bf(1e-5));
    let rv = values(16 * 3840, seed + 1, 2.);
    let wv: Vec<f32> = values(3840, seed + 2, 0.75)
        .into_iter()
        .map(|x| bf(x + 1.))
        .collect();
    let x = bf_upload::<m![B, H]>(d, &xv).await;
    let w = bf_upload::<m![H]>(d, &wv).await;
    let gate = bf_upload::<m![1 # 8]>(d, &vec![bf(gate); 8]).await;
    let mut out = zero::<bf16, m![B, H]>(d).await;
    let mut residual = bf_upload::<m![B, H]>(d, &rv).await;
    let mut gated = bf_upload::<m![B, H]>(d, &rv).await;
    launch(norm::rmsnorm_b16, (d, &x, &w, &mut out))
        .await
        .unwrap();
    let t = Instant::now();
    for _ in 0..16 {
        launch(norm::rmsnorm_b16, (d, &x, &w, &mut out))
            .await
            .unwrap();
    }
    println!(
        "norm seed={seed} warm B16 launch us={:.3}",
        t.elapsed().as_secs_f64() * 1e6 / 16.
    );
    launch(norm::post_attention_b16, (d, &x, &w, &mut residual))
        .await
        .unwrap();
    launch(norm::post_feedforward_b16, (d, &x, &w, &gate, &mut gated))
        .await
        .unwrap();
    let got = read(d, &out).await;
    let gotr = read(d, &residual).await;
    let gotg = read(d, &gated).await;
    let mut reference = Vec::new();
    let mut refr = Vec::new();
    let mut refg = Vec::new();
    for row in 0..16 {
        let xr = bf_upload::<m![H]>(d, &xv[row * 3840..(row + 1) * 3840]).await;
        let mut rr = bf_upload::<m![H]>(d, &rv[row * 3840..(row + 1) * 3840]).await;
        let mut o = zero::<bf16, m![H]>(d).await;
        let mut g = zero::<bf16, m![H]>(d).await;
        launch(
            norm::reference_norm_row,
            (d, &xr, &w, &mut o, &mut rr, &gate, &mut g),
        )
        .await
        .unwrap();
        reference.extend(read(d, &o).await);
        refr.extend(read(d, &rr).await);
        refg.extend(read(d, &g).await);
    }
    exact(&format!("norm seed={seed}"), &got, &reference);
    exact(&format!("residual seed={seed}"), &gotr, &refr);
    exact(&format!("gate seed={seed}"), &gotg, &refg);
}
#[tokio::main]
async fn main() {
    let mut d = Device::new(norm::rmsnorm_b16.topology()).unwrap();
    test(&mut d, 700, 0.375).await;
    test(&mut d, 900, -1.25).await;
    println!("PASS: B16 hidden norm/residual/gate bit-exact original scalar sum tree");
}
