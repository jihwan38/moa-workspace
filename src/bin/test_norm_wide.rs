//! P128 kernels vs original B16 kernels and independent double-precision RMS.
use furiosa_opt_gemma4::{Chip, EPS, axes::*, ops_norm_batch as norm, ops_norm_wide as wide};
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


fn cpu_metrics(label:&str,got:&[f32],expected:&[f32]) {
    let mut err=0f64;let mut reference=0f64;let mut max=0f32;let mut bits=0usize;
    for (&a,&b) in got.iter().zip(expected) {
        assert!(a.is_finite()&&b.is_finite());
        err+=(a as f64-b as f64).powi(2);reference+=(b as f64).powi(2);max=max.max((a-b).abs());bits+=usize::from(a.to_bits()!=b.to_bits());
    }
    let nrmse=(err/reference.max(1e-30)).sqrt();
    println!("CPUdouble {label}: n={} nrmse={nrmse:.8} max_abs={max} bit_mismatches={bits}",got.len());
    assert!(nrmse<0.003,"independent CPU RMS mismatch");
}
async fn test(d:&mut Device,seed:u64,gate:f32) {
    let mut xv=values(128*3840,seed,7.);
    for row in 0..128 {
        if row%17==0 {xv[row*3840..(row+1)*3840].fill(0.);}
        if row%17==1 {xv[row*3840..(row+1)*3840].fill(bf(1e-5));}
    }
    let rv=values(128*3840,seed+1,2.);
    let wv:Vec<f32>=values(3840,seed+2,0.75).into_iter().map(|x|bf(x+1.)).collect();
    let gate_value=bf(gate);
    let x=bf_upload::<m![P128,H]>(d,&xv).await;
    let w=bf_upload::<m![H]>(d,&wv).await;
    let gate=bf_upload::<m![1#8]>(d,&vec![gate_value;8]).await;
    let mut out=zero::<bf16,m![P128,H]>(d).await;
    let mut residual=bf_upload::<m![P128,H]>(d,&rv).await;
    let mut gated=bf_upload::<m![P128,H]>(d,&rv).await;
    launch(wide::rmsnorm_p128,(d,&x,&w,&mut out)).await.unwrap();
    let start=Instant::now();
    for _ in 0..16 {launch(wide::rmsnorm_p128,(d,&x,&w,&mut out)).await.unwrap();}
    println!("P128 rmsnorm seed={seed} warm_us={:.3}",start.elapsed().as_secs_f64()*1e6/16.);
    launch(wide::post_attention_p128,(d,&x,&w,&mut residual)).await.unwrap();
    launch(wide::post_feedforward_p128,(d,&x,&w,&gate,&mut gated)).await.unwrap();
    let got=read(d,&out).await;let gotr=read(d,&residual).await;let gotg=read(d,&gated).await;
    let mut reference=Vec::new();let mut refr=Vec::new();let mut refg=Vec::new();let mut norm_us=0.;
    for tile in 0..8 {
        let range=tile*16*3840..(tile+1)*16*3840;
        let xt=bf_upload::<m![B,H]>(d,&xv[range.clone()]).await;
        let mut rt=bf_upload::<m![B,H]>(d,&rv[range.clone()]).await;
        let mut gt=bf_upload::<m![B,H]>(d,&rv[range]).await;
        let mut yt=zero::<bf16,m![B,H]>(d).await;
        launch(norm::rmsnorm_b16,(d,&xt,&w,&mut yt)).await.unwrap();
        let start=Instant::now();launch(norm::rmsnorm_b16,(d,&xt,&w,&mut yt)).await.unwrap();norm_us+=start.elapsed().as_secs_f64()*1e6;
        launch(norm::post_attention_b16,(d,&xt,&w,&mut rt)).await.unwrap();
        launch(norm::post_feedforward_b16,(d,&xt,&w,&gate,&mut gt)).await.unwrap();
        reference.extend(read(d,&yt).await);refr.extend(read(d,&rt).await);refg.extend(read(d,&gt).await);
    }
    println!("original8xB16 rmsnorm seed={seed} warm_launch_sum_us={norm_us:.3} (excludes tile transfer launches)");
    exact(&format!("P128 norm seed={seed}"),&got,&reference);
    exact(&format!("P128 postattention seed={seed}"),&gotr,&refr);
    exact(&format!("P128 postFFN seed={seed}"),&gotg,&refg);
    let mut cpu=Vec::with_capacity(xv.len());let mut cpur=Vec::with_capacity(xv.len());let mut cpug=Vec::with_capacity(xv.len());
    for row in 0..128 {
        let xr=&xv[row*3840..(row+1)*3840];
        let rms=(xr.iter().map(|&x|(x as f64)*(x as f64)).sum::<f64>()/3840.+EPS as f64).sqrt();
        for col in 0..3840 {
            let normalized=bf(((xr[col] as f64/rms)*wv[col] as f64) as f32);
            let residual=bf(normalized+rv[row*3840+col]);
            cpu.push(normalized);cpur.push(residual);cpug.push(bf(residual*gate_value));
        }
    }
    cpu_metrics(&format!("norm seed={seed}"),&got,&cpu);
    cpu_metrics(&format!("postattention seed={seed}"),&gotr,&cpur);
    cpu_metrics(&format!("postFFN seed={seed}"),&gotg,&cpug);
}
#[tokio::main]
async fn main() {
    let mut d=Device::new(wide::rmsnorm_p128.topology()).unwrap();
    test(&mut d,700,0.375).await;
    test(&mut d,900,-1.25).await;
    println!("PASS: P128 norm/epilogues bit-exact B16 plus CPUdouble RMS, two seeds zero/tiny/signedgate");
}
