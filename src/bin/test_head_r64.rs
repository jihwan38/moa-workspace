//! Nonzero original-head oracle for true batch RMSNorm/RoPE.
use furiosa_opt_gemma4::{Chip, axes::*, ops_head_batch as heads, ops_head_r64 as wide};
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
    0, 1, 15, 16, 127, 255, 511, 512, 1008, 1023, 1024, 2047, 4095, 8191, 16383, 32767,
];
async fn offsets<Mp: M>(d: &mut Device, pos: &[usize], dim: usize) -> HbmTensor<i32, Chip, Mp> {
    HostTensor::<i32, Mp>::from_vec(pos.iter().map(|p| (p * dim * 2) as i32).collect::<Vec<_>>())
        .to_hbm(&mut d.pdma)
        .await
        .unwrap()
}
fn tables(dim: usize, rotated: usize, theta: f64) -> (Vec<f32>, Vec<f32>) {
    let mut cos = vec![0.; 32768 * dim];
    let mut sin = vec![0.; 32768 * dim];
    for &p in &POS {
        for i in 0..dim {
            let h = i % (dim / 2);
            if h < rotated / 2 {
                let a = (p as f32) / (theta as f32).powf((2 * h) as f32 / dim as f32);
                cos[p * dim + i] = bf(a.cos());
                sin[p * dim + i] = bf(a.sin() * if i < dim / 2 { -1. } else { 1. });
            } else {
                cos[p * dim + i] = 1.;
                sin[p * dim + i] = 0.;
            }
        }
    }
    (cos, sin)
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


async fn test_sliding(d:&mut Device){
 let mut qv=values(64*4096,110,4.);let mut kv=values(64*2048,120,4.);
 for tile in 0..4{qv[tile*16*4096..tile*16*4096+256].fill(0.);kv[tile*16*2048+2048..tile*16*2048+2304].fill(bf(1e-5));}
 let q=bf_upload::<m![R64,Qs]>(d,&qv).await;let k=bf_upload::<m![R64,Ps]>(d,&kv).await;
 let vv=values(64*2048,130,4.);let v=bf_upload::<m![R64,Ps]>(d,&vv).await;
 let qw:Vec<_>=values(256,140,0.5).into_iter().map(|x|bf(x+1.)).collect();let kw:Vec<_>=values(256,150,0.5).into_iter().map(|x|bf(x+1.)).collect();
 let qw=bf_upload::<m![Ds]>(d,&qw).await;let kw=bf_upload::<m![Ds]>(d,&kw).await;
 let(cos,sin)=tables(256,256,10000.0);let cos=bf_upload::<m![E,Ds]>(d,&cos).await;let sin=bf_upload::<m![E,Ds]>(d,&sin).await;
 let positions:Vec<_>=POS.iter().copied().cycle().take(64).collect();let offset=offsets::<m![R64]>(d,&positions,256).await;
 let mut qo=zero::<bf16,m![R64,Ns,Gs,Ds]>(d).await;let mut ko=zero::<bf16,m![R64,Ns,Ds]>(d).await;let mut vo=zero::<bf16,m![R64,Ns,Ds]>(d).await;
 launch(wide::sliding_heads_r64,(d,&q,&k,&v,&qw,&kw,&offset,&cos,&sin,&mut qo,&mut ko,&mut vo)).await.unwrap();let start=Instant::now();launch(wide::sliding_heads_r64,(d,&q,&k,&v,&qw,&kw,&offset,&cos,&sin,&mut qo,&mut ko,&mut vo)).await.unwrap();let new=start.elapsed().as_secs_f64()*1e6;
 let gotq=read(d,&qo).await;let gotk=read(d,&ko).await;let gotv=read(d,&vo).await;let mut eq=Vec::new();let mut ek=Vec::new();let mut ev=Vec::new();let mut total=0.;
 let offset=offsets::<m![B]>(d,&POS,256).await;
 for tile in 0..4{
 let qb=bf_upload::<m![B,Qs]>(d,&qv[tile*16*4096..(tile+1)*16*4096]).await;let kb=bf_upload::<m![B,Ps]>(d,&kv[tile*16*2048..(tile+1)*16*2048]).await;
 let vb=bf_upload::<m![B,Ps]>(d,&vv[tile*16*2048..(tile+1)*16*2048]).await;
 let mut qo=zero::<bf16,m![B,Ns,Gs,Ds]>(d).await;let mut ko=zero::<bf16,m![B,Ns,Ds]>(d).await;let mut vo=zero::<bf16,m![B,Ns,Ds]>(d).await;
 launch(heads::sliding_heads_b16,(d,&qb,&kb,&vb,&qw,&kw,&offset,&cos,&sin,&mut qo,&mut ko,&mut vo)).await.unwrap();let start=Instant::now();launch(heads::sliding_heads_b16,(d,&qb,&kb,&vb,&qw,&kw,&offset,&cos,&sin,&mut qo,&mut ko,&mut vo)).await.unwrap();total+=start.elapsed().as_secs_f64()*1e6;
 eq.extend(read(d,&qo).await);ek.extend(read(d,&ko).await);ev.extend(read(d,&vo).await);
 }
 exact("slidingQ R64",&gotq,&eq);exact("slidingK R64",&gotk,&ek);exact("slidingV R64",&gotv,&ev);
 println!("sliding native_warm_us={new:.3} sum4originalB16_warm_us={total:.3}");
}
async fn test_full(d:&mut Device){
 let mut qv=values(64*8192,210,4.);let mut kv=values(64*512,220,4.);
 for tile in 0..4{qv[tile*16*8192..tile*16*8192+512].fill(0.);kv[tile*16*512+512..tile*16*512+1024].fill(bf(1e-5));}
 let q=bf_upload::<m![R64,Qf]>(d,&qv).await;let k=bf_upload::<m![R64,Pf]>(d,&kv).await;
 let qw:Vec<_>=values(512,240,0.5).into_iter().map(|x|bf(x+1.)).collect();let kw:Vec<_>=values(512,250,0.5).into_iter().map(|x|bf(x+1.)).collect();
 let qw=bf_upload::<m![Df]>(d,&qw).await;let kw=bf_upload::<m![Df]>(d,&kw).await;
 let(cos,sin)=tables(512,128,1000000.0);let cos=bf_upload::<m![E,Df]>(d,&cos).await;let sin=bf_upload::<m![E,Df]>(d,&sin).await;
 let positions:Vec<_>=POS.iter().copied().cycle().take(64).collect();let offset=offsets::<m![R64]>(d,&positions,512).await;
 let mut qo=zero::<bf16,m![R64,Gf,Df]>(d).await;let mut ko=zero::<bf16,m![R64,Df]>(d).await;let mut vo=zero::<bf16,m![R64,Df]>(d).await;
 launch(wide::full_heads_r64,(d,&q,&k,&qw,&kw,&offset,&cos,&sin,&mut qo,&mut ko,&mut vo)).await.unwrap();let start=Instant::now();launch(wide::full_heads_r64,(d,&q,&k,&qw,&kw,&offset,&cos,&sin,&mut qo,&mut ko,&mut vo)).await.unwrap();let new=start.elapsed().as_secs_f64()*1e6;
 let gotq=read(d,&qo).await;let gotk=read(d,&ko).await;let gotv=read(d,&vo).await;let mut eq=Vec::new();let mut ek=Vec::new();let mut ev=Vec::new();let mut total=0.;
 let offset=offsets::<m![B]>(d,&POS,512).await;
 for tile in 0..4{
 let qb=bf_upload::<m![B,Qf]>(d,&qv[tile*16*8192..(tile+1)*16*8192]).await;let kb=bf_upload::<m![B,Pf]>(d,&kv[tile*16*512..(tile+1)*16*512]).await;
 let mut qo=zero::<bf16,m![B,Gf,Df]>(d).await;let mut ko=zero::<bf16,m![B,Df]>(d).await;let mut vo=zero::<bf16,m![B,Df]>(d).await;
 launch(heads::full_heads_b16,(d,&qb,&kb,&qw,&kw,&offset,&cos,&sin,&mut qo,&mut ko,&mut vo)).await.unwrap();let start=Instant::now();launch(heads::full_heads_b16,(d,&qb,&kb,&qw,&kw,&offset,&cos,&sin,&mut qo,&mut ko,&mut vo)).await.unwrap();total+=start.elapsed().as_secs_f64()*1e6;
 eq.extend(read(d,&qo).await);ek.extend(read(d,&ko).await);ev.extend(read(d,&vo).await);
 }
 exact("fullQ R64",&gotq,&eq);exact("fullK R64",&gotk,&ek);exact("fullV R64",&gotv,&ev);
 println!("full native_warm_us={new:.3} sum4originalB16_warm_us={total:.3}");
}
#[tokio::main]
async fn main(){let mut d=Device::new(wide::sliding_heads_r64.topology()).unwrap();test_sliding(&mut d).await;test_full(&mut d).await;println!("PASS: native R64 heads bitexact four B16 QKV, zero/tiny and RoPE positions0..32767");}
