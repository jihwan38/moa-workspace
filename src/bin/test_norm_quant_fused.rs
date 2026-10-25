//! Complete original8-launch vs fused1-launch QKV ingress comparison.
use furiosa_opt_gemma4::{Chip, axes::*, ops_norm_fused as fused,
    ops_norm_wide as norm, ops_query_native_fp8 as quant};
use furiosa_opt_std::prelude::*;
use std::time::Instant;
type Hidden = HbmTensor<bf16, Chip, m![P128,H]>;
type Quantized = HbmTensor<f8e4m3, Chip, m![P128,H]>;
type Scale = HbmTensor<f32, Chip, m![P128,1#8]>;
fn bf(x:f32)->f32 {
    f32::from_bits(x.to_bits().wrapping_add(0x7fff+((x.to_bits()>>16)&1))&0xffff0000)
}
fn values(n:usize,seed:u64,scale:f32)->Vec<f32> {
    (0..n).map(|i| {
        let mut x=(i as u64).wrapping_add(seed).wrapping_add(0x9e3779b97f4a7c15);
        x=(x^(x>>30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x=(x^(x>>27)).wrapping_mul(0x94d049bb133111eb);x^=x>>31;
        bf((((x>>40)as f32)/16777216.0*2.0-1.0)*scale)
    }).collect()
}
async fn upload<Mp:M>(d:&mut Device,x:&[f32])->HbmTensor<bf16,Chip,Mp> {
    assert_eq!(x.len(),Mp::SIZE);
    HostTensor::<bf16,Mp>::from_buf(x.iter().flat_map(|x|
        ((bf(*x).to_bits()>>16)as u16).to_le_bytes()).collect::<Vec<_>>())
        .to_hbm(&mut d.pdma).await.unwrap()
}
async fn bytes<D:ScalarBytes+MaterializableScalar,Mp:M>(d:&mut Device,x:&HbmTensor<D,Chip,Mp>)->Vec<u8> {
    let h:HostTensor<D,Mp>=x.to_host(&mut d.pdma).await.unwrap();h.into_inner().into_buf()
}
struct Output { q:Quantized, scale:Scale, normalized:Hidden }
impl Output {
    async fn new(d:&mut Device)->Self {
        Self {
            q:HostTensor::<f8e4m3,m![P128,H]>::from_buf(vec![0x7f;128*3840]).to_hbm(&mut d.pdma).await.unwrap(),
            scale:HostTensor::<f32,m![P128,1#8]>::from_buf((0..128*8).flat_map(|_|0x7fc00000u32.to_le_bytes()).collect::<Vec<_>>()).to_hbm(&mut d.pdma).await.unwrap(),
            normalized:upload(d,&vec![0.0;128*3840]).await,
        }
    }
}
async fn run(d:&mut Device,candidate:bool,input:&[Hidden;4],weight:&HbmTensor<bf16,Chip,m![H]>,out:&mut[Output;4]) {
    if candidate {
        let [a,b,c,e]=out;
        launch(fused::normalize_qkv_four,(d,&input[0],&input[1],&input[2],&input[3],weight,
            &mut a.q,&mut a.scale,&mut b.q,&mut b.scale,&mut c.q,&mut c.scale,&mut e.q,&mut e.scale)).await.unwrap();
    } else {
        for i in 0..4 {
            launch(norm::rmsnorm_p128,(d,&input[i],weight,&mut out[i].normalized)).await.unwrap();
            launch(quant::quantize_input,(d,&out[i].normalized,&mut out[i].q,&mut out[i].scale)).await.unwrap();
        }
    }
}
async fn test(d:&mut Device,seed:u64,order:[usize;4]) {
    let x:[Vec<f32>;4]=std::array::from_fn(|g| {
        let mut v=values(128*3840,seed+order[g]as u64*111,3.0);
        for row in 0..128 {
            if row%23==0 {v[row*3840..(row+1)*3840].fill(0.0);}
            if row%23==1 {v[row*3840..(row+1)*3840].fill(bf(1e-5));}
        }v
    });
    let input=[upload(d,&x[0]).await,upload(d,&x[1]).await,upload(d,&x[2]).await,upload(d,&x[3]).await];
    let weight=upload(d,&values(3840,seed+1500,1.75)).await;
    let mut candidate=[Output::new(d).await,Output::new(d).await,Output::new(d).await,Output::new(d).await];
    let mut control=[Output::new(d).await,Output::new(d).await,Output::new(d).await,Output::new(d).await];
    run(d,true,&input,&weight,&mut candidate).await;
    run(d,false,&input,&weight,&mut control).await;
    for group in 0..4 {
        let a=bytes(d,&control[group].q).await;let b=bytes(d,&candidate[group].q).await;
        assert_eq!(a.len(),128*3840);assert_eq!(a,b,"all FP8 input seed{seed} group{group}");
        let a=bytes(d,&control[group].scale).await;let b=bytes(d,&candidate[group].scale).await;
        assert_eq!(a.len(),128*8*4);assert_eq!(a.len(),b.len());
        for row in 0..128 {
            let i=row*8*4;
            assert_eq!(&a[i..i+4],&b[i..i+4],"scale seed{seed} group{group} row{row}");
            let scale=f32::from_le_bytes(b[i..i+4].try_into().unwrap());
            assert!(scale.is_finite()&&scale>0.0);
        }
    }
    println!("EXACT seed={seed} order={order:?} fp8_values={} live_scales=512",512*3840);
    let mut timing=[Vec::new(),Vec::new()];
    for variant in [false,true,true,false,false,true,true,false] {
        let out=if variant {&mut candidate}else{&mut control};
        let now=Instant::now();run(d,variant,&input,&weight,out).await;
        timing[variant as usize].push(now.elapsed().as_secs_f64()*1e6);
    }
    println!("TIMING seed={seed} control8launch_us={:?} fused1launch_us={:?}",timing[0],timing[1]);
}
#[tokio::main]
async fn main() {
    let mut d=Device::new(fused::normalize_qkv_four.topology()).unwrap();
    test(&mut d,713,[0,1,2,3]).await;
    test(&mut d,1771,[3,1,0,2]).await;
    test(&mut d,5001,[2,0,3,1]).await;
    println!("PASS QKV ingress exact all512rows and scales, independent poison outputs, zero/tiny/signed and group permutations");
}
