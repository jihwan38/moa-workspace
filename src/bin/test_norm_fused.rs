//! Independent-output exact oracle and complete-chain ABBA timing for norm fusion.
use furiosa_opt_gemma4::{Chip, EPS, axes::*, ops_norm_fused as fused,
    ops_norm_wide as norm, ops_prompt_group_io as io,
    ops_native_fp8_p512_blocks as quant, ops_batch_layouts::P512H960};
use furiosa_opt_std::prelude::*;
use std::time::Instant;
type Hidden = HbmTensor<bf16, Chip, m![P128,H]>;
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
async fn zero<D:ScalarBytes+MaterializableScalar,Mp:M>(d:&mut Device)->HbmTensor<D,Chip,Mp> {
    HostTensor::<D,Mp>::from_buf(vec![0u8;Mp::SIZE*D::BITS/8]).to_hbm(&mut d.pdma).await.unwrap()
}
async fn bytes<D:ScalarBytes+MaterializableScalar,Mp:M>(d:&mut Device,x:&HbmTensor<D,Chip,Mp>)->Vec<u8> {
    let h:HostTensor<D,Mp>=x.to_host(&mut d.pdma).await.unwrap();h.into_inner().into_buf()
}
fn exact(label:&str,a:&[u8],b:&[u8]) {
    assert_eq!(a.len(),b.len());let n=a.iter().zip(b).filter(|(x,y)|x!=y).count();
    println!("{label} bytes={} mismatches={n}",a.len());assert_eq!(n,0,"{label}");
}
struct State {
    residual:[Hidden;4], normalized:[Hidden;4],
    joined:HbmTensor<bf16,Chip,m![P512,H]>,
    q:HbmTensor<f8e4m3,Chip,P512H960>, scales:HbmTensor<f32,Chip,m![P512]>,
}
impl State {
    async fn new(d:&mut Device,rv:&[Vec<f32>;4])->Self {
        Self {
            residual:[upload(d,&rv[0]).await,upload(d,&rv[1]).await,upload(d,&rv[2]).await,upload(d,&rv[3]).await],
            normalized:[zero(d).await,zero(d).await,zero(d).await,zero(d).await],
            joined:zero(d).await,
            q:HostTensor::<f8e4m3,P512H960>::from_buf(vec![0x7f;P512::SIZE*H::SIZE]).to_hbm(&mut d.pdma).await.unwrap(),
            scales:HostTensor::<f32,m![P512]>::from_buf((0..P512::SIZE).flat_map(|_|0x7fc00000u32.to_le_bytes()).collect::<Vec<_>>()).to_hbm(&mut d.pdma).await.unwrap(),
        }
    }
    async fn reset_residual(&mut self,d:&mut Device,rv:&[Vec<f32>;4]) {
        self.residual=[upload(d,&rv[0]).await,upload(d,&rv[1]).await,upload(d,&rv[2]).await,upload(d,&rv[3]).await];
    }
}
async fn run(d:&mut Device,candidate:bool, projected:&[Hidden;4], post:&HbmTensor<bf16,Chip,m![H]>,
    pre:&HbmTensor<bf16,Chip,m![H]>,offsets:&[HbmTensor<i32,Chip,m![1]>;4],s:&mut State) {
    if candidate {
        let [a,b,c,e]=&mut s.residual;
        launch(fused::post_attention_ffn_input_four,(d,&projected[0],&projected[1],&projected[2],&projected[3],
            post,pre,a,b,c,e,&mut s.q,&mut s.scales)).await.unwrap();
    } else {
        for i in 0..4 {
            launch(norm::post_attention_p128,(d,&projected[i],post,&mut s.residual[i])).await.unwrap();
        }
        for i in 0..4 {
            launch(norm::rmsnorm_p128,(d,&s.residual[i],pre,&mut s.normalized[i])).await.unwrap();
        }
        launch(io::pack_hidden_group,(d,&s.normalized[0],&s.normalized[1],&s.normalized[2],&s.normalized[3],
            &offsets[0],&offsets[1],&offsets[2],&offsets[3],&mut s.joined)).await.unwrap();
        launch(quant::quantize_hidden,(d,&s.joined,&mut s.q,&mut s.scales)).await.unwrap();
    }
}
async fn test(d:&mut Device,seed:u64,order:[usize;4]) {
    let pv:[Vec<f32>;4]=std::array::from_fn(|g| {
        let mut v=values(128*3840,seed+order[g] as u64*111,3.0);
        for row in 0..128 {
            if row%23==0 {v[row*3840..(row+1)*3840].fill(0.0);}
            if row%23==1 {v[row*3840..(row+1)*3840].fill(bf(1e-5));}
        }v
    });
    let rv:[Vec<f32>;4]=std::array::from_fn(|g|values(128*3840,seed+order[g]as u64*137+17,0.75));
    let postv=values(3840,seed+1500,1.75);let prev=values(3840,seed+1700,1.25);
    let projected:[Hidden;4]=[upload(d,&pv[0]).await,upload(d,&pv[1]).await,upload(d,&pv[2]).await,upload(d,&pv[3]).await];
    let post=upload(d,&postv).await;let pre=upload(d,&prev).await;
    let mut offsets=Vec::new();
    for i in 0..4 {offsets.push(HostTensor::<i32,m![1]>::from_vec([(i*128*3840*2)as i32]).to_hbm(&mut d.pdma).await.unwrap());}
    let offsets:[_;4]=offsets.try_into().ok().unwrap();
    let mut control=State::new(d,&rv).await;let mut candidate=State::new(d,&rv).await;
    // Candidate executes first into independent poison outputs.
    run(d,true,&projected,&post,&pre,&offsets,&mut candidate).await;
    run(d,false,&projected,&post,&pre,&offsets,&mut control).await;
    exact("all blocked FP8 input",&bytes(d,&control.q).await,&bytes(d,&candidate.q).await);
    exact("all FP32 scales",&bytes(d,&control.scales).await,&bytes(d,&candidate.scales).await);
    let mut error=0f64;let mut reference=0f64;
    for g in 0..4 {
        let a=bytes(d,&control.residual[g]).await;let b=bytes(d,&candidate.residual[g]).await;
        exact(&format!("persisted residual group{g}"),&a,&b);
        for row in [0,1,2,31,63,64,127] {
            let rms=(pv[g][row*3840..(row+1)*3840].iter().map(|&x|(x as f64).powi(2)).sum::<f64>()/3840.0+EPS as f64).sqrt();
            for col in 0..3840 {
                let ix=row*3840+col;
                let expected=bf(bf((pv[g][ix]as f64/rms*postv[col]as f64)as f32)+rv[g][ix]);
                let got=f32::from_bits((u16::from_le_bytes([b[ix*2],b[ix*2+1]])as u32)<<16);
                assert!(got.is_finite());error+=(got as f64-expected as f64).powi(2);reference+=(expected as f64).powi(2);
            }
        }
    }
    let nrmse=(error/reference.max(1e-30)).sqrt();println!("CPU F64 residual seed={seed} nrmse={nrmse}");assert!(nrmse<0.003);
    let mut timing=[Vec::new(),Vec::new()];
    for variant in [false,true,true,false,false,true,true,false] {
        let state=if variant {&mut candidate}else{&mut control};
        state.reset_residual(d,&rv).await;
        let now=Instant::now();run(d,variant,&projected,&post,&pre,&offsets,state).await;
        timing[variant as usize].push(now.elapsed().as_secs_f64()*1e6);
    }
    println!("TIMING seed={seed} order={order:?} control10launch_us={:?} fused1launch_us={:?}",timing[0],timing[1]);
}
#[tokio::main]
async fn main() {
    let mut d=Device::new(fused::post_attention_ffn_input_four.topology()).unwrap();
    test(&mut d,713,[0,1,2,3]).await;
    test(&mut d,1771,[3,1,0,2]).await;
    test(&mut d,5001,[2,0,3,1]).await;
    println!("PASS exact fused postattention/residual/preFFN/FP8 ingress across all512rows, independent allocations, F64 residual, zero/tiny/signed inputs and quarter permutations");
}
