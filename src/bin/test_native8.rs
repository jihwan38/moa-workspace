//! Native FP8 speed bound with exactly representable input, checked against BF16.
use furiosa_opt_gemma4::{Chip,axes::{B,H,Qs},ops_batch_clusters,ops_native8};
use furiosa_opt_std::prelude::*;
fn word(i:usize,seed:u64)->u64 { let mut x=(i as u64).wrapping_add(seed);x=(x^(x>>30)).wrapping_mul(0xbf58476d1ce4e5b9);x=(x^(x>>27)).wrapping_mul(0x94d049bb133111eb);x^(x>>31) }
#[tokio::main]
async fn main()->Result<(),Box<dyn std::error::Error>> {
 let mut d=Device::new(ops_native8::project_query_native8.topology())?;
 let xv:Vec<f8e4m3>=(0..B::SIZE*H::SIZE).map(|i|f8e4m3::from_f32((word(i,37)%257) as f32/256.-0.5)).collect();
 let x8=HostTensor::<f8e4m3,m![B,H]>::from_vec(xv.iter().copied()).to_hbm(&mut d.pdma).await?;
 let xb=HostTensor::<bf16,m![B,H]>::from_vec(xv.iter().map(|x|bf16::from_f32(x.to_f32()))).to_hbm(&mut d.pdma).await?;
 let w=HostTensor::<f8e4m3,m![Qs,H]>::from_vec((0..Qs::SIZE*H::SIZE).map(|i|f8e4m3::from_f32((word(i,41)%33) as f32/32.-0.5))).to_hbm(&mut d.pdma).await?;
 let s=HostTensor::<bf16,m![Qs]>::from_vec((0..Qs::SIZE).map(|i|bf16::from_f32(((word(i,43)%8)+1) as f32/16.))).to_hbm(&mut d.pdma).await?;
 let mut o:HbmTensor<bf16,Chip,m![B,Qs]>=HostTensor::<bf16,m![B,Qs]>::zero().to_hbm(&mut d.pdma).await?;
 launch(ops_batch_clusters::project_query_clusters,(&mut d,&xb,&w,&s,&mut o)).await?;
 let control:HostTensor<bf16,m![B,Qs]>=o.to_host(&mut d.pdma).await?;
 launch(ops_native8::project_query_native8,(&mut d,&x8,&w,&s,&mut o)).await?;
 let cand:HostTensor<bf16,m![B,Qs]>=o.to_host(&mut d.pdma).await?;
 let mismatches=control.into_vec().iter().zip(cand.into_vec()).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();
 assert_eq!(mismatches,0);
 let mut times=[Vec::new(),Vec::new()];
 for variant in [0,1,1,0,0,1,1,0] {
 let start=std::time::Instant::now();
 for _ in 0..10 { if variant==0 {launch(ops_batch_clusters::project_query_clusters,(&mut d,&xb,&w,&s,&mut o)).await?;} else {launch(ops_native8::project_query_native8,(&mut d,&x8,&w,&s,&mut o)).await?;} }
 let _:HostTensor<bf16,m![B,Qs]>=o.to_host(&mut d.pdma).await?;
 times[variant].push(start.elapsed().as_secs_f64()*1000./10.);
 }
 println!("native8 query exact={} bf16_ms={:?} f8_ms={:?} scope=10launches+1read quantization=excluded",B::SIZE*Qs::SIZE,times[0],times[1]);Ok(())
}
