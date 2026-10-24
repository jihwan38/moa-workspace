//! P128 prepacked FP8 FFN matrix bound, with independent sampled F64 oracle.
use furiosa_opt_gemma4::{Chip,axes::{P128,H,L},ops_native8_wide};
use furiosa_opt_std::prelude::*;
fn word(i:usize,seed:u64)->u64 {let mut x=(i as u64).wrapping_add(seed);x=(x^(x>>30)).wrapping_mul(0xbf58476d1ce4e5b9);x=(x^(x>>27)).wrapping_mul(0x94d049bb133111eb);x^(x>>31)}
fn vals(n:usize,seed:u64)->Vec<f8e4m3>{(0..n).map(|i|f8e4m3::from_f32((word(i,seed)%129)as f32/128.-0.5)).collect()}
fn check(x:&[f8e4m3],w:&[f8e4m3],y:&[bf16],k:usize,n:usize){let mut diff=0;let mut max=0f32;for i in 0..1024{let r=word(i,19)as usize%128;let c=word(i,23)as usize%n;let mut sum=0f64;for j in 0..k{sum+=x[r*k+j].to_f32()as f64*w[c*k+j].to_f32()as f64;}let a=bf16::from_f32(sum as f32).to_f32();let b=y[r*n+c].to_f32();diff+=usize::from(a!=b);max=max.max((a-b).abs());assert!((a-b).abs()<=0.01+0.01*a.abs(),"oracle r{r} c{c}: {a} vs {b}");}println!("oracle k={k} n={n} samples=1024 bitdiff={diff} max_abs={max}");}
#[tokio::main]
async fn main()->Result<(),Box<dyn std::error::Error>>{
 let mut d=Device::new(ops_native8_wide::up_native8.topology())?;
 let xv=vals(P128::SIZE*H::SIZE,1);let wv=vals(L::SIZE*H::SIZE,2);
 let x=HostTensor::<f8e4m3,m![P128,H]>::from_vec(xv.iter().copied()).to_hbm(&mut d.pdma).await?;
 let w=HostTensor::<f8e4m3,m![L,H]>::from_vec(wv.iter().copied()).to_hbm(&mut d.pdma).await?;
 let mut o:HbmTensor<bf16,Chip,m![P128,L]>=HostTensor::<bf16,m![P128,L]>::zero().to_hbm(&mut d.pdma).await?;
 launch(ops_native8_wide::up_native8,(&mut d,&x,&w,&mut o)).await?;
 let y:HostTensor<bf16,m![P128,L]>=o.to_host(&mut d.pdma).await?;check(&xv,&wv,&y.into_vec(),H::SIZE,L::SIZE);
 let mut ts=Vec::new();for _ in 0..4{let now=std::time::Instant::now();for _ in 0..10{launch(ops_native8_wide::up_native8,(&mut d,&x,&w,&mut o)).await?;}let _:HostTensor<bf16,m![P128,L]>=o.to_host(&mut d.pdma).await?;ts.push(now.elapsed().as_secs_f64()*1000./10.);}println!("nativeP128up ms={ts:?} scope=10launch+1read prequantization=excluded");
 drop((x,w,o,xv,wv));
 let xv=vals(P128::SIZE*L::SIZE,3);let wv=vals(H::SIZE*L::SIZE,5);
 let x=HostTensor::<f8e4m3,m![P128,L]>::from_vec(xv.iter().copied()).to_hbm(&mut d.pdma).await?;
 let w=HostTensor::<f8e4m3,m![H,L]>::from_vec(wv.iter().copied()).to_hbm(&mut d.pdma).await?;
 let mut o:HbmTensor<bf16,Chip,m![P128,H]>=HostTensor::<bf16,m![P128,H]>::zero().to_hbm(&mut d.pdma).await?;
 launch(ops_native8_wide::down_native8,(&mut d,&x,&w,&mut o)).await?;
 let y:HostTensor<bf16,m![P128,H]>=o.to_host(&mut d.pdma).await?;check(&xv,&wv,&y.into_vec(),L::SIZE,H::SIZE);
 let mut ts=Vec::new();for _ in 0..4{let now=std::time::Instant::now();for _ in 0..10{launch(ops_native8_wide::down_native8,(&mut d,&x,&w,&mut o)).await?;}let _:HostTensor<bf16,m![P128,H]>=o.to_host(&mut d.pdma).await?;ts.push(now.elapsed().as_secs_f64()*1000./10.);}println!("nativeP128down ms={ts:?} scope=10launch+1read prequantization=excluded");Ok(())}
