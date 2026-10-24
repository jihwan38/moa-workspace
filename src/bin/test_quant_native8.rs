//! Independent host formula and every-byte check for runtime activation quantizers.
use furiosa_opt_gemma4::{Chip,axes::{P128,H,L},ops_quant_native8};
use furiosa_opt_std::prelude::*;
fn word(i:usize)->u64{let mut x=i as u64+77;x=(x^(x>>30)).wrapping_mul(0xbf58476d1ce4e5b9);x=(x^(x>>27)).wrapping_mul(0x94d049bb133111eb);x^(x>>31)}
macro_rules! check {
 ($device:ident,$axis:ident,$kernel:ident)=>{{
  let k=$axis::SIZE;
  let x:Vec<bf16>=(0..128*k).map(|i| {let row=i/k;let exp=[-120,-20,-5,0,20,110][row%6];bf16::from_f32(if row%17==0 {0.}else{((word(i)%1023) as f32-511.)/128.*2f32.powi(exp)})}).collect();
  let mut scale=Vec::new();let mut q=Vec::new();let mut e2=0f64;let mut x2=0f64;
  for row in x.chunks(k){let amax=row.iter().map(|x|x.to_f32().abs()).fold(0f32,f32::max).max(2f32.powi(-112));let s=f32::from_bits(amax.to_bits()&0x7f800000)/128.;scale.push(s);for x in row{let original=x.to_f32();let flushed=if original.abs()<f32::MIN_POSITIVE{f32::from_bits(original.to_bits()&0x8000_0000)}else{original};let qi=f8e4m3::from_f32(flushed/s);q.push(qi);let xf=x.to_f32()as f64;let diff=qi.to_f32()as f64*s as f64-xf;e2+=diff*diff;x2+=xf*xf;}}
  let input=HostTensor::<bf16,m![P128,$axis]>::from_vec(x.iter().copied()).to_hbm(&mut $device.pdma).await?;
  let mut output:HbmTensor<f8e4m3,Chip,m![P128,$axis]>=HostTensor::<f8e4m3,m![P128,$axis]>::zero().to_hbm(&mut $device.pdma).await?;
  let mut scales:HbmTensor<f32,Chip,m![P128]>=HostTensor::<f32,m![P128]>::zero().to_hbm(&mut $device.pdma).await?;
  launch(ops_quant_native8::$kernel,(&mut $device,&input,&mut output,&mut scales)).await?;
  let aq:HostTensor<f8e4m3,m![P128,$axis]>=output.to_host(&mut $device.pdma).await?;
  let asc:HostTensor<f32,m![P128]>=scales.to_host(&mut $device.pdma).await?;
  let actual=aq.into_vec();
  for (i,(e,a)) in q.iter().zip(&actual).enumerate().filter(|(_, (e,a))| e.to_f32().to_bits()!=a.to_f32().to_bits()).take(12){println!("mismatch i={i} row={} x={} scale={} expected={} actual={}",i/k,x[i].to_f32(),scale[i/k],e.to_f32(),a.to_f32());}
  let qdiff=q.iter().zip(actual).filter(|(a,b)|a.to_f32().to_bits()!=b.to_f32().to_bits()).count();
  let sdiff=scale.iter().zip(asc.into_vec()).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();
  assert_eq!(qdiff,0,"quantized bytes differ");assert_eq!(sdiff,0,"scales differ");
  let start=std::time::Instant::now();for _ in 0..10 {launch(ops_quant_native8::$kernel,(&mut $device,&input,&mut output,&mut scales)).await?;}let _:HostTensor<f8e4m3,m![P128,$axis]>=output.to_host(&mut $device.pdma).await?;
  println!("quantize k={k} exact_bytes={} exact_scales={} approximate_reconstruction_nrmse={} warm_ms={} scope=10launch+1read",q.len(),scale.len(),(e2/x2).sqrt(),start.elapsed().as_secs_f64()*1000./10.);
 }};
}
#[tokio::main]
async fn main()->Result<(),Box<dyn std::error::Error>>{let mut device=Device::new(ops_quant_native8::quantize_hidden.topology())?;check!(device,H,quantize_hidden);check!(device,L,quantize_ffn);Ok(())}
