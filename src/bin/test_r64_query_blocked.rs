//! R64 Qs blocked storage: every finite E4 permutation value, exact projection equivalence, and CPU dot oracles.
use furiosa_opt_gemma4::{Chip,axes::{R64,H,Qs},ops_batch_r64,ops_qkv_r64_blocked};
use furiosa_opt_std::prelude::*;
fn mix(i:usize,seed:u64)->u64{let mut x=(i as u64).wrapping_add(seed);x=(x^(x>>30)).wrapping_mul(0xbf58476d1ce4e5b9);x=(x^(x>>27)).wrapping_mul(0x94d049bb133111eb);x^(x>>31)}
fn dx(b:usize,k:usize)->f32{((mix((b%41)*H::SIZE+k,71)%63)as i32-31)as f32/32.0}
fn dw(n:usize,k:usize)->f32{((mix((n%37)*H::SIZE+k,73)%33)as i32-16)as f32/16.0}
fn varied(i:usize,seed:u64,top:f32)->f32{let b=mix(i,seed);(((b>>40)as f32/16777216.0)*2.0-1.0)*top*2f32.powi(-((b%9)as i32))}
macro_rules! probe{($name:ident,$axis:ident,$kernel:ident)=>{async fn $name(device:&mut Device){
 for random in [false,true] {
  let xs:Vec<bf16>=(0..R64::SIZE*H::SIZE).map(|i|bf16::from_f32(if random{varied(i,83,4.0)}else{dx(i/H::SIZE,i%H::SIZE)})).collect();
  let ws:Vec<f8e4m3>=(0..$axis::SIZE*H::SIZE).map(|i|f8e4m3::from_f32(if random{varied(i,89,1.0)}else{dw(i/H::SIZE,i%H::SIZE)})).collect();
  let scales:Vec<bf16>=(0..$axis::SIZE).map(|i|bf16::from_f32(0.25+(mix(i,79)%31)as f32/32.0)).collect();
  let x:HbmTensor<bf16,Chip,m![R64,H]>=HostTensor::<bf16,m![R64,H]>::from_vec(xs.iter().copied()).to_hbm(&mut device.pdma).await.unwrap();
  let w:HbmTensor<f8e4m3,Chip,m![$axis,H]>=HostTensor::<f8e4m3,m![$axis,H]>::from_vec(ws.iter().copied()).to_hbm(&mut device.pdma).await.unwrap();
  let s:HbmTensor<bf16,Chip,m![$axis]>=HostTensor::<bf16,m![$axis]>::from_vec(scales.iter().copied()).to_hbm(&mut device.pdma).await.unwrap();
  let mut out:HbmTensor<bf16,Chip,m![R64,$axis]>=HostTensor::<bf16,m![R64,$axis]>::zero().to_hbm(&mut device.pdma).await.unwrap();
  let mut outp:HbmTensor<bf16,Chip,m![R64,$axis]>=HostTensor::<bf16,m![R64,$axis]>::zero().to_hbm(&mut device.pdma).await.unwrap();
  let mut blocked:HbmTensor<f8e4m3,Chip,ops_qkv_r64_blocked::QueryWeight>=HostTensor::<f8e4m3,ops_qkv_r64_blocked::QueryWeight>::zero().to_hbm(&mut device.pdma).await.unwrap();
  launch(ops_qkv_r64_blocked::reblock_query,(device,&w,&mut blocked)).await.unwrap();
  let permuted:HostTensor<f8e4m3,ops_qkv_r64_blocked::QueryWeight>=blocked.to_host(&mut device.pdma).await.unwrap();let permuted=permuted.into_vec();
  let mut permfails=0;for ng in 0..64{for kg in 0..4{for n in 0..64{for k in 0..960{let dst=(((ng*4+kg)*64+n)*960)+k;let src=(ng*64+n)*H::SIZE+kg*960+k;if permuted[dst].to_f32().to_bits()!=ws[src].to_f32().to_bits(){permfails+=1;}}}}}
  println!("QKV_reblock random={random} checked={} mismatches={permfails}",permuted.len());assert_eq!(permfails,0);
  launch(ops_batch_r64::$kernel,(device,&x,&w,&s,&mut outp)).await.unwrap();let original:HostTensor<bf16,m![R64,$axis]>=outp.to_host(&mut device.pdma).await.unwrap();let original=original.into_vec();
  launch(ops_qkv_r64_blocked::$kernel,(device,&x,&blocked,&s,&mut out)).await.unwrap();let actual:HostTensor<bf16,m![R64,$axis]>=out.to_host(&mut device.pdma).await.unwrap();let actual=actual.into_vec();
  let differences=actual.iter().zip(&original).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();println!("R64_{} random={random} checked={} rowmajor_bit_differences={differences}",stringify!($axis),actual.len());assert_eq!(differences,0);
  if !random {
    let mut dots=vec![0f32;41*37];for b in 0..41{for n in 0..37{dots[b*37+n]=(0..H::SIZE).map(|k|dx(b,k)*dw(n,k)).sum();}}
    let failures=actual.iter().enumerate().filter(|(i,a)|{let n=i%$axis::SIZE;let b=i/$axis::SIZE;let expected=bf16::from_f32(bf16::from_f32(dots[(b%41)*37+n%37]).to_f32()*scales[n].to_f32());a.to_bits()!=expected.to_bits()}).count();println!("R64_{} independent_dyadic_failures={failures}",stringify!($axis));assert_eq!(failures,0);
  } else {
    let(mut failures,mut bitdiff,mut maxabs)=(0,0,0f32);for j in 0..512{let b=(mix(j,97)as usize)%R64::SIZE;let n=(mix(j,101)as usize)%$axis::SIZE;let dot:f64=(0..H::SIZE).map(|k|xs[b*H::SIZE+k].to_f32()as f64*ws[n*H::SIZE+k].to_f32()as f64).sum();let expected=bf16::from_f32(bf16::from_f32(dot as f32).to_f32()*scales[n].to_f32());let got=actual[b*$axis::SIZE+n];let e=(expected.to_f32()-got.to_f32()).abs();bitdiff+=usize::from(expected.to_bits()!=got.to_bits());maxabs=maxabs.max(e);if !got.to_f32().is_finite()||e>0.001+0.01*expected.to_f32().abs(){failures+=1;}}
    println!("R64_{} F64_samples=512 bitdiff={bitdiff} tolerance_failures={failures} max_abs={maxabs}",stringify!($axis));assert_eq!(failures,0);
    let mut times=[Vec::<f64>::new(),Vec::<f64>::new()];for variant in [0,1,1,0]{let start=std::time::Instant::now();for _ in 0..5{if variant==0{launch(ops_batch_r64::$kernel,(device,&x,&w,&s,&mut outp)).await.unwrap();}else{launch(ops_qkv_r64_blocked::$kernel,(device,&x,&blocked,&s,&mut out)).await.unwrap();}}if variant==0{let _:HostTensor<bf16,m![R64,$axis]>=outp.to_host(&mut device.pdma).await.unwrap();}else{let _:HostTensor<bf16,m![R64,$axis]>=out.to_host(&mut device.pdma).await.unwrap();}times[variant].push(start.elapsed().as_secs_f64()*1000.0/5.0);}
    println!("R64_{} rowmajor_ms={:?} blocked_ms={:?} scope=5launches+own_output_read_divided5",stringify!($axis),times[0],times[1]);
  }
 }
}}}
probe!(query,Qs,project_query_batch);
#[tokio::main]async fn main(){let mut device=Device::new(ops_batch_r64::project_query_batch.topology()).unwrap();query(&mut device).await;}
