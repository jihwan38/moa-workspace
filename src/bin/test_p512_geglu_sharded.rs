//! Exact blocked GeGLU/row quantization gate, candidate first on two fresh inputs.
use furiosa_opt_gemma4::{Chip,axes::{P512,L},ops_batch_layouts::{P512L240,P512L960},ops_native_fp8_p512_blocks as control,ops_native_fp8_p512_blocks_sharded as candidate};
use furiosa_opt_std::prelude::*;
#[path="support/npu_profile.rs"] mod npu_profile;
fn word(i:usize,seed:u64)->u64 {
 let mut x=(i as u64).wrapping_add(seed);
 x=(x^(x>>30)).wrapping_mul(0xbf58476d1ce4e5b9);
 x=(x^(x>>27)).wrapping_mul(0x94d049bb133111eb);x^(x>>31)
}
fn values(seed:u64)->Vec<bf16> {
 let mut out=vec![bf16::from_f32(0.0);P512::SIZE*L::SIZE];
 for cluster in 0..2 {for part in 0..64 {for token in 0..256 {for lane in 0..240 {
  let row=cluster*256+token;let channel=part*240+lane;let i=row*L::SIZE+channel;
  let x=match row%128 {
   0=>if channel%2==0 {0.0}else{-0.0},
   1=>if channel%2==0 {2f32.powi(-127)}else{-2f32.powi(-127)},
   _=>(((word(i,seed)>>40)as f32/16777216.0)*2.0-1.0)*8.0*2f32.powi(-((word(row,seed+2)%9)as i32)),
  };
  out[(((cluster*64+part)*256+token)*240)+lane]=bf16::from_f32(x);
 }}}}
 out
}
async fn upload<D:MaterializableScalar,E:M>(device:&mut Device,values:impl IntoIterator<Item=D>)->HbmTensor<D,Chip,E> {
 HostTensor::<D,E>::from_vec(values).to_hbm(&mut device.pdma).await.unwrap()
}
#[tokio::main]
async fn main(){
 let collector=npu_profile::Collector::default();tracing::subscriber::set_global_default(collector.clone()).unwrap();
 let mut device=Device::new(candidate::geglu_quantize.topology()).unwrap();
 let ug:HbmTensor<f32,Chip,m![1]>=upload(&mut device,[0.137f32]).await;
 let gg:HbmTensor<f32,Chip,m![1]>=upload(&mut device,[0.183f32]).await;
 let mut qc:HbmTensor<f8e4m3,Chip,P512L960>=HostTensor::<f8e4m3,P512L960>::zero().to_hbm(&mut device.pdma).await.unwrap();
 let mut qr:HbmTensor<f8e4m3,Chip,P512L960>=HostTensor::<f8e4m3,P512L960>::zero().to_hbm(&mut device.pdma).await.unwrap();
 let mut sc:HbmTensor<f32,Chip,m![P512]>=HostTensor::<f32,m![P512]>::zero().to_hbm(&mut device.pdma).await.unwrap();
 let mut sr:HbmTensor<f32,Chip,m![P512]>=HostTensor::<f32,m![P512]>::zero().to_hbm(&mut device.pdma).await.unwrap();
 for case in 0..2 {
  let up:HbmTensor<bf16,Chip,P512L240>=upload(&mut device,values(31+case*100)).await;
  let gate:HbmTensor<bf16,Chip,P512L240>=upload(&mut device,values(73+case*100)).await;
  // Candidate precedes the control in every case: stale control DM cannot prove replication.
  launch(candidate::geglu_quantize,(&mut device,&up,&gate,&ug,&gg,&mut qc,&mut sc)).await.unwrap();
  let qch:HostTensor<f8e4m3,P512L960>=qc.to_host(&mut device.pdma).await.unwrap();let qch=qch.into_vec();
  let sch:HostTensor<f32,m![P512]>=sc.to_host(&mut device.pdma).await.unwrap();let sch=sch.into_vec();
  launch(control::geglu_quantize,(&mut device,&up,&gate,&ug,&gg,&mut qr,&mut sr)).await.unwrap();
  let qrh:HostTensor<f8e4m3,P512L960>=qr.to_host(&mut device.pdma).await.unwrap();let qrh=qrh.into_vec();
  let srh:HostTensor<f32,m![P512]>=sr.to_host(&mut device.pdma).await.unwrap();let srh=srh.into_vec();
  let diff=qch.iter().zip(&qrh).filter(|(a,b)|a.to_f32().to_bits()!=b.to_f32().to_bits()).count();
  let sdiff=sch.iter().zip(&srh).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();
  println!("P512_GeGLU_sharded case={case} q_checked={} q_bitdiff={diff} scales_checked={} scales_bitdiff={sdiff}",qch.len(),sch.len());
  for(i,(a,b))in qch.iter().zip(&qrh).enumerate().filter(|(_, (a,b))|a.to_f32().to_bits()!=b.to_f32().to_bits()).take(12){eprintln!("qdiff index={i} candidate={} control={}",a.to_f32(),b.to_f32());}
  for(i,(a,b))in sch.iter().zip(&srh).enumerate().filter(|(_, (a,b))|a.to_bits()!=b.to_bits()).take(12){eprintln!("scalediff row={i} candidate={a} control={b}");}
  assert_eq!(diff,0);assert_eq!(sdiff,0);
  if case==1 {
   let mut times=[Vec::new(),Vec::new()];
   for variant in [0,1,1,0]{let start=std::time::Instant::now();for _ in 0..10 {
    if variant==0 {launch(control::geglu_quantize,(&mut device,&up,&gate,&ug,&gg,&mut qr,&mut sr)).await.unwrap();}
    else {launch(candidate::geglu_quantize,(&mut device,&up,&gate,&ug,&gg,&mut qc,&mut sc)).await.unwrap();}
   }times[variant].push(start.elapsed().as_secs_f64()*1000.0/10.0);}
   println!("P512_GeGLU_sharded control_ms={:?} candidate_ms={:?} scope=10_completed_launches_no_host_read",times[0],times[1]);
   if std::env::var_os("FURIOSA_OPT_PROFILE").is_some(){
    println!("PROFILE_VARIANT control");collector.enable(true);launch(control::geglu_quantize,(&mut device,&up,&gate,&ug,&gg,&mut qr,&mut sr)).await.unwrap();collector.enable(false);
    println!("PROFILE_VARIANT sharded");collector.enable(true);launch(candidate::geglu_quantize,(&mut device,&up,&gate,&ug,&gg,&mut qc,&mut sc)).await.unwrap();collector.enable(false);
   }
  }
 }
}
