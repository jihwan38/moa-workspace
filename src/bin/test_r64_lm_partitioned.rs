#[path = "support/npu_profile.rs"] mod npu_profile;
use furiosa_opt_gemma4::{axes::{R64,H,W},LOGIT_SOFTCAP,ops_r64_io_blocked::{self as control,BlockedLogits},ops_r64_lm_partitioned as candidate};
use furiosa_opt_std::prelude::*;
fn word(mut x:u64)->u64{x=(x^(x>>30)).wrapping_mul(0xbf58476d1ce4e5b9);x=(x^(x>>27)).wrapping_mul(0x94d049bb133111eb);x^(x>>31)}
fn hidden(r:usize,h:usize,seed:u64)->bf16 {
 let x=word((r*H::SIZE+h)as u64+37+seed);
 bf16::from_f32(if seed==0 {((x%16)as i32-8)as f32/16.0}else{((x>>40)as f32/16777216.0-0.5)*2.0})
}
fn weight(n:usize,h:usize,seed:u64)->bf16 {
 let x=word(((n%257)*H::SIZE+h)as u64+79+seed*71);
 bf16::from_f32(if seed==0 {((x%16)as i32-8)as f32/128.0}else{((x>>40)as f32/16777216.0-0.5)*0.0625})
}
fn read(raw:&[u8],r:usize,n:usize)->f32 {
 let offset=((((n/8192)*256+(n/32)%256)*64+r)*32+n%32)*2;
 bf16::from_bits(u16::from_le_bytes([raw[offset],raw[offset+1]])).to_f32()
}
#[tokio::main]
async fn main()->Result<(),Box<dyn std::error::Error>> {
 let collector=npu_profile::Collector::default();tracing::subscriber::set_global_default(collector.clone())?;
 let mut d=Device::new(candidate::logits_batch.topology())?;
 let mut output=HostTensor::<bf16,BlockedLogits>::zero().to_hbm(&mut d.pdma).await?;
 let mut reference=HostTensor::<bf16,BlockedLogits>::zero().to_hbm(&mut d.pdma).await?;
 for seed in [0,31,79] {
  let patterns:Vec<Vec<u8>>=(0..257).map(|n|(0..H::SIZE).flat_map(|h|weight(n,h,seed).to_bits().to_le_bytes()).collect()).collect();
  let mut bytes=vec![0u8;W::SIZE*H::SIZE*2];
  for (n,row) in bytes.chunks_exact_mut(H::SIZE*2).enumerate(){row.copy_from_slice(&patterns[n%257]);}
  let weights=HostTensor::<bf16,m![W,H]>::from_buf(bytes).to_hbm(&mut d.pdma).await?;
  let input=HostTensor::<bf16,m![R64,H]>::from_vec((0..64).flat_map(|r|(0..H::SIZE).map(move|h|hidden(r,h,seed)))).to_hbm(&mut d.pdma).await?;
  launch(control::logits_batch,(&mut d,&input,&weights,&mut reference)).await?;
  launch(candidate::logits_batch,(&mut d,&input,&weights,&mut output)).await?;
  let expected=reference.to_host::<BlockedLogits>(&mut d.pdma).await?.into_inner().into_buf();
  let actual=output.to_host::<BlockedLogits>(&mut d.pdma).await?.into_inner().into_buf();
  let mut differences=0usize;let mut max_abs=0f32;let mut e2=0f64;let mut r2=0f64;let mut argmax_differences=0usize;
  for r in 0..64 {
   let mut ca=(0usize,f32::NEG_INFINITY);let mut cb=ca;
   for n in 0..W::SIZE {
    let a=read(&actual,r,n);let b=read(&expected,r,n);assert!(a.is_finite()&&b.is_finite());
    differences+=usize::from(a.to_bits()!=b.to_bits());max_abs=max_abs.max((a-b).abs());e2+=(a as f64-b as f64).powi(2);r2+=(b as f64).powi(2);
    if a.total_cmp(&ca.1).is_gt(){ca=(n,a)}if b.total_cmp(&cb.1).is_gt(){cb=(n,b)}
   }
   if ca.0!=cb.0{println!("LM_ARGMAX_DIFF seed={seed} row={r} candidate={ca:?} control={cb:?}");argmax_differences+=1;}
  }
  println!("LM_PARTITION_DIFFERENTIAL seed={seed} checked={} differences={differences} max_abs={max_abs} relative_l2={} argmax_differences={argmax_differences}",64*W::SIZE,(e2/r2).sqrt());
  if seed==0{assert_eq!(differences,0,"dyadic dots must remain exactly representable");}
  assert_eq!(argmax_differences,0,"synthetic greedy ranking regression");
  let mut cpu_failures=0usize;let mut cpu_max=0f32;
  for r in 0..64 {for sample in 0..16 {
   let n=(word((r*16+sample)as u64+997+seed)as usize)%W::SIZE;
   let dot=(0..H::SIZE).map(|h|hidden(r,h,seed).to_f32()as f64*weight(n,h,seed).to_f32()as f64).sum::<f64>();
   let oracle=bf16::from_f32((bf16::from_f32(dot as f32).to_f32()/LOGIT_SOFTCAP).tanh()*LOGIT_SOFTCAP).to_f32();
   let err=(read(&actual,r,n)-oracle).abs();cpu_max=cpu_max.max(err);cpu_failures+=usize::from(err>0.002+0.01*oracle.abs());
  }}
  println!("LM_PARTITION_F64 seed={seed} sampled=1024 tolerance_failures={cpu_failures} max_abs={cpu_max}");assert_eq!(cpu_failures,0);
  if seed==31 {
   let mut times=[Vec::new(),Vec::new()];
   for variant in [0,1,1,0] {let begin=std::time::Instant::now();for _ in 0..10 {
    if variant==0{launch(control::logits_batch,(&mut d,&input,&weights,&mut reference)).await?;}else{launch(candidate::logits_batch,(&mut d,&input,&weights,&mut output)).await?;}
   }times[variant].push(begin.elapsed().as_secs_f64()*1000.0/10.0);}
   println!("LM_PARTITION_TIMING control_ms={:?} candidate_ms={:?} scope=10completed_launches",times[0],times[1]);
   if std::env::var_os("FURIOSA_OPT_PROFILE").is_some(){for variant in 0..2{println!("PROFILE_VARIANT {variant}");collector.enable(true);if variant==0{launch(control::logits_batch,(&mut d,&input,&weights,&mut reference)).await?;}else{launch(candidate::logits_batch,(&mut d,&input,&weights,&mut output)).await?;}collector.enable(false);}}
  }
 }
 println!("PASS vocabulary-sharded BF16LM: dyadic allbits, random alloutputs/argmax, F64samples; productionnotchanged");Ok(())
}
