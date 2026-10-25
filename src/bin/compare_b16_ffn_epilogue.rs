//! Exact B16 post-attention/FFN launch composition with disjoint pools and full KV oracle.
use furiosa_opt_gemma4::{axes::W, host::{load, model::{Input,Logits,Model,Sequence}, runtime::{Blocks,Workspace}},ops};
use furiosa_opt_std::prelude::*;
use std::{time::Instant,collections::HashSet};
use rand::{SeedableRng,rngs::StdRng};
use furiosa_opt_gemma4::host::sampling::{sample,SamplingConfig};
type Result<T>=std::result::Result<T,Box<dyn std::error::Error>>;
fn input(request:usize,position:usize,logits:bool)->Input<'static>{Input{token:2+(request*313+position*37)%8000,soft:None,logits,image:None,audio:None}}
async fn raw(d:&mut Device,x:&Logits)->Result<Vec<u16>>{
 let h=x.to_host::<m![W]>(&mut d.pdma).await?.into_inner().into_buf();
 let values:Vec<_>=h.chunks_exact(2).map(|x|u16::from_le_bytes([x[0],x[1]])).collect();
 assert_eq!(values.len(),W::SIZE);assert!(values.iter().all(|x|x&0x7f80!=0x7f80),"nonfinite logits");Ok(values)
}
fn selected(step:usize)->Vec<usize>{match step{
 0=>vec![0],1=>vec![1,16],2=>(0..8).map(|i|(i*7+2)%17).collect(),
 3=>(0..10).map(|i|(i*3+4)%17).collect(),4=>(0..16).rev().collect(),
 5=>(0..17).collect(),6=>vec![2,16],7=>(0..10).map(|i|(i*5+2)%17).collect(),
 _=>(0..17).rev().collect(),
}}
fn check(pass:usize,step:usize,request:usize,a:&[u16],b:&[u16]){
 assert_eq!(a.len(),b.len());let mut differences=0;
 for(i,(&a,&b))in a.iter().zip(b).enumerate(){if a!=b{if differences<6{eprintln!("FFN_DIFF pass={pass} step={step} request={request} token={i} original={a:04x} staged_fused={b:04x}");}differences+=1;}}
 assert_eq!(differences,0,"B16 complete epilogue changed final model logits");
}
#[tokio::main]
async fn main()->Result<()>{
 let started=Instant::now();let mut d=Device::new(ops::embed_token.topology())?;
 let weights=load::load_weights(&mut d).await?;
 let mut ws=Workspace::new_canonical64(&mut d,&weights,2048,64).await?;
 let mut model=Model::new(&mut d,weights,&mut ws).await?;model.prepare_group_scratch(&mut d).await?;
 let mut expected:Vec<Vec<Vec<u16>>>=Vec::new();let mut checked=0usize;
 let mut reservations:Vec<Vec<Blocks>>=Vec::new();
 for pass in 0..2{
  model.set_b16_epilogue_fused(pass==1);
  let blocks:Vec<_>=(0..17).map(|r|{let slot=if pass==0{r}else{32+r};Blocks{sliding:slot,full:(0..4).map(|p|if pass==0{r*4+p}else{255-(r*4+p)}).collect()}}).collect();
  reservations.push(blocks.clone());
  let fill=Instant::now();
  for group in 0..4{for pos in (0..512).step_by(128){
   let ins:Vec<Vec<_>>=(0..4).map(|i|(pos..pos+128).map(|p|input(group*4+i,p,false)).collect()).collect();
   let seq:Vec<_>=(0..4).map(|i|Sequence{inputs:&ins[i],start_position:pos,blocks:&blocks[group*4+i]}).collect();
   model.start_prefill_group(&mut d,&seq,&mut ws).await?;
   while model.advance_prefill_group(&mut d,&mut ws).await?.is_none(){}
  }}
  for pos in (0..512).step_by(128) {
   let ins:Vec<_>=(pos..pos+128).map(|p|input(16,p,false)).collect();
   model.forward_sequences(&mut d,&[Sequence{inputs:&ins,start_position:pos,blocks:&blocks[16]}],&mut ws).await?;
  }
  println!("B16_EPILOGUE_FILL pass={pass} actual_history_tokens=8704 ms={:.3}",fill.elapsed().as_secs_f64()*1000.0);
  let mut positions=[512usize;17];
  for step in 0..9{
   let chosen=selected(step);
   if step==6{for &r in &chosen{positions[r]=0;}}
   // Interleave a short prompt after decode to exercise Prefill's retained canonical raw buffers.
   if step==7{
    let request=2;let ins:Vec<_>=(positions[request]..positions[request]+31).map(|p|input(request,p,false)).collect();
    model.forward_sequences(&mut d,&[Sequence{inputs:&ins,start_position:positions[request],blocks:&blocks[request]}],&mut ws).await?;
    positions[request]+=31;
   }
   let ins:Vec<_>=chosen.iter().map(|&r|[input(r,positions[r],true)]).collect();
   let seq:Vec<_>=chosen.iter().enumerate().map(|(i,&r)|Sequence{inputs:&ins[i],start_position:positions[r],blocks:&blocks[r]}).collect();
   model.begin_stage_profile();let start=Instant::now();let out=model.forward_sequences(&mut d,&seq,&mut ws).await?;
   let model_ms=start.elapsed().as_secs_f64()*1000.0;let mut found=Vec::new();
   for(i,x)in out.iter().enumerate(){let values=raw(&mut d,x).await?;if pass==1{check(pass,step,chosen[i],&expected[step][i],&values);sampling_check(step,chosen[i],&expected[step][i],&values);checked+=values.len();}found.push(values);}
   if pass==0{expected.push(found);}
   println!("B16_EPILOGUE_STEP pass={pass} staged_fused={} step={step} live={} model_ms={model_ms:.3} logits_bitdiff=0",pass==1,chosen.len());
   for stage in model.take_stage_profile(){println!("B16_EPILOGUE_STAGE pass={pass} step={step} stage={} calls={} total_ms={:.6}",stage.stage,stage.calls,stage.milliseconds);}
   for r in chosen{positions[r]+=1;}
  }
 }
 for owner in 0..17 {
  let values=model.compare_canonical_kv(&mut d,&ws,&reservations[0][owner],&reservations[1][owner]).await?;
  println!("B16_EPILOGUE_CACHE owner={owner} bf16_values={values} bitdiff=0");
 }
 println!("PASS B16 complete epilogue actual model checked={checked} BF16logits;scalar width1, B16 widths2/8/10/16 and R64 width17;512-token histories,physical pool remap,full-page transition,position0 reuse,and31-token prompt interleave;CPU greedy and seeded stochastic parity;elapsed_s={:.3};timing diagnostic not HTTP/SLO",started.elapsed().as_secs_f64());Ok(())
}

fn sampling_check(step:usize,request:usize,a:&[u16],b:&[u16]) {
 let a:Vec<f32>=a.iter().map(|&x|furiosa_opt_gemma4::host_logit_bytes::bf16_bits_to_f32(x)).collect();
 let b:Vec<f32>=b.iter().map(|&x|furiosa_opt_gemma4::host_logit_bytes::bf16_bits_to_f32(x)).collect();
 let bans:HashSet<usize>=[0,1,106,107].into_iter().collect();
 for config in [SamplingConfig::DEFAULT,SamplingConfig{temperature:0.8,top_k:37,top_p:0.9}] {
  let seed=(step as u64)*1009+request as u64;
  let mut ar=StdRng::seed_from_u64(seed);let mut br=StdRng::seed_from_u64(seed);
  let first=sample(&a,&config,&bans,&mut ar);let second=sample(&b,&config,&bans,&mut br);
  assert_eq!(first,second,"sampling parity step={step} request={request} temperature={}",config.temperature);
 }
}
