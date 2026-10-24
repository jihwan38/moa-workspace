//! Diagnostic actual-model decoder stage costs, including host reads and sampling.
use furiosa_opt_gemma4::{axes::*,host::{generate::read_logits,load,model::{ForwardOutput,Input,Model,Sequence},runtime::{Blocks,Workspace},sampling::{sample,SamplingConfig}},ops};
use furiosa_opt_std::prelude::*;
use std::{collections::HashSet,time::Instant};
#[tokio::main]
async fn main()->Result<(),Box<dyn std::error::Error>> {
 let mut device=Device::new(ops::embed_token.topology())?;
 let weights=load::load_weights(&mut device).await?;
 let mut ws=Workspace::new_canonical64(&mut device,&weights,512,64).await?;
 let mut model=Model::new(&mut device,weights,&mut ws).await?;
 let blocks:Vec<_>=(0..64).map(|r|Blocks{sliding:r,full:vec![r]}).collect();
 let banned:HashSet<usize>=[258882,258883].into_iter().collect();let mut rng=rand::rng();
 for width in [1,2,4,8,16,17,32,64] {
  for step in 0..3 {
   let rows:Vec<_>=(0..width).map(|r|[Input{token:2+(r*313+step*37)%8000,soft:None,logits:true,image:None,audio:None}]).collect();
   let sequences:Vec<_>=(0..width).map(|r|Sequence{inputs:&rows[r],start_position:step,blocks:&blocks[r]}).collect();
   model.begin_stage_profile();let start=Instant::now();
   let output=model.forward_outputs(&mut device,&sequences,&mut ws).await?;
   let model_ms=start.elapsed().as_secs_f64()*1000.0;let start=Instant::now();
   let logits=match output {
    ForwardOutput::Rows(outputs)=>{let mut out=Vec::new();for row in outputs {out.push(read_logits(&mut device,row).await?);}out},
    ForwardOutput::Blocked64(output,live)=>{let host=output.to_host::<furiosa_opt_gemma4::ops_r64_io_blocked::BlockedLogits>(&mut device.pdma).await?;furiosa_opt_gemma4::host_logit_bytes::decode_blocked_r64(&host.into_inner().into_buf(),live,8)}
   };
   let read_ms=start.elapsed().as_secs_f64()*1000.0;let start=Instant::now();
   let tokens:Vec<_>=logits.iter().map(|row|sample(row,&SamplingConfig::DEFAULT,&banned,&mut rng)).collect();
   let sample_ms=start.elapsed().as_secs_f64()*1000.0;
   assert!(tokens.iter().all(|&t|t<W::SIZE));
   println!("width={width} step={step} model_ms={model_ms:.3} read_ms={read_ms:.3} sample_ms={sample_ms:.3} total_ms={:.3}",model_ms+read_ms+sample_ms);
   if step==2 {for s in model.take_stage_profile(){println!("width={width} stage={} calls={} ms={:.3}",s.stage,s.calls,s.milliseconds);}}
  }
 }
 Ok(())
}
