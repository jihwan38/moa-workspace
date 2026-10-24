//! Real model blocked logits: device-selected greedy tokens versus exact CPU sampler.
use furiosa_opt_gemma4::{axes::*,host::{greedy::GreedySelector,load,model::{ForwardOutput,Input,Model,Sequence},runtime::{Blocks,Workspace},sampling::{sample,SamplingConfig}},ops};
use furiosa_opt_std::prelude::*;
use std::{collections::HashSet,time::Instant};
#[tokio::main]
async fn main()->Result<(),Box<dyn std::error::Error>> {
 let mut device=Device::new(ops::embed_token.topology())?;
 let weights=load::load_weights(&mut device).await?;
 let mut ws=Workspace::new_canonical64(&mut device,&weights,512,64).await?;
 let mut model=Model::new(&mut device,weights,&mut ws).await?;
 let mut selector=GreedySelector::new(&mut device).await?;
 let blocks:Vec<_>=(0..64).map(|r|Blocks{sliding:(r*13+31)%64,full:vec![63-r]}).collect();
 let mut rng=rand::rng();
 for width in [17,31,64] { for step in 0..3 {
  let rows:Vec<_>=(0..width).map(|r|[Input{token:2+(r*313+step*37)%8000,soft:None,logits:true,image:None,audio:None}]).collect();
  let sequences:Vec<_>=(0..width).map(|r|Sequence{inputs:&rows[r],start_position:step,blocks:&blocks[r]}).collect();
  let start=Instant::now();
  let ForwardOutput::Blocked64(output,live)=model.forward_outputs(&mut device,&sequences,&mut ws).await? else {return Err("expected R64 blocked output".into());};
  let model_ms=start.elapsed().as_secs_f64()*1000.;
  let mut bans:Vec<HashSet<usize>>=(0..live).map(|r| if r%2==0 {[258882,258883].into_iter().collect()}else{[258882,258883,1,50,100,101,106].into_iter().collect()}).collect();
  let start=Instant::now();
  let selections=selector.select(&mut device,output,&bans.iter().map(Some).collect::<Vec<_>>()).await?;
  let select_ms=start.elapsed().as_secs_f64()*1000.;
  let host=output.to_host::<furiosa_opt_gemma4::ops_r64_io_blocked::BlockedLogits>(&mut device.pdma).await?;
  let reference=furiosa_opt_gemma4::host_logit_bytes::decode_blocked_r64(&host.into_inner().into_buf(),live,8);
  let mut fallback=0;
  for row in 0..live {
   let expected=sample(&reference[row],&SamplingConfig::DEFAULT,&bans[row],&mut rng);
   let actual=match selections[row] {Some(t)=>t,None=>{fallback+=1;let values=selector.read_fallback(&mut device,output,row).await?;assert_eq!(values.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),reference[row].iter().map(|x|x.to_bits()).collect::<Vec<_>>());sample(&values,&SamplingConfig::DEFAULT,&bans[row],&mut rng)}};
   assert_eq!(actual,expected,"real model row{row}");bans[row].insert(expected);
  }
  let selections=selector.select(&mut device,output,&bans.iter().map(Some).collect::<Vec<_>>()).await?;
  for row in 0..live {
   let expected=sample(&reference[row],&SamplingConfig::DEFAULT,&bans[row],&mut rng);
   let actual=match selections[row] {Some(t)=>t,None=>{let values=selector.read_fallback(&mut device,output,row).await?;sample(&values,&SamplingConfig::DEFAULT,&bans[row],&mut rng)}};
   assert_eq!(actual,expected,"real model suppressed winner row{row}");
  }
  println!("MODEL_GREEDY_EXACT width={width} step={step} normal_and_suppressed_winner=true fallback={fallback} model_ms={model_ms:.3} select_with_ban_upload_ms={select_ms:.3} serving_compute_ms={:.3}",model_ms+select_ms);
 }}
 println!("PASS real model device greedy across17/31/64 rows, histories, physical regroup, and changed bans");Ok(())
}
