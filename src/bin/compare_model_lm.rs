//! Actual provided weights and actual 128-token per-request histories; only LM sum association differs.
use furiosa_opt_gemma4::{axes::*,host::{load,model::{Input,Model,Sequence},runtime::{Blocks,Workspace}},ops};
use furiosa_opt_std::prelude::*;
use std::collections::HashSet;
fn input(r:usize,p:usize,logits:bool)->Input<'static>{Input{token:2+(r*313+p*37)%8000,soft:None,logits,image:None,audio:None}}
fn value(raw:&[u8],row:usize,token:usize)->f32 {
 let i=((((token/8192)*256+(token/32)%256)*64+row)*32+token%32)*2;
 bf16::from_bits(u16::from_le_bytes([raw[i],raw[i+1]])).to_f32()
}
fn winner(raw:&[u8],row:usize,bans:&HashSet<usize>)->usize {
 let mut best=(usize::MAX,f32::NEG_INFINITY);
 for token in 0..W::SIZE {let v=value(raw,row,token);if (best.0==usize::MAX||v.total_cmp(&best.1).is_gt())&&!bans.contains(&token){best=(token,v);}}
 assert_ne!(best.0,usize::MAX);best.0
}
#[tokio::main]
async fn main()->Result<(),Box<dyn std::error::Error>> {
 let mut device=Device::new(ops::embed_token.topology())?;
 let weights=load::load_weights(&mut device).await?;
 let mut ws=Workspace::new_canonical64(&mut device,&weights,512,64).await?;
 let mut model=Model::new(&mut device,weights,&mut ws).await?;
 let blocks:Vec<_>=(0..64).map(|r|Blocks{sliding:(r*13+31)%64,full:vec![255-r]}).collect();
 for r in 0..64 {
  let inputs:Vec<_>=(0..128).map(|p|input(r,p,false)).collect();
  model.forward_sequences(&mut device,&[Sequence{inputs:&inputs,start_position:0,blocks:&blocks[r]}],&mut ws).await?;
 }
 let mut positions=[128usize;64];let mut checked=0usize;
 for width in [17,31,64] {for step in 0..3 {
  let chosen:Vec<_>=(0..width).map(|r|if step==1{63-r}else{r}).collect();
  let rows:Vec<_>=chosen.iter().map(|&r|[input(r,positions[r],true)]).collect();
  let sequences:Vec<_>=chosen.iter().enumerate().map(|(row,&r)|Sequence{inputs:&rows[row],start_position:positions[r],blocks:&blocks[r]}).collect();
  let (control,candidate,times)=model.compare_lm_projections(&mut device,&sequences,&mut ws).await?;
  let mut diff=0usize;let mut max_abs=0f32;let mut e2=0f64;let mut r2=0f64;let mut argdiff=0usize;
  for row in 0..width {
   for token in 0..W::SIZE {let a=value(&control,row,token);let b=value(&candidate,row,token);assert!(a.is_finite()&&b.is_finite());diff+=usize::from(a.to_bits()!=b.to_bits());max_abs=max_abs.max((a-b).abs());e2+=(a as f64-b as f64).powi(2);r2+=(a as f64).powi(2);}
   let mut bans:HashSet<_>=[258882,258883].into_iter().collect();
   if row%2==1{bans.extend([1,50,100,101,106]);}
   for suppressed in 0..2 {let a=winner(&control,row,&bans);let b=winner(&candidate,row,&bans);if a!=b{println!("MODEL_LM_ARGMAX_DIFF width={width} step={step} row={row} suppressed={suppressed} control={a} candidate={b}");argdiff+=1;}bans.insert(a);}
  }
  checked+=width*W::SIZE;
  println!("MODEL_LM_RESULT width={width} step={step} checked={} differences={diff} max_abs={max_abs} relative_l2={} normal_suppressed_argmax_differences={argdiff} control_ms={:.6} candidate_ms={:.6}",width*W::SIZE,(e2/r2).sqrt(),times[0],times[1]);
  assert_eq!(argdiff,0,"provided-model vocabulary ranking changed");
  for r in chosen{positions[r]+=1;}
 }}
 println!("PASS actual model LM all_logits={checked},64distinct128tokenprefixes,17/31/64regroup,normal/suppressedwinners; productionheadunchanged");Ok(())
}
