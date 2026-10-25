//! Same-arithmetic whole-model storage-layout gate with real independent histories.
use furiosa_opt_gemma4::{axes::W, host::{load, model::{Input,Logits,Model,Sequence}, runtime::{Blocks,Workspace}},ops};
use furiosa_opt_std::prelude::*;
use std::time::Instant;
type Result<T>=std::result::Result<T,Box<dyn std::error::Error>>;
fn input(request:usize,position:usize,logits:bool)->Input<'static>{Input{token:2+(request*313+position*37)%8000,soft:None,logits,image:None,audio:None}}
async fn raw(d:&mut Device,x:&Logits)->Result<Vec<u16>>{
 let h=x.to_host::<m![W]>(&mut d.pdma).await?.into_inner().into_buf();
 let values:Vec<_>=h.chunks_exact(2).map(|x|u16::from_le_bytes([x[0],x[1]])).collect();
 assert_eq!(values.len(),W::SIZE);assert!(values.iter().all(|x|x&0x7f80!=0x7f80),"nonfinite logits");Ok(values)
}
fn selected(step:usize)->Vec<usize>{match step{
 0=>(0..64).collect(),1=>vec![3,27,55,63],2=>(0..17).map(|i|(i*7+5)%64).collect(),
 3=>(16..32).rev().collect(),4=>(0..64).rev().collect(),5=>vec![3,27,55,63],
 _=>(0..17).map(|i|(i*11+3)%64).collect(),
}}
fn check(pass:usize,step:usize,request:usize,a:&[u16],b:&[u16]){
 assert_eq!(a.len(),b.len());let mut differences=0;
 for(i,(&a,&b))in a.iter().zip(b).enumerate(){if a!=b{if differences<6{eprintln!("LAYOUT_DIFF pass={pass} step={step} request={request} token={i} canonical={a:04x} hsplit={b:04x}");}differences+=1;}}
 assert_eq!(differences,0,"B16 output cluster split changed final model logits");
}
#[tokio::main]
async fn main()->Result<()>{
 let started=Instant::now();let mut d=Device::new(ops::embed_token.topology())?;
 let weights=load::load_weights(&mut d).await?;
 let mut ws=Workspace::new_canonical64(&mut d,&weights,2048,64).await?;
 let mut model=Model::new(&mut d,weights,&mut ws).await?;model.prepare_group_scratch(&mut d).await?;
 let mut expected:Vec<Vec<Vec<u16>>>=Vec::new();let mut checked=0usize;
 for pass in 0..2{
  model.set_b16_output_hsplit(pass==1);
  let blocks:Vec<_>=(0..64).map(|r|{let slot=if pass==0{r}else{(r*13+31)%64};Blocks{sliding:slot,full:(0..4).map(|p|if pass==0{slot*4+p}else{255-(slot*4+p)}).collect()}}).collect();
  let fill=Instant::now();
  for group in 0..16{for pos in (0..512).step_by(128){
   let ins:Vec<Vec<_>>=(0..4).map(|i|(pos..pos+128).map(|p|input(group*4+i,p,false)).collect()).collect();
   let seq:Vec<_>=(0..4).map(|i|Sequence{inputs:&ins[i],start_position:pos,blocks:&blocks[group*4+i]}).collect();
   model.start_prefill_group(&mut d,&seq,&mut ws).await?;
   while model.advance_prefill_group(&mut d,&mut ws).await?.is_none(){}
  }}
  println!("B16_OUTPUT_FILL pass={pass} actual_history_tokens=32768 ms={:.3}",fill.elapsed().as_secs_f64()*1000.0);
  let mut positions=[512usize;64];
  for step in 0..7{
   let chosen=selected(step);
   if step==5{for &r in &chosen{positions[r]=0;}}
   // Interleave a short prompt after decode to exercise Prefill's retained canonical raw buffers.
   if step==6{
    let request=3;let ins:Vec<_>=(positions[request]..positions[request]+15).map(|p|input(request,p,false)).collect();
    model.forward_sequences(&mut d,&[Sequence{inputs:&ins,start_position:positions[request],blocks:&blocks[request]}],&mut ws).await?;
    positions[request]+=15;
   }
   let ins:Vec<_>=chosen.iter().map(|&r|[input(r,positions[r],true)]).collect();
   let seq:Vec<_>=chosen.iter().enumerate().map(|(i,&r)|Sequence{inputs:&ins[i],start_position:positions[r],blocks:&blocks[r]}).collect();
   model.begin_stage_profile();let start=Instant::now();let out=model.forward_sequences(&mut d,&seq,&mut ws).await?;
   let model_ms=start.elapsed().as_secs_f64()*1000.0;let mut found=Vec::new();
   for(i,x)in out.iter().enumerate(){let values=raw(&mut d,x).await?;if pass==1{check(pass,step,chosen[i],&expected[step][i],&values);checked+=values.len();}found.push(values);}
   if pass==0{expected.push(found);}
   println!("B16_OUTPUT_STEP pass={pass} hsplit={} step={step} live={} model_ms={model_ms:.3} logits_bitdiff=0",pass==1,chosen.len());
   for stage in model.take_stage_profile(){println!("B16_OUTPUT_STAGE pass={pass} step={step} stage={} calls={} total_ms={:.6}",stage.stage,stage.calls,stage.milliseconds);}
   for r in chosen{positions[r]+=1;}
  }
 }
 println!("PASS B16 output cluster split actual model checked={checked} BF16logits;64/17 R64 and16/4 B16;512-token histories,physical pool remap,full-page transition,position0 reuse,and15-token prompt interleave;elapsed_s={:.3};timing diagnostic not HTTP/SLO",started.elapsed().as_secs_f64());Ok(())
}
