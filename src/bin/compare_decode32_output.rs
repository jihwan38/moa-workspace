//! Exact first32 O versus accepted hybrid32 control; all other arithmetic/routes match.
//! Stage-profile companion to the frozen exactness probe; model/selector intervals exclude printing and logit reads.
//! Args32 (uniform512, all32 owners) or17 (mixed boundaries, active prefill interleave).
use furiosa_opt_gemma4::{LAYERS,axes::*,ops,ops_r64_io_blocked::BlockedLogits,
 host::{load,model::{Input,Model,Sequence,ForwardOutput,StageTiming},runtime::{Blocks,Workspace},
 greedy::GreedySelector,kv_diagnostic}};
use furiosa_opt_std::prelude::*;
use std::{collections::HashSet,time::Instant};
type Result<T>=std::result::Result<T,Box<dyn std::error::Error>>;
fn input(r:usize,p:usize,logits:bool)->Input<'static>{Input{token:2+(r*313+p*37)%8000,soft:None,image:None,audio:None,logits}}
async fn fill(d:&mut Device,m:&mut Model,w:&mut Workspace,blocks:&[Blocks],n:usize,positions:&[usize])->Result<()> {
 for r in 0..n {let mut pos=0;while pos<positions[r] {
  let left=positions[r]-pos;let count=if left>=512&&pos%512==0 {512}else if left>=128&&pos%128==0 {128}else{left.min(16).min(512-pos%512)};
  let inputs:Vec<_>=(pos..pos+count).map(|p|input(r,p,false)).collect();
  for slot in [32+r,r] {let s=Sequence{inputs:&inputs,start_position:pos,blocks:&blocks[slot]};
   if count==512 {m.start_prefill512(d,&s,w).await?;for layer in 0..LAYERS {assert_eq!(m.advance_prefill(d,w).await?.is_some(),layer+1==LAYERS);}}
   else {m.forward_sequences(d,&[s],w).await?;}
  }pos+=count;
 }}Ok(())
}
async fn run(d:&mut Device,m:&mut Model,w:&mut Workspace,selector:&mut GreedySelector,sequences:&[Sequence<'_>],candidate:bool)->Result<(Vec<u8>,Vec<Option<usize>>,f64,f64,Vec<StageTiming>)> {
 m.begin_stage_profile();let t=Instant::now();let out=if candidate {m.forward_outputs_capacity32_output(d,sequences,w).await?}else{m.forward_outputs_capacity32(d,sequences,w).await?};let model_ms=t.elapsed().as_secs_f64()*1e3;
 let ForwardOutput::Blocked64(out,live)=out else {return Err("native R64 logits expected".into())};assert_eq!(live,sequences.len());
 let banned:HashSet<usize>=[258882,258883,7,11].into_iter().collect();let bans:Vec<_>=(0..live).map(|_|Some(&banned)).collect();
 let t=Instant::now();let tokens=selector.select(d,out,&bans).await?;let selector_ms=t.elapsed().as_secs_f64()*1e3;
 let raw=out.to_host::<BlockedLogits>(&mut d.pdma).await?.into_inner().into_buf();
 let stages=m.take_stage_profile();
 Ok((raw,tokens,model_ms,selector_ms,stages))
}
fn print_stages(stages:&[StageTiming],phase:&str,step:usize,live:usize,model_ms:f64) {
 let mut recorded=0.0;
 for stage in stages {
  recorded+=stage.milliseconds;
  println!("OUTPUT32_STAGE phase={phase} step={step} live={live} name={} calls={} ms={:.6}",stage.stage,stage.calls,stage.milliseconds);
 }
 println!("OUTPUT32_STAGE_TOTAL phase={phase} step={step} live={live} recorded_ms={recorded:.6} model_ms={model_ms:.6} unrecorded_ms={:.6}",model_ms-recorded);
}
fn check(a:&[u8],b:&[u8],live:usize,step:usize){let mut differences=0;for band in 0..W::SIZE/32 {for row in 0..live {for col in 0..32 {
 let i=((band*64+row)*32+col)*2;let aa=u16::from_le_bytes([a[i],a[i+1]]);let bb=u16::from_le_bytes([b[i],b[i+1]]);assert_ne!(aa&0x7f80,0x7f80,"nonfinite step{step} row{row}");
 if aa!=bb {if differences<4 {println!("OUTPUT32_DIFF step={step} row={row} token={} actual={aa:04x} expected={bb:04x}",band*32+col);}differences+=1;}
 }}}
 println!("OUTPUT32_LOGITS step={step} live={live} values={} differences={differences}",live*W::SIZE);assert_eq!(differences,0);
}
#[tokio::main]
async fn main()->Result<()> {
 let n:usize=std::env::args().nth(1).unwrap_or_else(||"32".into()).parse()?;assert!([17,32].contains(&n));
 let begin=Instant::now();let mut d=Device::new(ops::embed_token.topology())?;let weights=load::load_weights(&mut d).await?;let mut ws=Workspace::new_canonical64(&mut d,&weights,2048,64).await?;let mut model=Model::new(&mut d,weights,&mut ws).await?;let mut selector=GreedySelector::new(&mut d).await?;
 let blocks:Vec<_>=(0..64).map(|s|{let ring=(s*13+31)%64;Blocks{sliding:ring,full:(0..4).map(|p|255-ring*4-p).collect()}}).collect();
 let mut positions=vec![512;n];if n==17 {positions[..8].copy_from_slice(&[0,511,512,767,768,1023,1024,1151]);}
 fill(&mut d,&mut model,&mut ws,&blocks,n,&positions).await?;
 println!("OUTPUT32_HISTORY owners={n} computed_tokens={} startup_and_fill_ms={:.3}",positions.iter().sum::<usize>()*2,begin.elapsed().as_secs_f64()*1e3);
 let prompt:Vec<_>=(0..128).map(|p|input(90,p,false)).collect();
 if n==17 {model.start_prefill(&mut d,&Sequence{inputs:&prompt,start_position:0,blocks:&blocks[31]},&mut ws).await?;}
 let widths=if n==32 {[32,17,24,32,7,16,32,32]}else{[17,17,17,17,7,16,17,17]};let mut checked=0usize;
 for (step,live) in widths.into_iter().enumerate() {
  if step==7 {positions[0]=0;}
  let chosen:Vec<_>=(0..live).map(|i|(i*5+step*3)%n).collect();
  let inputs:Vec<_>=chosen.iter().map(|&r|[input(r,positions[r],true)]).collect();
  let cs:Vec<_>=chosen.iter().enumerate().map(|(i,&r)|Sequence{inputs:&inputs[i],start_position:positions[r],blocks:&blocks[32+r]}).collect();
  let rs:Vec<_>=chosen.iter().enumerate().map(|(i,&r)|Sequence{inputs:&inputs[i],start_position:positions[r],blocks:&blocks[r]}).collect();
  // Alternate the completed full-step ordering; each side has its own true cache.
  let (a,b)=if step%2==0 {(run(&mut d,&mut model,&mut ws,&mut selector,&cs,true).await?,run(&mut d,&mut model,&mut ws,&mut selector,&rs,false).await?)}else{let b=run(&mut d,&mut model,&mut ws,&mut selector,&rs,false).await?;(run(&mut d,&mut model,&mut ws,&mut selector,&cs,true).await?,b)};
  check(&a.0,&b.0,live,step);assert_eq!(a.1,b.1,"live greedy outputs");checked+=live*W::SIZE;
  println!("OUTPUT32_STEP step={step} live={live} enabled={} control_model_ms={:.3} candidate_model_ms={:.3} control_select_ms={:.3} candidate_select_ms={:.3}",(17..=32).contains(&live),b.2,a.2,b.3,a.3);
  // Print only after both complete model/selection intervals and their exact comparisons.
  print_stages(&b.4,"controlHybrid32",step,live,b.2);
  print_stages(&a.4,"candidateO32",step,live,a.2);
  for r in chosen {positions[r]+=1;}
  if n==17 {for j in 0..6 {assert_eq!(model.advance_prefill(&mut d,&mut ws).await?.is_some(),step==7&&j==5);}}
 }
 let t=Instant::now();let mut bytes=0usize;for r in 0..n {bytes+=kv_diagnostic::compare_canonical_kv(&mut d,&ws,&blocks[32+r],&blocks[r]).await?;}
 println!("OUTPUT32_CACHE owners={n} all_reserved_bytes={bytes} ms={:.3}",t.elapsed().as_secs_f64()*1e3);
 // Outside-capacity33/64: both APIs must retain the same complete64 projection.
 // Each arm resets its own request at0; no result/history is copied or reused.
 for live in [33,64] {
  let inputs:Vec<_>=(0..live).map(|r|[input(r+300,0,true)]).collect();
  let sequences:Vec<_>=(0..live).map(|r|Sequence{inputs:&inputs[r],start_position:0,blocks:&blocks[r]}).collect();
  let a=run(&mut d,&mut model,&mut ws,&mut selector,&sequences,true).await?;
  let b=run(&mut d,&mut model,&mut ws,&mut selector,&sequences,false).await?;
  check(&a.0,&b.0,live,live);assert_eq!(a.1,b.1);checked+=live*W::SIZE;
  println!("OUTPUT32_GUARD live={live} enabled=false candidate_ms={:.3} control_ms={:.3} all_logits_greedy_exact=true",a.2,b.2);
 }
 println!("PASS same-schedule O32 versus hybrid32 logits={checked}, greedy, allcachebytes, regroup/reset/interleave={} total_s={:.3}",n==17,begin.elapsed().as_secs_f64());Ok(())
}
