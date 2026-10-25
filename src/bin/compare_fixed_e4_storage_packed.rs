//! Fixed E4 numerical surrogate: same-program replay across disjoint owners.
//! OFF/ON comparison uses bounded row summaries, never inferred all-bit parity.
use furiosa_opt_gemma4::{LAYERS,host::{generate::read_logits,load,model::{Input,Model,Sequence},runtime::{Blocks,Workspace}},ops};
use furiosa_opt_std::prelude::*;
use furiosa_opt_gemma4::host::fixed_e4_storage_diagnostic::{CacheSnapshot,verify_identity};
static LOGITS:std::sync::Mutex<Vec<Vec<u16>>>=std::sync::Mutex::new(Vec::new());
type Result<T>=std::result::Result<T,Box<dyn std::error::Error>>;
#[cfg(backend="npu")]
fn snapshot(d:&Device,label:&str)->Result<()> {let m=d.memory()?;println!("FIXED_E4_PACKED_MEMORY label={label} capacity={} available={} largest={}",m.capacity,m.available,m.largest);Ok(())}
#[cfg(not(backend="npu"))]
fn snapshot(_d:&Device,_label:&str)->Result<()> {Ok(())}
#[derive(Clone,Copy,Debug)] enum Route {Group,Single,Pair,Tail256,Tail384,Three}
fn input(owner:usize,p:usize,logits:bool)->Input<'static>{Input{token:2+(owner*733+p*37)%8000,soft:None,logits,image:None,audio:None}}
fn exact(label:&str,a:&[f32],b:&[f32]) {
 furiosa_opt_gemma4::host::fixed_e4_model_diagnostic::exact_f32(label,a,b);
 LOGITS.lock().unwrap().push(a.iter().map(|x|(x.to_bits()>>16)as u16).collect());
}
async fn fill(d:&mut Device,m:&mut Model,w:&mut Workspace,b:&Blocks,owner:usize,end:usize)->Result<()> {
 let mut p=0;
 while p<end {
  let n=if end-p>=512 {512}else{128};let rows:Vec<_>=(p..p+n).map(|p|input(owner,p,false)).collect();
  let seq=Sequence{inputs:&rows,start_position:p,blocks:b};
  if n==512 {m.start_prefill512(d,&seq,w).await?;for layer in 0..LAYERS {assert_eq!(m.advance_prefill(d,w).await?.is_some(),layer+1==LAYERS);}}
  else {m.forward_sequences_native_qkv(d,&[seq],w).await?;}
  p+=n;
 }Ok(())
}
async fn start(d:&mut Device,m:&mut Model,w:&mut Workspace,route:Route,seq:&[Sequence<'_>],padding:bf16)->Result<()> {
 match route {
  Route::Group=>m.start_prefill_group(d,seq,w).await?,
  Route::Single=>m.start_prefill512(d,&seq[0],w).await?,
  Route::Pair=>m.start_prefill_pair512(d,seq,w).await?,
  Route::Tail256=>m.start_prefill_tail256_with_padding(d,seq,w,padding).await?,
  Route::Tail384=>m.start_prefill_tail384_with_padding(d,seq,w,padding).await?,
  Route::Three=>m.start_prefill_three_owner512(d,seq,w).await?,
 }Ok(())
}
async fn run(d:&mut Device,m:&mut Model,w:&mut Workspace,route:Route,blocks:&[Blocks],starts:&[usize],rows:&[Vec<Input<'static>>],candidate:bool,padding:bf16,interleave:bool,label:&str)->Result<(Vec<Vec<f32>>,f64)> {
 assert!(m.set_active_prefill_grouped_norm_stores(candidate).is_err());
 let offset=if candidate{4}else{0};
 let seq:Vec<_>=rows.iter().enumerate().map(|(i,inputs)|Sequence{inputs,start_position:starts[i],blocks:&blocks[offset+i]}).collect();
 let now=std::time::Instant::now();start(d,m,w,route,&seq,padding).await?;
 m.set_active_prefill_ffn_global_preload(true)?; // Both arms use the independently validated preload core.
 m.set_active_prefill_ffn_epilogue_composed(true)?;
 m.set_active_prefill_weight_stationary_output(true)?; // Both sliding and full O stay current in both arms.
 m.set_active_prefill_projection_owned_ffn(true)?;
 m.set_active_prefill_weight_stationary_qkv(true)?;
 m.set_active_prefill_grouped_norm_stores(true)?;
 // Select both arms explicitly, independently of future production defaults.
 let mut ms=now.elapsed().as_secs_f64()*1000.;let mut result=Vec::new();
 assert_eq!(m.active_prefill_rings(),seq.iter().map(|s|s.blocks.sliding).collect::<Vec<_>>());
 let mut decode_positions=[0usize;11];
 m.begin_stage_profile();
 for layer in 0..LAYERS {
  let now=std::time::Instant::now();
  let outputs=match route {
   Route::Single=>m.advance_prefill(d,w).await?.map(std::slice::from_ref),
   Route::Group=>m.advance_prefill_group(d,w).await?,
   Route::Pair=>m.advance_prefill_pair512(d,w).await?,
   Route::Tail256=>m.advance_prefill_tail256(d,w).await?,
   Route::Tail384=>m.advance_prefill_tail384(d,w).await?,
   Route::Three=>m.advance_prefill_three_owner512(d,w).await?,
  };
  ms+=now.elapsed().as_secs_f64()*1000.;
  if let Some(outputs)=outputs {
   assert_eq!(layer+1,LAYERS);assert_eq!(outputs.len(),rows.len());
   for output in outputs {result.push(read_logits(d,output).await?);}
  }else{assert!(layer+1<LAYERS);}
  assert!(m.set_active_prefill_grouped_norm_stores(!candidate).is_err());
  if interleave && [0,1,23,24,46,47].contains(&layer) {
   let count=if layer%2==0 {4}else{11};let mut expected=Vec::new();
   let tokens:Vec<_>=(0..count).map(|i|[input(100+i,decode_positions[i],true)]).collect();
   for base in [8,19] {
    let sequences:Vec<_>=(0..count).map(|i|Sequence{inputs:&tokens[i],start_position:decode_positions[i],blocks:&blocks[base+i]}).collect();
    // Paired B16 interleaves use22 slots; a separate history gate exercises17..32 R32/R64.
    let output=m.forward_sequences(d,&sequences,w).await?;
    for i in 0..count {let got=read_logits(d,&output[i]).await?;
     if base==8{expected.push(got);}else{exact(&format!("{label}_decode{count}_layer{layer}_row{i}"),&expected[i],&got);}
    }
   }
   for p in &mut decode_positions[..count]{*p+=1;}
  }
 }
 for s in m.take_stage_profile(){println!("FIXED_E4_PACKED_STAGE label={label} stage={} calls={} ms={:.6}",s.stage,s.calls,s.milliseconds);}
 assert!(m.active_prefill_rings().is_empty());assert_eq!(result.len(),rows.len());
 println!("FIXED_E4_PACKED_TIME label={label} route={route:?} candidate={candidate} total_ms={ms:.6} excludes_download_interleave_cache=true");
 Ok((result,ms))
}
#[tokio::main]
async fn main()->Result<()> {
 let case=std::env::args().nth(1).unwrap_or_else(||"0".into()).parse::<usize>()?;
 let interleave=std::env::args().nth(2).map_or(case==0,|v|v=="1");
 let cases=[
  (Route::Group,vec![128,128,128,128],vec![0,256,896,1920]),
  (Route::Single,vec![512],vec![512]),
  (Route::Pair,vec![256,256],vec![896,512]),
  (Route::Tail256,vec![256],vec![896]),
  (Route::Tail256,vec![128,128],vec![1920,896]),
  (Route::Tail384,vec![384],vec![1664]),
  (Route::Tail384,vec![128,256],vec![1920,896]),
  (Route::Tail384,vec![128,128,128],vec![0,512,1920]),
  (Route::Three,vec![256,128,128],vec![1664,896,1920]),
  (Route::Three,vec![128,256,128],vec![896,1664,1920]),
  (Route::Three,vec![128,128,256],vec![896,1920,1664]),
 ];
 let (route,counts,starts)=cases.get(case).ok_or("case must be0..10")?.clone();
 let mut d=Device::new(ops::embed_token.topology())?;let mut weights=Some(load::load_weights(&mut d).await?);
 let capacity=2560;let pages=Workspace::blocks_for(capacity);assert_eq!(pages,5);
 let mut reference_rows=None;let mut reference_cache=None;let mut total=0usize;
 for compressed in [false,true] {
 let mut w=if compressed {Workspace::new_canonical_fixed_e4(&mut d,weights.as_ref().unwrap(),capacity,32).await?}
 else {Workspace::new_canonical64(&mut d,weights.as_ref().unwrap(),capacity,32).await?};
 verify_identity(&w,capacity,compressed)?;
 let mut m=Model::new(&mut d,weights.take().unwrap(),&mut w).await?;m.prepare_group_scratch(&mut d).await?;
 snapshot(&d,"before_join")?;
 m.prepare_projection_owned_ffn(&mut d).await?;
 snapshot(&d,"after_join")?;
 //32 distinct real slots:8 prompt+22 decoder+2sentinel. No shared ring or page.
 let original:Vec<_>=(0..32).map(|i|Blocks{sliding:(i*13+7)%32,full:(0..pages).map(|p|255-i*pages-p).collect()}).collect();
 for slot in [30,31]{fill(&mut d,&mut m,&mut w,&original[slot],90,128).await?;}
 for round in 0..2 {
  let mut blocks=original.clone();let mut starts=starts.clone();
  if round==1 {starts.fill(0);blocks[..counts.len()].reverse();blocks[4..4+counts.len()].reverse();}
  for owner in 0..counts.len(){for slot in [owner,owner+4]{fill(&mut d,&mut m,&mut w,&blocks[slot],owner,starts[owner]).await?;}}
  let rows:Vec<Vec<_>>=counts.iter().enumerate().map(|(owner,&n)|(starts[owner]..starts[owner]+n).map(|p|input(owner,p,p+1==starts[owner]+n)).collect()).collect();
  let padding=bf16::from_bits(if round==0{0}else{0x7fc1});let mut outputs:[Vec<Vec<f32>>;2]=[Vec::new(),Vec::new()];let mut ms=[0f64;2];
  // Reverse candidate/control order on reset. Both compute the identical route.
  for candidate in if round==0{[false,true]}else{[true,false]} {
   let label=format!("case{case}_round{round}_candidate{candidate}");
   let (out,time)=run(&mut d,&mut m,&mut w,route,&blocks,&starts,&rows,candidate,padding,interleave&&candidate&&round==0,&label).await?;
   outputs[candidate as usize]=out;ms[candidate as usize]=time;
  }
  for owner in 0..counts.len(){
   exact(&format!("case{case}_round{round}_owner{owner}"),&outputs[0][owner],&outputs[1][owner]);
   let checked=m.compare_canonical_kv(&mut d,&w,&blocks[owner],&blocks[owner+4]).await?;
   println!("FIXED_E4_PACKED_CACHE case={case} round={round} owner={owner} bf16_values={checked} bitdiff=0 all_layers=true");
  }
  let checked=m.compare_canonical_kv(&mut d,&w,&blocks[30],&blocks[31]).await?;
  println!("FIXED_E4_PACKED_SENTINEL case={case} round={round} bf16_values={checked} bitdiff=0");
  for step in 0..3 {
   let tokens:Vec<_>=(0..counts.len()).map(|i|[input(i,starts[i]+counts[i]+step,true)]).collect();let mut expected=Vec::new();
   for base in [0,4] {let seq:Vec<_>=(0..counts.len()).map(|i|Sequence{inputs:&tokens[i],start_position:starts[i]+counts[i]+step,blocks:&blocks[base+i]}).collect();
    let output=m.forward_sequences(&mut d,&seq,&mut w).await?;
    for i in 0..counts.len(){let got=read_logits(&mut d,&output[i]).await?;if base==0{expected.push(got);}else{exact(&format!("case{case}_round{round}_post{step}_owner{i}"),&expected[i],&got);}}
   }
  }
  println!("FIXED_E4_PACKED_PASS case={case} round={round} control_ms={:.6} candidate_ms={:.6}",ms[0],ms[1]);
 }
 let rows=std::mem::take(&mut *LOGITS.lock().unwrap());
 if compressed {
  let expected:Vec<Vec<u16>>=reference_rows.take().unwrap();assert_eq!(expected.len(),rows.len());
  for (i,(a,b)) in expected.iter().zip(&rows).enumerate(){let first=a.iter().zip(b).position(|(a,b)|a!=b);assert!(first.is_none(),"cross-storage packed full vocabulary row{i} first={first:?}");total+=a.len();}
  let cache:CacheSnapshot=reference_cache.take().unwrap();cache.compare_e4(&mut d,&w).await?;
 } else {reference_rows=Some(rows);reference_cache=Some(CacheSnapshot::capture_bf16(&mut d,&w).await?);}
 weights=Some(m.into_weights_for_storage_diagnostic()?);drop(w);snapshot(&d,"dropped_pass_pools_and_scratch")?;
 }
 println!("PASS FIXED_E4_STORAGE_PACKED case={case} cross_storage_all_logits={total} full_cache_bf16_bytes=12884901888 serving_default=false");Ok(())
}
