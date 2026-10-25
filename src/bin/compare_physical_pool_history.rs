//! Fixed logical32-owner schedule on separately frozen physical64/32 programs.
//! In-process equal-history pairs compare all bits; SHA256 permits cross-program comparison.
use furiosa_opt_gemma4::{LAYERS,axes::*,ops,ops_pool_views::Rc64,
 host::{load,generate::read_logits,model::{Input,Model,Sequence},runtime::{Blocks,Workspace}}};
use furiosa_opt_std::prelude::*;
type Result<T>=std::result::Result<T,Box<dyn std::error::Error>>;
fn input(owner:usize,p:usize,logits:bool)->Input<'static>{Input{token:2+(owner*313+p*37)%8000,soft:None,logits,image:None,audio:None}}
fn digest(v:&[f32])->String {let b:Vec<_>=v.iter().flat_map(|x|x.to_bits().to_le_bytes()).collect();ring::digest::digest(&ring::digest::SHA256,&b).as_ref().iter().map(|x|format!("{x:02x}")).collect()}
#[cfg(backend="npu")]fn memory(d:&Device,label:&str)->Result<()> {let m=d.memory()?;println!("PHYSICAL_HISTORY_MEMORY label={label} physical={} capacity={} available={} largest={}",Rc64::SIZE,m.capacity,m.available,m.largest);Ok(())}
#[cfg(not(backend="npu"))]fn memory(_: &Device,_:&str)->Result<()>{Ok(())}
async fn fill(d:&mut Device,m:&mut Model,w:&mut Workspace,b:&Blocks,owner:usize,end:usize)->Result<()> {
 let mut p=0;while p<end {let n=if end-p>=512{512}else{128};assert!(p+n<=end);
 let rows:Vec<_>=(p..p+n).map(|p|input(owner,p,false)).collect();let seq=Sequence{inputs:&rows,start_position:p,blocks:b};
 if n==512 {m.start_prefill512(d,&seq,w).await?;for l in 0..LAYERS{assert_eq!(m.advance_prefill(d,w).await?.is_some(),l+1==LAYERS);}}
 else {m.forward_sequences_native_qkv(d,&[seq],w).await?;}p+=n;}Ok(())
}
async fn step(d:&mut Device,m:&mut Model,w:&mut Workspace,blocks:&[Blocks],positions:&mut[usize],chosen:&[usize],half:usize,label:&str)->Result<usize>{
 let ins:Vec<_>=chosen.iter().map(|&i|[input(i%half,positions[i],true)]).collect();
 let seq:Vec<_>=chosen.iter().enumerate().map(|(j,&i)|Sequence{inputs:&ins[j],start_position:positions[i],blocks:&blocks[i]}).collect();
 let begin=std::time::Instant::now();let out=m.forward_sequences(d,&seq,w).await?;let ms=begin.elapsed().as_secs_f64()*1000.;
 let mut rows=Vec::new();for (j,&i)in chosen.iter().enumerate(){let v=read_logits(d,&out[j]).await?;assert!(v.iter().all(|x|x.is_finite()));println!("PHYSICAL_HISTORY_LOGITS label={label} owner={i} position={} values={} sha256={}",positions[i],v.len(),digest(&v));rows.push(v);}
 let mut checked=0;for(i,&a)in chosen.iter().enumerate(){if a<half {if let Some(j)=chosen.iter().position(|&b|b==a+half){let dif=rows[i].iter().zip(&rows[j]).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();assert_eq!(dif,0,"paired history output changed");checked+=W::SIZE;}}}
 for &i in chosen{positions[i]+=1;}println!("PHYSICAL_HISTORY_STEP label={label} width={} model_ms={ms:.6} paired_values={checked}",chosen.len());Ok(checked)
}
#[tokio::main]async fn main()->Result<()> {
 let prefix=std::env::args().nth(1).unwrap_or("512".into()).parse::<usize>()?;
 let width=std::env::args().nth(2).unwrap_or("32".into()).parse::<usize>()?;
 assert!([512,1024,8064].contains(&prefix));assert!([2,17,31,32].contains(&width));assert!(prefix!=8064||width==2);
 let owners=if width==2{2}else{32};let half=owners/2;let pairs=width/2;
 let mut chosen:Vec<_>=(0..pairs).chain(half..half+pairs).collect();if width%2==1{chosen.push(half-1);}assert_eq!(chosen.len(),width);
 let mut d=Device::new(ops::embed_token.topology())?;let weights=load::load_weights(&mut d).await?;
 let mut w=Workspace::new_canonical64(&mut d,&weights,8192,32).await?;let mut m=Model::new(&mut d,weights,&mut w).await?;
 m.prepare_group_scratch(&mut d).await?;m.prepare_projection_owned_ffn(&mut d).await?;memory(&d,"prepared")?;
 let pages=Workspace::blocks_for(prefix+if width==17{136}else{8});assert!(owners*pages<=256);let mut checked=0;
 let rounds=if width==17||width==31{1}else{2};
 for round in 0..rounds {
  // Both odd permutations preserve the paired16-slot difference in32-owner mode.
  let blocks:Vec<_>=(0..owners).map(|i|Blocks{sliding:if round==0{(i*13+31)%32}else{(i*7+11)%32},
   full:(0..pages).map(|p|255-i*pages-p).collect()}).collect();
  let begin=std::time::Instant::now();for i in 0..owners{fill(&mut d,&mut m,&mut w,&blocks[i],i%half,prefix).await?;}
  println!("PHYSICAL_HISTORY_FILL round={round} owners={owners} real_tokens={} ms={:.3}",owners*prefix,begin.elapsed().as_secs_f64()*1000.);
  let mut pos=vec![prefix;owners];
  if width==17&&round==0 {
   let a=half-1;let b=owners-1;assert!(!chosen.contains(&b));
   // The odd unpaired decode owner must not collide with this active prompt.
   let mut decode=chosen.clone();*decode.last_mut().unwrap()=half-2;assert!(!decode.contains(&a));
   let rows:Vec<_>=(prefix..prefix+128).map(|p|input(a,p,p==prefix+127)).collect();
   m.start_prefill(&mut d,&Sequence{inputs:&rows,start_position:prefix,blocks:&blocks[a]},&mut w).await?;
   let invalid=[input(a,prefix,true)];assert!(m.forward_sequences(&mut d,&[Sequence{inputs:&invalid,start_position:prefix,blocks:&blocks[a]}],&mut w).await.is_err());
   let mut prompt=Vec::new();for l in 0..LAYERS{
    if let Some(out)=m.advance_prefill(&mut d,&mut w).await?{assert_eq!(l+1,LAYERS);prompt=read_logits(&mut d,out).await?;}
    if [0,1,23,24,46,47].contains(&l){checked+=step(&mut d,&mut m,&mut w,&blocks,&mut pos,&decode,half,&format!("r{round}_layer{l}")).await?;}
   }
   m.start_prefill(&mut d,&Sequence{inputs:&rows,start_position:prefix,blocks:&blocks[b]},&mut w).await?;
   for l in 0..LAYERS{if let Some(out)=m.advance_prefill(&mut d,&mut w).await?{assert_eq!(l+1,LAYERS);let v=read_logits(&mut d,out).await?;assert_eq!(prompt.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),v.iter().map(|v|v.to_bits()).collect::<Vec<_>>());println!("PHYSICAL_HISTORY_PROMPT round={round} values={} sha256={}",v.len(),digest(&v));checked+=v.len();}}
   m.compare_canonical_kv(&mut d,&w,&blocks[a],&blocks[b]).await?;
  }else{
   for s in 0..3{if s==2{pos[0]=0;pos[half]=0;}checked+=step(&mut d,&mut m,&mut w,&blocks,&mut pos,&chosen,half,&format!("r{round}_step{s}")).await?;}
  }
  for i in 0..pairs{m.compare_canonical_kv(&mut d,&w,&blocks[i],&blocks[i+half]).await?;}
 }
 let invalid=[input(0,0,true)];let bad=Blocks{sliding:32,full:vec![255]};assert!(m.forward_sequences(&mut d,&[Sequence{inputs:&invalid,start_position:0,blocks:&bad}],&mut w).await.is_err());
 assert!(m.compare_canonical_kv(&mut d,&w,&bad,&bad).await.is_err());
 memory(&d,"warm")?;println!("PASS physical_history physical={} prefix={prefix} width={width} paired_logits={checked} full_reserved_cache_exact rounds={rounds}; cross_program_equality_requires_separate_digest_match",Rc64::SIZE);Ok(())
}
