//! Independent total_cmp oracle for generic packed-ban device greedy selection.
use furiosa_opt_gemma4::{Chip, axes::{B,W}, ops_batch_io,
    ops_greedy_b16::{self as greedy,BanWords,GreedyResult},
    host::{generate::read_logits,sampling::{sample,SamplingConfig}}};
use std::{collections::HashSet,time::Instant};
type Logits=HbmTensor<bf16,Chip,m![B,W]>;
type Row=HbmTensor<bf16,Chip,m![W]>;
use furiosa_opt_std::prelude::*;
fn hash(mut x:u64)->u64{x=(x^(x>>30)).wrapping_mul(0xbf58476d1ce4e5b9);x=(x^(x>>27)).wrapping_mul(0x94d049bb133111eb);x^(x>>31)}
fn bits(row:usize,n:usize)->u16 {
 let h=hash((row*W::SIZE+n)as u64+73);let random=(h as u16&0x8000)|(((h>>16)as u16)%0x7f80);
 match row {
  0=>{let b=n as u16;if b&0x7fff>=0x7f80 {0}else{b}},
  1=>0,2=>0x8000,
  3=>if n==W::SIZE-1{0}else{0x8000},
  4=>if n==W::SIZE-1{1}else{0x8000},
  5=>if n==8192{0x8000}else{0x8001},
  6|7=>random,
  8=>if n==31||n==8192{0x7f7f}else{0x3f80},
  9=>if n==0||n==32{0x4000}else{0xbf80},
  16=>if n==8192{0x7f80}else{random},
  17=>if n==31{0xff80}else{random},
  18=>if n==W::SIZE-1{0x7fc1}else{random},
  19=>if n==32{0x7f81}else{random},
  20=>if n==8191{0xffc7}else{random},
  21=>if n==0{0xff81}else{random},
  22=>0x7fc1,23=>0xff81,
  24=>match n%4{0=>0x7f80,1=>0xff80,2=>0x7f81,_=>0xff81},
  25=>if n==8191{0x007f}else{0x0001},
  26=>if n==8192{0x807e}else{0x807f},
  27=>if n==W::SIZE-1{0xff7e}else{0xff7f},
  28=>if n==0{0x3f80}else if n==8192{0x007f}else{0x0001},
  29=>if n==W::SIZE-1{0x007f}else{0x0001},
  30..=63=>{let special=[0,31,32,8191,8192,W::SIZE-1][row%6];if n==special||n==(special+16384)%W::SIZE{0x7f7f}else{random&0xff7e}},
  _=>random,
 }
}
fn banned(row:usize,n:usize)->bool{match row {
 0..=5|16..=27=>false,6=>true,7=>n!=12345,8=>n==31,9=>n==0,
 10=>n%2==0,11=>n%32!=31,12=>n%8192!=8191,13=>n!=W::SIZE-1,
 14=>n!=0,15|29=>true,28=>n==0,_=>hash((row*W::SIZE+n)as u64+9163)%7==0,
}}
fn key(bits:u16)->i32{let raw=bits as i16 as i32;raw^((raw>>15)&0x7fff)}
fn oracle(row:usize)->(usize,usize,i32,u16){
 let mut all=None;let mut allowed=None;let mut abs=0;
 for n in 0..W::SIZE {
  let b=bits(row,n);let v=bf16::from_bits(b).to_f32();abs=abs.max(b&0x7fff);
  if all.is_none_or(|(old,_):(f32,usize)|v.total_cmp(&old).is_gt()){all=Some((v,n));}
  if !banned(row,n)&&allowed.is_none_or(|(old,_):(f32,usize)|v.total_cmp(&old).is_gt()){allowed=Some((v,n));}
 }
 let unmasked=all.unwrap().1;let selected=allowed.map_or(unmasked,|x|x.1);
 let maximum=allowed.map_or(-65536,|(_,n)|key(bits(row,n)));
 (selected,unmasked,maximum,abs)
}

fn chosen(result:&[i32])->Option<usize> {
 let key=if result[0]==-65536 {result[4]} else {result[0]};
 if result[3]>=0x7f80||(-1..=0).contains(&key) {None}
 else {Some(if result[0]==-65536 {-result[2]} else {-result[1]} as usize)}
}
async fn upload_rows(d:&mut Device,base:usize)->Result<(Logits,Vec<i32>),Box<dyn std::error::Error>> {
 let mut raw=vec![0u8;16*W::SIZE*2];let mut bans=vec![0i32;16*W::SIZE/32];
 for row in 0..16 {for n in 0..W::SIZE {
  let case=base+row;let i=row*W::SIZE+n;
  raw[i*2..i*2+2].copy_from_slice(&bits(case,n).to_le_bytes());
  if banned(case,n) {bans[(n/32)*16+row]|=(1u32<<(n%32))as i32;}
 }}
 Ok((HostTensor::<bf16,m![B,W]>::from_buf(raw).to_hbm(&mut d.pdma).await?,bans))
}
async fn check_group(d:&mut Device,base:usize,expected:&[(usize,usize,i32,u16)],
 ids:&HbmTensor<i32,Chip,m![W]>,shifts:&HbmTensor<i32,Chip,m![W%32]>) ->Result<(),Box<dyn std::error::Error>> {
 let (logits,bans)=upload_rows(d,base).await?;
 let words=HostTensor::<i32,BanWords>::from_vec(bans).to_hbm(&mut d.pdma).await?;
 let mut out=HostTensor::<i32,GreedyResult>::from_vec(std::iter::repeat_n(-123456,16*8)).to_hbm(&mut d.pdma).await?;
 launch(greedy::greedy_argmax_b16,(d,&logits,&words,ids,shifts,&mut out)).await?;
 let got=out.to_host::<GreedyResult>(&mut d.pdma).await?.into_vec();assert_eq!(got.len(),16*8);
 let mut fallback=0;let mut row_logits:Row=HostTensor::<bf16,m![W]>::zero().to_hbm(&mut d.pdma).await?;
 let mut rng=rand::rng();
 for row in 0..16 {
  let case=base+row;let result=&got[row*8..row*8+8];let (selected,unmasked,maximum,abs)=expected[case];
  let actual=chosen(result);let expected_fallback=abs>=0x7f80||(bits(case,selected)&0x7fff)<0x80;
  assert_eq!(actual.is_none(),expected_fallback,"fallback case={case} fields={result:?}");
  assert_eq!(result[3]>=0x7f80,abs>=0x7f80,"nonfinite case={case}");
  if let Some(actual)=actual {
   assert_eq!(actual,selected,"case={case}");assert_eq!(result[0],maximum,"maximum case={case}");
   assert_eq!(-result[2],unmasked as i32,"unmasked case={case}");assert_eq!(result[3],abs as i32,"abs case={case}");
  } else {
   fallback+=1;
   let offset=HostTensor::<i32,m![1]>::from_vec([(row*W::SIZE*2)as i32]).to_hbm(&mut d.pdma).await?;
   launch(ops_batch_io::extract_logits_row,(d,&logits,&offset,&mut row_logits)).await?;
   let banned:HashSet<_>=(0..W::SIZE).filter(|&n|banned(case,n)).collect();
   let actual=sample(&read_logits(d,&row_logits).await?,&SamplingConfig::DEFAULT,&banned,&mut rng);
   assert_eq!(actual,selected,"real CPU fallback case={case}");
  }
 }
 println!("GREEDY_ORACLE_PASS cases={}..{} finite={} fallback={} live_rows=16 canonical_B_W=true",base,base+16,16-fallback,fallback);
 Ok(())
}
async fn timing(d:&mut Device,live:usize,ids:&HbmTensor<i32,Chip,m![W]>,shifts:&HbmTensor<i32,Chip,m![W%32]>) ->Result<(),Box<dyn std::error::Error>> {
 let raw:Vec<u8>=(0..16).flat_map(|r|(0..W::SIZE).flat_map(move|n|bits(30+r,n).to_le_bytes())).collect();
 let logits=HostTensor::<bf16,m![B,W]>::from_buf(raw).to_hbm(&mut d.pdma).await?;
 let banned:HashSet<usize>=[258882,258883].into_iter().collect();let mut rng=rand::rng();
 let mut out=HostTensor::<i32,GreedyResult>::zero().to_hbm(&mut d.pdma).await?;
 let mut rows=Vec::new();
 for row in 0..live {
  let offset=HostTensor::<i32,m![1]>::from_vec([(row*W::SIZE*2)as i32]).to_hbm(&mut d.pdma).await?;
  let mut result:Row=HostTensor::<bf16,m![W]>::zero().to_hbm(&mut d.pdma).await?;
  launch(ops_batch_io::extract_logits_row,(d,&logits,&offset,&mut result)).await?;rows.push(result);
 }
 // Baseline inputs are already extracted, matching the existing post-model CPU
 // selection boundary. Any removed per-row extraction is an additional saving.
 let mut expected=Vec::new();for row in &rows {expected.push(sample(&read_logits(d,row).await?,&SamplingConfig::DEFAULT,&banned,&mut rng));}
 for candidate in [false,true,true,false,false,true,true,false] {
  if candidate {
   let begin=Instant::now();let mut words=vec![0i32;W::SIZE/32*16];
   for row in 0..live {for &token in &banned {words[(token/32)*16+row]|=(1u32<<(token%32))as i32;}}
   let words=HostTensor::<i32,BanWords>::from_vec(words).to_hbm(&mut d.pdma).await?;
   let upload=begin.elapsed().as_secs_f64()*1e6;let part=Instant::now();
   launch(greedy::greedy_argmax_b16,(d,&logits,&words,ids,shifts,&mut out)).await?;
   let kernel=part.elapsed().as_secs_f64()*1e6;let part=Instant::now();
   let got=out.to_host::<GreedyResult>(&mut d.pdma).await?.into_vec();
   let actual:Vec<_>=(0..live).map(|row|chosen(&got[row*8..row*8+8]).expect("ordinary finite timing row")).collect();
   let read=part.elapsed().as_secs_f64()*1e6;let total=begin.elapsed().as_secs_f64()*1e6;
   assert_eq!(actual,expected);
   println!("GREEDY_INCLUSIVE live={live} ban_pack_upload_us={upload:.3} kernel_us={kernel:.3} read_parse512B_us={read:.3} total_us={total:.3}");
  } else {
   let begin=Instant::now();let mut read=0.0;let mut cpu=0.0;let mut actual=Vec::new();
   for row in &rows {
    let part=Instant::now();let values=read_logits(d,row).await?;read+=part.elapsed().as_secs_f64()*1e6;
    let part=Instant::now();actual.push(sample(&values,&SamplingConfig::DEFAULT,&banned,&mut rng));cpu+=part.elapsed().as_secs_f64()*1e6;
   }
   let total=begin.elapsed().as_secs_f64()*1e6;assert_eq!(actual,expected);
   println!("CPU_INCLUSIVE live={live} row_read_us={read:.3} sample_us={cpu:.3} total_us={total:.3}");
  }
 }
 Ok(())
}
#[tokio::main(flavor="current_thread")]
async fn main()->Result<(),Box<dyn std::error::Error>> {
 let mut finite:Vec<_>=(0..=u16::MAX).filter(|b|b&0x7fff<0x7f80).collect();
 finite.sort_by(|&a,&b|bf16::from_bits(a).to_f32().total_cmp(&bf16::from_bits(b).to_f32()));
 assert!(finite.windows(2).all(|w|key(w[0])<key(w[1])));
 let expected:Vec<_>=(0..64).map(oracle).collect();
 println!("FINITE_KEY_CPU_PASS patterns={} special_cases=64",finite.len());
 if std::env::args().any(|x|x=="--cpu-only") {return Ok(());}
 let mut d=Device::new(greedy::greedy_argmax_b16.topology())?;
 let ids=HostTensor::<i32,m![W]>::from_vec((0..W::SIZE).map(|n|-(n as i32))).to_hbm(&mut d.pdma).await?;
 let shifts=HostTensor::<i32,m![W%32]>::from_vec(0..32).to_hbm(&mut d.pdma).await?;
 for group in 0..4 {check_group(&mut d,group*16,&expected,&ids,&shifts).await?;}
 timing(&mut d,8,&ids,&shifts).await?;timing(&mut d,16,&ids,&shifts).await?;
 println!("GREEDY_B16_PASS all64cases_in4groups arbitrary_bans_allbanned_ties_signedzeros_subnormals_nonfinite_and_actualCPUfallback");Ok(())
}
