//! Independent total_cmp oracle for generic packed-ban device greedy selection.
use furiosa_opt_gemma4::{axes::{R64,W},ops_r64_io_blocked::BlockedLogits,ops_greedy_r64::{self as greedy,BanWords,GreedyResult}};
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
  28..=63=>{let special=[0,31,32,8191,8192,W::SIZE-1][row%6];if n==special||n==(special+16384)%W::SIZE{0x7f7f}else{random&0xff7e}},
  _=>random,
 }
}
fn banned(row:usize,n:usize)->bool{match row {
 0..=5|16..=27=>false,6=>true,7=>n!=12345,8=>n==31,9=>n==0,
 10=>n%2==0,11=>n%32!=31,12=>n%8192!=8191,13=>n!=W::SIZE-1,
 14=>n!=0,15=>true,_=>hash((row*W::SIZE+n)as u64+9163)%7==0,
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
#[tokio::main(flavor="current_thread")]
async fn main()->Result<(),Box<dyn std::error::Error>> {
 // Independently check the finite ordering key including subnormals and signed zeros.
 let mut finite:Vec<_>=(0..=u16::MAX).filter(|b|b&0x7fff<0x7f80).collect();
 finite.sort_by(|&a,&b|bf16::from_bits(a).to_f32().total_cmp(&bf16::from_bits(b).to_f32()));
 assert!(finite.windows(2).all(|w|key(w[0])<key(w[1])));
 println!("FINITE_KEY_CPU_PASS patterns={} signed_zero_keys={:?}",finite.len(),[key(0x8000),key(0)]);
 let expected:Vec<_>=(0..R64::SIZE).map(oracle).collect();
 if std::env::args().any(|x|x=="--cpu-only"){println!("GREEDY_CPU_CASES_PASS rows=64");return Ok(());}
 let mut device=Device::new(greedy::greedy_argmax_r64.topology())?;
 let mut raw=vec![0u8;R64::SIZE*W::SIZE*2];let mut bans=vec![0i32;R64::SIZE*W::SIZE/32];
 for row in 0..R64::SIZE{for n in 0..W::SIZE{
  let i=((n/32)*64+row)*32+n%32;raw[i*2..i*2+2].copy_from_slice(&bits(row,n).to_le_bytes());
  if banned(row,n){bans[(n/32)*64+row]|=(1u32<<(n%32))as i32;}
 }}
 let logits=HostTensor::<bf16,BlockedLogits>::from_buf(raw).to_hbm(&mut device.pdma).await?;
 let start=std::time::Instant::now();let ban_words=HostTensor::<i32,BanWords>::from_vec(bans).to_hbm(&mut device.pdma).await?;
 println!("GREEDY_BAN_INITIAL_UPLOAD_MS {:.6}",start.elapsed().as_secs_f64()*1000.);
 let ids=HostTensor::<i32,m![W]>::from_vec((0..W::SIZE).map(|n|-(n as i32))).to_hbm(&mut device.pdma).await?;
 let shifts=HostTensor::<i32,m![W%32]>::from_vec(0..32).to_hbm(&mut device.pdma).await?;
 let mut out=HostTensor::<i32,GreedyResult>::from_vec(std::iter::repeat_n(-123456,R64::SIZE*8)).to_hbm(&mut device.pdma).await?;
 launch(greedy::greedy_argmax_r64,(&mut device,&logits,&ban_words,&ids,&shifts,&mut out)).await?;
 let got=out.to_host::<GreedyResult>(&mut device.pdma).await?.into_vec();
 let mut fallback=0;
 for row in 0..R64::SIZE {
  let result=&got[row*8..row*8+8];let (selected,unmasked,maximum,abs)=expected[row];
  let flag=result[3]>=0x7f80;assert_eq!(flag,abs>=0x7f80,"nonfinite flag row={row} fields={result:?} expected_abs={abs}");
  if flag {fallback+=1;println!("GREEDY_FALLBACK row={row} fields={result:?} CPU_selected={selected}");continue;}
  let actual=if result[0]==-65536{-result[2]}else{-result[1]};
  println!("GREEDY_FINITE row={row} selected={actual} expected={selected} fields={result:?}");
  assert_eq!(result[0],maximum,"masked max row={row}");assert_eq!(actual as usize,selected,"selected row={row}");
  assert_eq!(-result[2],unmasked as i32,"unmasked row={row}");assert_eq!(result[3],abs as i32,"maxabs row={row}");
 }
 println!("GREEDY_ORACLE_PASS rows=64 finite={} explicit_nonfinite_fallback={fallback} arbitrary_bans_allbanned_ties_signedzero_subnormals=true",64-fallback);
 let mut kernel=Vec::new();let mut small_read=Vec::new();
 for _ in 0..4 {
  let start=std::time::Instant::now();for _ in 0..3{launch(greedy::greedy_argmax_r64,(&mut device,&logits,&ban_words,&ids,&shifts,&mut out)).await?;}
  kernel.push(start.elapsed().as_secs_f64()*1000./3.);
  let start=std::time::Instant::now();let host=out.to_host::<GreedyResult>(&mut device.pdma).await?.into_vec();std::hint::black_box(host);
  small_read.push(start.elapsed().as_secs_f64()*1000.);
 }
 println!("GREEDY_TIMING warm_kernel_ms={kernel:?} result_read_2KiB_ms={small_read:?}");
 println!("GREEDY_DEVICE_PASS");Ok(())
}
