//! Exact integer packing oracle. The device never quantizes floating-point input.
use furiosa_opt_gemma4::{Chip,ops_int4_pack_words as op,ops_int4_pack_words::PackNibbles};
use furiosa_opt_std::prelude::*;
#[path="support/npu_profile.rs"] mod npu_profile;
fn mix(mut x:u64)->u64 {x=(x^(x>>30)).wrapping_mul(0xbf58476d1ce4e5b9);x=(x^(x>>27)).wrapping_mul(0x94d049bb133111eb);x^(x>>31)}
fn fixture(case:usize)->Vec<i32> {
 (0..op::Codes::SIZE).map(|i| {let r=i/(480*8);let w=i/8%480;let n=i%8;
  match case {
   0=>0,
   1=>-1,
   2=>if n%2==0 {-8}else{7},
   3=>((w+r*3+n*5)%16)as i32-8,
   4=>((mix((r*3840+w*8+n)as u64+137)%16)as i32)-8,
   5=>{let r=(5*r+7)%16;let w=479-w;((mix((r*3840+w*8+n)as u64+137)%16)as i32)-8},
   6=>if n==w%8 {((w/8+r)%16)as i32-8}else{0},
   7=>if n==7 {((w+r)%16)as i32-8}else{((mix((w*7+n+r*101)as u64)%16)as i32)-8},
   _=>unreachable!()
  }
 }).collect()
}
fn pack(codes:&[i32])->Vec<u8> {
 assert_eq!(codes.len()%16,0);
 let mut bytes=Vec::with_capacity(codes.len()/2);
 for group in codes.chunks_exact(8) {
  let mut word=0u32;
  for (n,&c) in group.iter().enumerate(){assert!((-8..=7).contains(&c));word|=((c as u32)&15)<<(4*n);}
  bytes.extend_from_slice(&word.to_le_bytes());
 }
 bytes
}
fn prove_addsat(codes:&[i32]) {
 // Independently enumerate every possible partial subset for representative words.
 for group in codes.chunks_exact(8).step_by(29) {
  for subset in 0..256 {let mut sum=0i64;for (n,&c) in group.iter().enumerate(){if subset&(1<<n)!=0 {sum+=(((c as u32)&15)<<(4*n))as i32 as i64;}}
   assert!(sum>=i32::MIN as i64 && sum<=i32::MAX as i64);
  }
 }
}
async fn raw<D:MaterializableScalar,E:M>(d:&mut Device,x:&HbmTensor<D,Chip,E>)->Vec<u8>{x.to_host::<E>(&mut d.pdma).await.unwrap().into_inner().into_buf()}
#[tokio::main]
async fn main(){
 let profile=npu_profile::Collector::default();tracing::subscriber::set_global_default(profile.clone()).unwrap();
 let mut d=Device::new(op::pack_i32_codes.topology()).unwrap();
 let shifts_v:Vec<i32>=(0..8).map(|n|4*n).collect();
 let shifts=HostTensor::<i32,m![PackNibbles]>::from_vec(shifts_v.iter().copied()).to_hbm(&mut d.pdma).await.unwrap();
 let expected_shifts:Vec<u8>=shifts_v.iter().flat_map(|v|v.to_le_bytes()).collect();
 let base=pack(&fixture(4));
 for case in 0..8 {
  let codes=fixture(case);prove_addsat(&codes);let expected=pack(&codes);
  if case==5 {for r in 0..16 {for w in 0..480 {let from=((5*r+7)%16*480+479-w)*4;assert_eq!(&expected[(r*480+w)*4..(r*480+w+1)*4],&base[from..from+4]);}}}
  let code_bytes:Vec<u8>=codes.iter().flat_map(|v|v.to_le_bytes()).collect();
  let input=HostTensor::<i32,op::Codes>::from_vec(codes.iter().copied()).to_hbm(&mut d.pdma).await.unwrap();
  let mut output=HostTensor::<i32,op::PackedWords>::from_vec(std::iter::repeat_n(0x5a3c17e9i32,op::PackedWords::SIZE)).to_hbm(&mut d.pdma).await.unwrap();
  launch(op::pack_i32_codes,(&mut d,&input,&shifts,&mut output)).await.unwrap();
  let got=raw(&mut d,&output).await;
  let byte_diff=got.iter().zip(&expected).filter(|(a,b)|a!=b).count();
  let mut nibble_diff=0;for(i,&want)in codes.iter().enumerate(){let c=(got[i/2]>>(4*(i%2)))&15;let c=if c>=8{c as i32-16}else{c as i32};if c!=want{nibble_diff+=1;}}
  println!("INT4_PACK_EXACT case={case} bytes={} codes={} byte_diff={byte_diff} nibble_diff={nibble_diff}",got.len(),codes.len());
  assert_eq!(got.len(),30720);assert_eq!(byte_diff,0);assert_eq!(nibble_diff,0);
  assert_eq!(raw(&mut d,&input).await,code_bytes,"input mutated");assert_eq!(raw(&mut d,&shifts).await,expected_shifts,"shift input mutated");
  for repeat in 0..2 {let start=std::time::Instant::now();for _ in 0..20 {launch(op::pack_i32_codes,(&mut d,&input,&shifts,&mut output)).await.unwrap();}
   println!("INT4_PACK_TIME case={case} repeat={repeat} samples=20 us={:.6} scope=whole_pack_kernel_no_host_reads_or_uploads",start.elapsed().as_secs_f64()*50000.0);}
  if case==7 {println!("PROFILE_INT4_PACK");profile.enable(true);launch(op::pack_i32_codes,(&mut d,&input,&shifts,&mut output)).await.unwrap();profile.enable(false);}
 }
 println!("PASS exact denseI32 signed-nibble packing; no floating-point quantizer or raw-I4-consumer claim");
}
