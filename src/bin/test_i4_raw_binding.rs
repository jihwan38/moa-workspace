//! Integer ABI proof only. No model quantizer, dtype cast, unsafe code or SDK change.
use furiosa_opt_gemma4::{Chip,axes::{B,H},ops_i4_raw_binding as k,i4_raw_adapter as raw,ops_int4_pack_words as pack};
use furiosa_opt_std::prelude::*;
use pack::PackNibbles;
use std::time::Instant;
#[path="support/npu_profile.rs"] mod npu_profile;
type Result<T>=std::result::Result<T,Box<dyn std::error::Error>>;
fn mix(mut x:u64)->u64{x=(x^(x>>30)).wrapping_mul(0xbf58476d1ce4e5b9);x=(x^(x>>27)).wrapping_mul(0x94d049bb133111eb);x^(x>>31)}
fn signed(code:usize)->i32 {let v=(code&15)as i32;if v>=8{v-16}else{v}}
// Two independent host encoders: pair-of-nibbles byte writer versus eight-field I32 sum.
fn packed_bytes(codes:&[i32])->Vec<u8>{assert_eq!(codes.len()%2,0);codes.chunks_exact(2).map(|c|{
 assert!(c.iter().all(|x|(-8..=7).contains(x)));((c[0]as u8)&15)|(((c[1]as u8)&15)<<4)
}).collect()}
fn words(codes:&[i32])->Vec<i32>{assert_eq!(codes.len()%8,0);codes.chunks_exact(8).map(|c|{
 let mut word=0u32;for(j,&x)in c.iter().enumerate(){assert!((-8..=7).contains(&x));word=word.checked_add(((x as u32)&15)<<(j*4)).unwrap();}word as i32
}).collect()}
fn fixture(case:usize)->(Vec<i32>,Vec<i32>){
 let x=(0..B::SIZE*H::SIZE).map(|i|{let r=i/H::SIZE;let h=i%H::SIZE;match case{
 0=>signed(r),1=>signed(h*5+r*3),2=>signed(mix((i+13)as u64)as usize),3=>-8,4=>7,5=>0,
 6=>if h%17==0{signed(r+h/17)}else{0},_=>signed(mix(((h*73+19)%H::SIZE+((r*5+3)%B::SIZE)*H::SIZE+13)as u64)as usize)
 }}).collect();
 let w=(0..k::RawM32::SIZE*H::SIZE).map(|i|{let m=i/H::SIZE;let h=i%H::SIZE;match case{
 0=>signed(m),1=>signed(h*7+m*11),2=>signed(mix((i+991)as u64)as usize),3=>if m%2==0{-8}else{7},4=>signed(h+m),5=>signed(mix(i as u64)as usize),
 6=>if h==(m*127+31)%H::SIZE{1}else{0},_=>signed(mix((m*H::SIZE+(h*73+19)%H::SIZE+991)as u64)as usize)
 }}).collect();(x,w)
}
fn oracle(x:&[i32],w:&[i32])->Vec<i32>{
 (0..k::RawM32::SIZE).flat_map(|m|(0..B::SIZE).map(move|b|{
  let sum=(0..H::SIZE).map(|h|x[b*H::SIZE+h]as i64*w[m*H::SIZE+h]as i64).sum::<i64>();i32::try_from(sum).unwrap()
 })).collect()
}
async fn upload<D:MaterializableScalar,E:M>(d:&mut Device,b:Vec<u8>)->HbmTensor<D,Chip,E>{HostTensor::<D,E>::from_buf(b).to_hbm(&mut d.pdma).await.unwrap()}
async fn read<D:MaterializableScalar,E:M>(d:&mut Device,x:&HbmTensor<D,Chip,E>)->Vec<u8>{x.to_host::<E>(&mut d.pdma).await.unwrap().into_inner().into_buf()}
fn values(bytes:&[u8])->Vec<i32>{bytes.chunks_exact(4).map(|b|i32::from_le_bytes(b.try_into().unwrap())).collect()}
#[tokio::main]
async fn main()->Result<()> {
 let collector=npu_profile::Collector::default();tracing::subscriber::set_global_default(collector.clone())?;
 let mut d=Device::new(k::dot.topology())?;
 let mut count=0usize;let mut checked_bytes=0usize;let mut packed_checked=0usize;
 let mut case2_outputs:Option<Vec<i32>>=None;
 for case in 0..8 {
  let(x,w)=fixture(case);let packed=packed_bytes(&x);let ww=packed_bytes(&w);let hostwords=words(&x);
  let wordbytes:Vec<u8>=hostwords.iter().flat_map(|w|w.to_le_bytes()).collect();assert_eq!(wordbytes,packed,"independent packing case={case}");
  assert_eq!(wordbytes.len(),B::SIZE*H::SIZE/2);let expected=oracle(&x,&w);
  let input:HbmTensor<i4,Chip,k::DenseInput>=upload(&mut d,packed.clone()).await;
  let weight:HbmTensor<i4,Chip,k::Weight>=upload(&mut d,ww.clone()).await;
  let packed_owner:HbmTensor<i32,Chip,k::PackedWords>=HostTensor::<i32,k::PackedWords>::from_vec(hostwords).to_hbm(&mut d.pdma).await?;
  let code_input:HbmTensor<i32,Chip,pack::Codes>=HostTensor::<i32,pack::Codes>::from_vec(x.clone()).to_hbm(&mut d.pdma).await?;
  let shifts:HbmTensor<i32,Chip,m![PackNibbles]>=HostTensor::<i32,m![PackNibbles]>::from_vec((0..8).map(|n|n*4).collect::<Vec<_>>()).to_hbm(&mut d.pdma).await?;
  let mut device_words:HbmTensor<i32,Chip,pack::PackedWords>=HostTensor::<i32,pack::PackedWords>::from_vec(vec![0x13579bdf;B::SIZE*H::SIZE/8]).to_hbm(&mut d.pdma).await?;
  let poison:Vec<u8>=(0..k::RawM32::SIZE*B::SIZE).flat_map(|i|(0x6a09e667i32^(i as i32)).to_le_bytes()).collect();
  let mut candidate:HbmTensor<i32,Chip,k::Output>=upload(&mut d,poison.clone()).await;
  let mut control:HbmTensor<i32,Chip,k::Output>=upload(&mut d,poison.clone()).await;
  let mut combined:HbmTensor<i32,Chip,k::Output>=upload(&mut d,poison.clone()).await;
  // The actual normal typed call both registers and validates the same consumer image.
  launch(k::dot,(&mut d,&input,&weight,&mut control)).await?;
  raw::run(&mut d,&packed_owner,raw::PackedGeometry::B16_H3840,&weight,&mut candidate).await?;
  let got=read(&mut d,&candidate).await;let old=read(&mut d,&control).await;
  assert_eq!(values(&old),expected,"typed integer oracle case={case}");
  assert_eq!(got,old,"raw vs typed case={case}");assert_eq!(values(&got),expected,"raw integer oracle case={case}");
  // This producer returns an ordinary I32 owner; the adapter accepts its own named
  // axes directly. Await both writes before each immutable raw binding.
  launch(pack::pack_i32_codes,(&mut d,&code_input,&shifts,&mut device_words)).await?;
  assert_eq!(read(&mut d,&device_words).await,packed,"device pack byte oracle case={case}");
  raw::run(&mut d,&device_words,raw::PackedGeometry::B16_H3840,&weight,&mut combined).await?;
  assert_eq!(read(&mut d,&combined).await,old,"device pack -> raw consumer case={case}");
  assert_eq!(read(&mut d,&device_words).await,packed,"consumer changed device packed owner");
  assert_eq!(values(&read(&mut d,&code_input).await),x,"producer input immutable");
  assert_eq!(values(&read(&mut d,&shifts).await),(0..8).map(|n|n*4).collect::<Vec<_>>(),"shifts immutable");
  packed_checked+=packed.len();
  if case==2 {case2_outputs=Some(values(&got));}
  if case==7 {let baseline=case2_outputs.as_ref().unwrap();for m in 0..k::RawM32::SIZE {for r in 0..B::SIZE {
   assert_eq!(expected[m*B::SIZE+r],baseline[m*B::SIZE+(r*5+3)%B::SIZE],"paired row/column permutation");
  }}}
  assert_eq!(read(&mut d,&packed_owner).await,packed,"I32 owner immutable");
  assert_eq!(read(&mut d,&input).await,packed,"typed I4 input immutable");assert_eq!(read(&mut d,&weight).await,ww,"weight immutable");
  checked_bytes+=packed.len()*2+ww.len();count+=expected.len();
  if case==0 {
   raw::reject_bad_bindings(&mut d,&packed_owner,&weight,&mut candidate).await?;
   assert_eq!(read(&mut d,&candidate).await,got,"rejected call touched output");
  }
  // Repeated awaited use proves the original owner remains valid across temporary Buffer drops.
  for _ in 0..3 {raw::run(&mut d,&packed_owner,raw::PackedGeometry::B16_H3840,&weight,&mut candidate).await?;}
  assert_eq!(read(&mut d,&candidate).await,got);
  let mut timing=[Vec::new(),Vec::new(),Vec::new()];
  for mode in [0,1,2,2,1,0] {let start=Instant::now();for _ in 0..8 {
   if mode==0{launch(k::dot,(&mut d,&input,&weight,&mut control)).await?;}
   else if mode==1{raw::run(&mut d,&packed_owner,raw::PackedGeometry::B16_H3840,&weight,&mut candidate).await?;}
   else {launch(pack::pack_i32_codes,(&mut d,&code_input,&shifts,&mut device_words)).await?;
    raw::run(&mut d,&device_words,raw::PackedGeometry::B16_H3840,&weight,&mut combined).await?;}
  }timing[mode].push(start.elapsed().as_secs_f64()*1e6/8.0);}
  assert_eq!(read(&mut d,&combined).await,old,"combined output after timed repetitions");
  assert_eq!(read(&mut d,&device_words).await,packed,"packed owner after timed repetitions");
  println!("I4_RAW_CASE case={case} values={} packed_bytes={} cpu_integer_exact=true typed_exact=true immutable=true typed_us={:?} raw_us={:?} pack_plus_raw_us={:?} raw_excludes_packing=true combined_includes_packing=true",expected.len(),packed.len(),timing[0],timing[1],timing[2]);
  if case==2&&std::env::var_os("FURIOSA_OPT_PROFILE").is_some(){collector.enable(true);raw::run(&mut d,&packed_owner,raw::PackedGeometry::B16_H3840,&weight,&mut candidate).await?;collector.enable(false);}
 }
 println!("PASS I4_RAW_ABI exact_outputs={count} immutable_input_bytes={checked_bytes} device_packed_bytes={packed_checked} combined_exact_outputs={count} true_paired_permutation=true all_signed_nibbles=true no_unsafe_or_dtype_cast=true no_quantizer_or_model=true cancellation_lifetime=source_audited_not_stress_tested");
 Ok(())
}
#[cfg(test)] mod tests{
 use super::*;
 #[test]fn fixture_seven_is_paired_permutation(){let(x,w)=fixture(2);let(px,pw)=fixture(7);for r in 0..B::SIZE{for h in 0..H::SIZE{assert_eq!(px[r*H::SIZE+h],x[((r*5+3)%B::SIZE)*H::SIZE+(h*73+19)%H::SIZE]);}}for m in 0..k::RawM32::SIZE{for h in 0..H::SIZE{assert_eq!(pw[m*H::SIZE+h],w[m*H::SIZE+(h*73+19)%H::SIZE]);}}}
 #[test]fn all_codes_words_and_nibbles_agree(){for rotate in 0..16{let c:Vec<i32>=(0..128).map(|i|signed(i+rotate)).collect();let b:Vec<u8>=words(&c).iter().flat_map(|w|w.to_le_bytes()).collect();assert_eq!(b,packed_bytes(&c));for(i,&x)in c.iter().enumerate(){let code=((b[i/2]>>(4*(i%2)))&15)as usize;assert_eq!(signed(code),x);}}}
}
