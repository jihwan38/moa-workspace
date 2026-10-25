//! Supplied-weight complete four-call prefix oracle, independent poisoned stages.
use furiosa_opt_std::prelude::*;
use crate::{Chip,EPS,axes::*,ops_b16_prefix as fused,ops_qkv_blocked_b16 as raw};
use super::load::{self,Layer};
type Result<T>=std::result::Result<T,Box<dyn std::error::Error>>;
const POS:[usize;16]=[0,1,15,16,127,255,511,512,1008,1023,1024,2047,4095,8191,16383,32767];
struct Scratch {
 norm:HbmTensor<bf16,Chip,m![B,H]>,quant:HbmTensor<f8e4m3,Chip,m![B,H]>,scale:HbmTensor<f32,Chip,m![B,1#8]>,
 qs:HbmTensor<bf16,Chip,raw::QueryOutput>,ks:HbmTensor<bf16,Chip,raw::KeyOutput>,vs:HbmTensor<bf16,Chip,raw::KeyOutput>,
 qf:HbmTensor<bf16,Chip,raw::FullQueryOutput>,kf:HbmTensor<bf16,Chip,raw::FullKeyOutput>,
 q_s:HbmTensor<bf16,Chip,m![B,Ns,Gs,Ds]>,k_s:HbmTensor<bf16,Chip,m![B,Ns,Ds]>,v_s:HbmTensor<bf16,Chip,m![B,Ns,Ds]>,
 q_f:HbmTensor<bf16,Chip,m![B,Gf,Df]>,k_f:HbmTensor<bf16,Chip,m![B,Df]>,v_f:HbmTensor<bf16,Chip,m![B,Df]>,
}
impl Scratch {
 async fn new(d:&mut Device)->Result<Self> {
  macro_rules! p {($ty:ty,$shape:ty,$value:expr)=>{HostTensor::<$ty,$shape>::from_vec((0..<$shape>::SIZE).map(|_|$value)).to_hbm(&mut d.pdma).await?};}
  macro_rules! bf {($shape:ty)=>{p!(bf16,$shape,bf16::from_bits(0x7fc1))};}
  Ok(Self {norm:bf!(m![B,H]),quant:p!(f8e4m3,m![B,H],f8e4m3::from_f32(f32::NAN)),scale:p!(f32,m![B,1#8],f32::NAN),
   qs:bf!(raw::QueryOutput),ks:bf!(raw::KeyOutput),vs:bf!(raw::KeyOutput),qf:bf!(raw::FullQueryOutput),kf:bf!(raw::FullKeyOutput),
   q_s:bf!(m![B,Ns,Gs,Ds]),k_s:bf!(m![B,Ns,Ds]),v_s:bf!(m![B,Ns,Ds]),
   q_f:bf!(m![B,Gf,Df]),k_f:bf!(m![B,Df]),v_f:bf!(m![B,Df])})
 }
 async fn snapshot(&self,d:&mut Device,sliding:bool)->Result<Vec<(&'static str,Vec<u8>)>> {
  macro_rules! get {($field:ident,$shape:ty)=>{(stringify!($field),self.$field.to_host::<$shape>(&mut d.pdma).await?.into_inner().into_buf())};}
  let raw=self.scale.to_host::<m![B,1#8]>(&mut d.pdma).await?.into_inner().into_buf();
  assert_eq!(raw.len(),B::SIZE*32);let live=raw.chunks_exact(32).flat_map(|row|row[..4].iter().copied()).collect();
  let mut out=vec![get!(norm,m![B,H]),get!(quant,m![B,H]),("scale",live)];
  if sliding {out.extend([get!(qs,raw::QueryOutput),get!(ks,raw::KeyOutput),get!(vs,raw::KeyOutput),get!(q_s,m![B,Ns,Gs,Ds]),get!(k_s,m![B,Ns,Ds]),get!(v_s,m![B,Ns,Ds])]);}
  else {out.extend([get!(qf,raw::FullQueryOutput),get!(kf,raw::FullKeyOutput),get!(q_f,m![B,Gf,Df]),get!(k_f,m![B,Df]),get!(v_f,m![B,Df])]);}
  Ok(out)
 }
}
struct Rope {
 rs:HbmTensor<i32,Chip,m![B]>,rf:HbmTensor<i32,Chip,m![B]>,
 cs:HbmTensor<bf16,Chip,m![E,Ds]>,ss:HbmTensor<bf16,Chip,m![E,Ds]>,
 cf:HbmTensor<bf16,Chip,m![E,Df]>,sf:HbmTensor<bf16,Chip,m![E,Df]>,
}
async fn tables<D:AxisName>(d:&mut Device,dim:usize,theta:f32,rotated:usize)->Result<(HbmTensor<bf16,Chip,m![E,D]>,HbmTensor<bf16,Chip,m![E,D]>)> {
 let mut cos=Vec::with_capacity(E::SIZE*dim);let mut sin=Vec::with_capacity(E::SIZE*dim);
 for pos in 0..E::SIZE {for half in 0..2 {for i in 0..dim/2 {
  if i<rotated/2 {let angle=pos as f32/theta.powf((2*i) as f32/dim as f32);cos.push(bf16::from_f32(angle.cos()));sin.push(bf16::from_f32(angle.sin()*if half==0 {-1.0}else{1.0}));}
  else {cos.push(bf16::from_f32(1.0));sin.push(bf16::from_f32(0.0));}
 }}}
 Ok((HostTensor::<bf16,m![E,D]>::from_vec(cos).to_hbm(&mut d.pdma).await?,HostTensor::<bf16,m![E,D]>::from_vec(sin).to_hbm(&mut d.pdma).await?))
}
async fn chain(d:&mut Device,input:&HbmTensor<bf16,Chip,m![B,H]>,s:&mut Scratch,l:&Layer,r:&Rope,candidate:bool)->Result<()> {
 let weight=match l {Layer::Sliding(l)=>&l.input_norm,Layer::Full(l)=>&l.input_norm};
 if !candidate {
  launch(crate::ops_norm_batch::rmsnorm_b16,(d,input,weight,&mut s.norm)).await?;
  launch(crate::ops_query_b16_native_fp8::quantize_input,(d,&s.norm,&mut s.quant,&mut s.scale)).await?;
 }
 match l {
  Layer::Sliding(l)=>if candidate {
   launch(fused::sliding,(d,input,&l.input_norm,&l.q_weight,&l.k_weight,&l.v_weight,&l.q_weight_scale,&l.k_weight_scale,&l.v_weight_scale,&l.q_norm,&l.k_norm,&r.rs,&r.cs,&r.ss,&mut s.norm,&mut s.quant,&mut s.scale,&mut s.qs,&mut s.ks,&mut s.vs,&mut s.q_s,&mut s.k_s,&mut s.v_s)).await?;
  } else {
   launch(raw::project_sliding_qkv,(d,&s.quant,&s.scale,&l.q_weight,&l.k_weight,&l.v_weight,&l.q_weight_scale,&l.k_weight_scale,&l.v_weight_scale,&mut s.qs,&mut s.ks,&mut s.vs)).await?;
   launch(crate::ops_head_b16_blocked::sliding_heads_b16,(d,&s.qs,&s.ks,&s.vs,&l.q_norm,&l.k_norm,&r.rs,&r.cs,&r.ss,&mut s.q_s,&mut s.k_s,&mut s.v_s)).await?;
  },
  Layer::Full(l)=>if candidate {
   launch(fused::full,(d,input,&l.input_norm,&l.q_weight,&l.k_weight,&l.q_weight_scale,&l.k_weight_scale,&l.q_norm,&l.k_norm,&r.rf,&r.cf,&r.sf,&mut s.norm,&mut s.quant,&mut s.scale,&mut s.qf,&mut s.kf,&mut s.q_f,&mut s.k_f,&mut s.v_f)).await?;
  } else {
   launch(raw::project_full_qk,(d,&s.quant,&s.scale,&l.q_weight,&l.k_weight,&l.q_weight_scale,&l.k_weight_scale,&mut s.qf,&mut s.kf)).await?;
   launch(crate::ops_head_b16_blocked::full_heads_b16,(d,&s.qf,&s.kf,&l.q_norm,&l.k_norm,&r.rf,&r.cf,&r.sf,&mut s.q_f,&mut s.k_f,&mut s.v_f)).await?;
  },
 }
 Ok(())
}
fn input_values(seed:usize,perm:bool)->Vec<bf16> {
 (0..B::SIZE*H::SIZE).map(|i|{let row=if perm {(i/H::SIZE*5+3)%16}else{i/H::SIZE};let col=i%H::SIZE;
  let base=((row*937+col*73+seed*131)%1009) as f32/504.-1.;
  bf16::from_f32(if row==0 {0.0}else if row==1 {base*1e-30}else{base*2f32.powi((row%9) as i32-4)})}).collect()
}
fn norm_oracle(input:&[bf16],weight:&[bf16],actual:&[u8])->Result<()> {
 let mut max_ulp=0u16;
 for row in 0..B::SIZE {let xs=&input[row*H::SIZE..(row+1)*H::SIZE];let rms=(xs.iter().map(|v|{let x=v.to_f32() as f64;x*x}).sum::<f64>()/H::SIZE as f64+EPS as f64).sqrt();
  for col in 0..H::SIZE {let i=(row*H::SIZE+col)*2;let got=u16::from_le_bytes([actual[i],actual[i+1]]);let expected=bf16::from_f32((xs[col].to_f32() as f64/rms*weight[col].to_f32() as f64) as f32).to_bits();max_ulp=max_ulp.max(got.abs_diff(expected));}
 }
 println!("B16_PREFIX_F64 norm_max_bf16_ulp={max_ulp}");assert!(max_ulp<=1);Ok(())
}
pub async fn run()->Result<()> {
 let mut d=Device::new(fused::sliding.topology())?;let weights=load::load_weights(&mut d).await?;
 let (cs,ss)=tables::<Ds>(&mut d,Ds::SIZE,10_000.,Ds::SIZE).await?;let (cf,sf)=tables::<Df>(&mut d,Df::SIZE,1_000_000.,128).await?;
 let rs=HostTensor::<i32,m![B]>::from_vec(POS.map(|p|(p*Ds::SIZE*2) as i32)).to_hbm(&mut d.pdma).await?;
 let rf=HostTensor::<i32,m![B]>::from_vec(POS.map(|p|(p*Df::SIZE*2) as i32)).to_hbm(&mut d.pdma).await?;
 let rope=Rope{rs,rf,cs,ss,cf,sf};let mut total=0usize;
 for (case,index) in [0usize,5,1,23,46,47].into_iter().enumerate() {
  let l=&weights.layers[index];let sliding=matches!(l,Layer::Sliding(_));let vals=input_values(case+1,case%2==1);
  let input=HostTensor::<bf16,m![B,H]>::from_vec(vals.clone()).to_hbm(&mut d.pdma).await?;
  let mut a=Scratch::new(&mut d).await?;let mut b=Scratch::new(&mut d).await?;
  chain(&mut d,&input,&mut b,l,&rope,true).await?;let candidate=b.snapshot(&mut d,sliding).await?;
  let untouched=input.to_host::<m![B,H]>(&mut d.pdma).await?.into_vec();assert!(vals.iter().zip(untouched).all(|(a,b)|a.to_bits()==b.to_bits()));
  chain(&mut d,&input,&mut a,l,&rope,false).await?;let control=a.snapshot(&mut d,sliding).await?;
  for ((name,a),(name_b,b)) in control.iter().zip(&candidate) {assert_eq!(name,name_b);assert_eq!(a.len(),b.len());let diff=a.iter().zip(b).filter(|(a,b)|a!=b).count();total+=a.len();println!("B16_PREFIX_EXACT case={case} layer={index} sliding={sliding} stage={name} bytes={} byte_diff={diff}",a.len());assert_eq!(diff,0);if *name!="quant"&&*name!="scale" {assert!(b.chunks_exact(2).all(|p|u16::from_le_bytes([p[0],p[1]])&0x7f80!=0x7f80));}}
  let w=match l {Layer::Sliding(l)=>&l.input_norm,Layer::Full(l)=>&l.input_norm};let w=w.to_host::<m![H]>(&mut d.pdma).await?.into_vec();norm_oracle(&vals,&w,&candidate[0].1)?;
  for (round,candidate) in [false,true,true,false].into_iter().enumerate(){let s=if candidate {&mut b}else{&mut a};let mut ms=0.;for _ in 0..8 {let t=std::time::Instant::now();chain(&mut d,&input,s,l,&rope,candidate).await?;ms+=t.elapsed().as_secs_f64()*1000.;}println!("B16_PREFIX_TIME case={case} layer={index} sliding={sliding} round={round} candidate={candidate} complete_ms={:.6} samples=8 excludes_upload_download=true",ms/8.);}
 }
 println!("PASS B16 prefix exact complete retained stages bytes={total}; no serving activation");Ok(())
}
