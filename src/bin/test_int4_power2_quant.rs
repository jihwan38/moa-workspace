//! New, independently specified power2 A4 gate. Old amax/7 DivF failures remain held.
use furiosa_opt_gemma4::{Chip,axes::{B,H},host::load,ops_int4_power2_quant as op};
use furiosa_opt_std::prelude::*;
#[path="support/npu_profile.rs"] mod npu_profile;
fn mix(mut x:u64)->u64{x=(x^(x>>30)).wrapping_mul(0xbf58476d1ce4e5b9);x=(x^(x>>27)).wrapping_mul(0x94d049bb133111eb);x^(x>>31)}
fn ftz(x:f32)->f64{if x.abs()<f32::MIN_POSITIVE{0.0}else{x as f64}}
fn reference(row:&[bf16])->(Vec<i32>,f32){
 let a=row.iter().map(|v|ftz(v.to_f32()).abs()).fold(2f64.powi(-112),f64::max);
 // Independent comparisons: no exponent masks, logarithms or approximate division.
 let mut s=2f64.powi(-114);while 7.0*s<a{s*=2.0;}
 let q=row.iter().map(|v|(ftz(v.to_f32())/s).round().clamp(-7.0,7.0)as i32).collect();(q,s as f32)
}
fn fixture(case:usize)->Vec<bf16>{
 if case==7 {let base=fixture(6);return (0..16*3840).map(|i|base[((5*(i/3840)+7)%16)*3840+3839-i%3840]).collect();}
 let exps=[-126,-114,-112,-50,-1,0,1,10,50,100,124,125,126,127,-80,20];
 let mant=[0,1,63,95,96,97,126,127];
 let mut out=Vec::with_capacity(16*3840);
 for r in 0..16 {
  let bits=(((exps[r]+127)as u16)<<7)|mant[(case+r)%8];
  let a=bf16::from_bits(bits).to_f32();let mut row=Vec::with_capacity(3840);
  for k in 0..3840 {let v=if k==0{a}else if k==1{-a}else{let n=((mix((r*3840+k)as u64+case as u64*100003)>>40)as f32)/16777216.;a*(2.*n-1.)};row.push(bf16::from_f32(v));}
  if case==0&&r<4 {let v=match r{0=>0u16,1=>0x8000,2=>0x007f,_=>0x0080};for(k,x)in row.iter_mut().enumerate(){*x=bf16::from_bits(v|if k%2==0{0}else{0x8000});}}
  else {let(_,s)=reference(&row);let mut at=2;for n in 0..7 {let t=(n as f32+0.5)*s;if t.is_finite()&&t<=a {let b=bf16::from_f32(t).to_bits();for b in[b.saturating_sub(1),b,b+1]{let x=bf16::from_bits(b).to_f32();if x.is_finite()&&x<=a{row[at]=bf16::from_bits(b);row[at+1]=bf16::from_bits(b|0x8000);at+=2;}}}}}
  out.extend(row);
 }
 out
}
async fn raw<D:MaterializableScalar,E:M>(d:&mut Device,x:&HbmTensor<D,Chip,E>)->Vec<u8>{x.to_host::<E>(&mut d.pdma).await.unwrap().into_inner().into_buf()}
fn e4(b:u8)->f32{let sign=if b&128==0{1.}else{-1.};let e=(b>>3)&15;let m=b&7;assert!(e!=15||m!=7,"nonfinite provided E4 blockscale");sign*if e==0{(m as f32)*2f32.powi(-9)}else{(8+m)as f32*2f32.powi(e as i32-10)}}
fn f4(b:u8)->f32{let a=[0.,0.5,1.,1.5,2.,3.,4.,6.][(b&7)as usize];if b&8==0{a}else{-a}}
fn supplied_weight_stats()->Result<(),Box<dyn std::error::Error>>{
 let dir=match load::model_dir(){Ok(d)=>d,Err(e)=>{println!("A4P2_WEIGHT unavailable={e}");return Ok(());}};
 let file=std::fs::File::open(dir.join("model.safetensors"))?;let mmap=unsafe{memmap2::Mmap::map(&file)?};let tensors=safetensors::SafeTensors::deserialize(&mmap)?;
 for layer in[0,6,47] {for projection in["up_proj","gate_proj","down_proj"]{
  let prefix=format!("model.language_model.layers.{layer}.mlp.{projection}");let packed=tensors.tensor(&format!("{prefix}.weight_packed"))?;let scales=tensors.tensor(&format!("{prefix}.weight_scale"))?;
  let shape=packed.shape();assert_eq!(shape.len(),2);let rows=shape[0];let width=shape[1]*2;assert_eq!(scales.shape(),[rows,width/16]);assert_eq!(packed.data().len(),rows*width/2);assert_eq!(scales.data().len(),rows*width/16);
  let(mut norm,mut row_error,mut block_error)=(0f64,0f64,0f64);
  for pick in 0..64 {let r=if pick<4{[0,rows/2-1,rows/2,rows-1][pick]}else{mix(pick as u64+layer as u64*313)as usize%rows};
   let values:Vec<bf16>=(0..width).map(|k|{let b=packed.data()[r*width/2+k/2];bf16::from_f32(f4((b>>(4*(k%2)))&15)*e4(scales.data()[r*width/16+k/16]))}).collect();
   let(q,s)=reference(&values);for(k,v)in values.iter().enumerate(){let v=v.to_f32()as f64;norm+=v*v;row_error+=(q[k]as f64*s as f64-v).powi(2);}
   for v in values.chunks_exact(960){let(q,s)=reference(v);for(i,x)in v.iter().enumerate(){block_error+=(q[i]as f64*s as f64-x.to_f32()as f64).powi(2);}}
  }
  println!("A4P2_WEIGHT layer={layer} projection={projection} sampled_rows=64 values={} row_relative_l2={} block960_relative_l2={} source=provided_F4_times_E4_before_global",64*width,(row_error/norm).sqrt(),(block_error/norm).sqrt());
 }}Ok(())
}
#[tokio::main]
async fn main()->Result<(),Box<dyn std::error::Error>>{
 let profile=npu_profile::Collector::default();tracing::subscriber::set_global_default(profile.clone())?;
 let mut d=Device::new(op::quantize_row.topology())?;
 for case in 0..8 {
  let values=fixture(case);assert!(values.iter().all(|v|v.to_f32().is_finite()));
  let mut expected=Vec::new();let mut scale_expected=Vec::new();for row in values.chunks_exact(3840){let(q,s)=reference(row);expected.extend(q);scale_expected.push(s);}
  let input_bytes:Vec<u8>=values.iter().flat_map(|v|v.to_bits().to_le_bytes()).collect();
  let x=HostTensor::<bf16,m![B,H]>::from_vec(values).to_hbm(&mut d.pdma).await?;
  let mut q=HostTensor::<i32,m![B,H]>::from_vec(std::iter::repeat_n(0x5a3c17e9i32,16*3840)).to_hbm(&mut d.pdma).await?;
  let mut s=HostTensor::<f32,m![B]>::from_vec(std::iter::repeat_n(f32::from_bits(0x7fc10001),16)).to_hbm(&mut d.pdma).await?;
  launch(op::quantize_row,(&mut d,&x,&mut q,&mut s)).await?;
  let qr=raw(&mut d,&q).await;let sr=raw(&mut d,&s).await;
  let got:Vec<i32>=qr.chunks_exact(4).map(|b|i32::from_le_bytes(b.try_into().unwrap())).collect();
  let mut code_diff=0;for(i,(&a,&b))in got.iter().zip(&expected).enumerate(){if a!=b {if code_diff<8{println!("A4P2_CODE_DIFF case={case} row={} col={} actual={a} expected={b}",i/3840,i%3840);}code_diff+=1;}}
  let scale_diff=sr.chunks_exact(4).zip(&scale_expected).filter(|(b,v)|u32::from_le_bytes((*b).try_into().unwrap())!=v.to_bits()).count();
  println!("A4P2_EXACT case={case} codes={} code_diff={code_diff} scales=16 scale_diff={scale_diff}",expected.len());
  assert_eq!(code_diff,0);assert_eq!(scale_diff,0);assert_eq!(raw(&mut d,&x).await,input_bytes,"input mutated");
  for repeat in 0..2{let start=std::time::Instant::now();for _ in 0..20{launch(op::quantize_row,(&mut d,&x,&mut q,&mut s)).await?;}println!("A4P2_TIME case={case} repeat={repeat} samples=20 us={:.6} scope=complete_I32_code_quantizer_without_pack_or_consumer",start.elapsed().as_secs_f64()*50000.);}
  if case==7{println!("PROFILE_A4P2");profile.enable(true);launch(op::quantize_row,(&mut d,&x,&mut q,&mut s)).await?;profile.enable(false);}
 }
 supplied_weight_stats()?;
 println!("PASS new finiteBF16 power2A4 codes/scales; oldamax_div7 remainsHOLD; no modelquality claim");Ok(())
}
