//! Optional approximate B16 weights. Original E4 weights remain authoritative for
//! all other paths. Preparation publishes only a complete immutable48-layer set.
use crate::{Chip,axes::{H,L},ops_b16_f4_partialscale::{UpWeight,DownWeight,UpBlockScale,DownBlockScale}};
use furiosa_opt_std::prelude::*;
use super::{load::{Weights,Layer},native_fp8_weights::NativeFp8Weights};
type Result<T>=std::result::Result<T,Box<dyn std::error::Error>>;

/// q019 research quality candidate: explicit MseDyadic startup, B16-only dispatch.
pub const ENABLED:bool=true;

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum F4Quantizer { PowerOfTwoRange, MseDyadic }

#[derive(Clone,Debug,Default)]
pub struct StartupTiming {
 pub projections:usize,
 pub source_bytes:usize,
 pub packed_bytes:usize,
 pub scale_bytes:usize,
 pub read_ms:f64,
 pub encode_ms:f64,
 pub upload_ms:f64,
}
pub(crate) struct LayerWeights {
 pub up:HbmTensor<f4e2m1,Chip,UpWeight>,
 pub gate:HbmTensor<f4e2m1,Chip,UpWeight>,
 pub down:HbmTensor<f4e2m1,Chip,DownWeight>,
 pub up_scale:HbmTensor<f32,Chip,UpBlockScale>,
 pub gate_scale:HbmTensor<f32,Chip,UpBlockScale>,
 pub down_scale:HbmTensor<f32,Chip,DownBlockScale>,
}
pub(crate) struct Shadow {
 pub layers:Vec<LayerWeights>,
 pub quantizer:F4Quantizer,
 pub timing:StartupTiming,
}

async fn projection<E:M,S:M>(device:&mut Device,source:&HbmTensor<f8e4m3,Chip,E>,
 rows:usize,cols:usize,layer:usize,label:&str,quantizer:F4Quantizer,total:&mut StartupTiming)
 ->Result<(HbmTensor<f4e2m1,Chip,E>,HbmTensor<f32,Chip,S>)> {
 let begin=std::time::Instant::now();
 let raw=source.to_host::<E>(&mut device.pdma).await?.into_inner().into_buf();
 let read_ms=begin.elapsed().as_secs_f64()*1000.;let source_bytes=raw.len();
 let begin=std::time::Instant::now();
 let packed=match quantizer {
  F4Quantizer::PowerOfTwoRange=>super::f4_group960::quantize_blocked_e4_packed(&raw,rows,cols)?,
  F4Quantizer::MseDyadic=>super::f4_group960_mse::quantize_blocked_e4_mse(&raw,rows,cols)?,
 };
 let encode_ms=begin.elapsed().as_secs_f64()*1000.;
 if packed.packed_f4.len()*2!=E::SIZE || packed.group_scales.len()!=S::SIZE {
  return Err("F4 startup projection shape mismatch".into());
 }
 let packed_bytes=packed.packed_f4.len();let scale_bytes=packed.group_scales.len()*4;
 drop(raw); // No CPU source survives the upload or next projection.
 let begin=std::time::Instant::now();
 let weight=HostTensor::<f4e2m1,E>::from_buf(packed.packed_f4).to_hbm(&mut device.pdma).await?;
 let scale=HostTensor::<f32,S>::from_vec(packed.group_scales).to_hbm(&mut device.pdma).await?;
 let upload_ms=begin.elapsed().as_secs_f64()*1000.;
 total.projections+=1;total.source_bytes+=source_bytes;total.packed_bytes+=packed_bytes;total.scale_bytes+=scale_bytes;
 total.read_ms+=read_ms;total.encode_ms+=encode_ms;total.upload_ms+=upload_ms;
 println!("B16_F4_PREPARE layer={layer} projection={label} quantizer={quantizer:?} source_bytes={source_bytes} packed_bytes={packed_bytes} scale_bytes={scale_bytes} read_ms={read_ms:.6} encode_ms={encode_ms:.6} upload_ms={upload_ms:.6}");
 Ok((weight,scale))
}
impl LayerWeights {
 async fn prepare(device:&mut Device,n:&NativeFp8Weights,layer:usize,quantizer:F4Quantizer,total:&mut StartupTiming)->Result<Self>{
  let(up,up_scale)=projection(device,&n.up,L::SIZE,H::SIZE,layer,"up",quantizer,total).await?;
  let(gate,gate_scale)=projection(device,&n.gate,L::SIZE,H::SIZE,layer,"gate",quantizer,total).await?;
  let(down,down_scale)=projection(device,&n.down,H::SIZE,L::SIZE,layer,"down",quantizer,total).await?;
  Ok(Self{up,gate,down,up_scale,gate_scale,down_scale})
 }
}
impl Shadow {
 pub(crate) async fn prepare(device:&mut Device,weights:&Weights,quantizer:F4Quantizer)->Result<Self>{
  if weights.layers.len()!=crate::LAYERS{return Err("F4 startup requires complete model weights".into());}
  let mut layers=Vec::with_capacity(crate::LAYERS);let mut timing=StartupTiming::default();
  for(index,layer)in weights.layers.iter().enumerate(){
   let native=match layer{Layer::Sliding(l)=>&l.mlp.native_fp8,Layer::Full(l)=>&l.mlp.native_fp8};
   layers.push(LayerWeights::prepare(device,native,index,quantizer,&mut timing).await?);
  }
  // This value has never been exposed to Model. Any earlier error or canceled
  // future drops all partially prepared owners while originals remain intact.
  Ok(Self{layers,quantizer,timing})
 }
}
