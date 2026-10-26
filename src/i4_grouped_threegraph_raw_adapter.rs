//! Public checked multi-output binding. Only the entry packed input changes dtype.
//! No graph contains aliased I32/I4 formal owners; all intermediates stay HBM.
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::{Buffers, DeviceSend};
use crate::{Chip, axes::{H,L}, ops_batch_layouts::{P512H960,P512L960}};
use crate::{ops_int4_grouped_threegraph as kernel, ops_int4_grouped_activation_projection as projection,
    ops_int4_ffn_bridges as bridge, ops_int4_grouped_activation_store as quant,
    ops_norm_blocked_residual as tail};
use quant::{Mantissa, Record};
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn bytes(n:usize,bits:usize)->Result<usize>{
    let bits=n.checked_mul(bits).ok_or("tensor extent overflow")?;
    if bits==0 || bits%8!=0{return Err("invalid tensor byte extent".into());}
    Ok(bits/8)
}
fn expected_upgate_hidden()->Result<[usize;13]>{
    let packed=bytes(quant::PackedInput::SIZE,32)?;
    if packed!=bytes(P512H960::SIZE,4)?{return Err("packed H does not match typed I4 geometry".into());}
    Ok([packed,bytes(projection::InputScale::SIZE,32)?,bytes(projection::JoinedWeight::SIZE,4)?,
        bytes(projection::JoinedBlockScale::SIZE,32)?,4,4,bytes(<m![Mantissa,Record]>::SIZE,32)?,32,
        bytes(projection::JoinedOutput::SIZE,16)?,bytes(bridge::HiddenBf16::SIZE,16)?,
        bytes(quant::PackedHidden::SIZE,32)?,bytes(quant::HiddenScales::SIZE,32)?,bytes(quant::HiddenScales::SIZE,32)?])
}
fn expected_down_tail()->Result<[usize;9]>{
    let packed=bytes(quant::PackedHidden::SIZE,32)?;
    if packed!=bytes(P512L960::SIZE,4)?{return Err("packed L does not match typed I4 geometry".into());}
    Ok([packed,bytes(projection::HiddenScale::SIZE,32)?,bytes(projection::DownWeight::SIZE,4)?,
        bytes(projection::DownBlockScale::SIZE,32)?,4,bytes(H::SIZE,16)?,16,
        bytes(projection::Output::SIZE,16)?,bytes(tail::Residual::SIZE,16)?])
}
fn validate(buffers:&Buffers,expected:&[usize],mutable_start:usize)->Result<()>{
    if buffers.len()!=expected.len(){return Err("graph argument count mismatch".into());}
    for (b,&n) in buffers.iter().zip(expected){
        if b.size()!=n{return Err("graph exact byte extent mismatch".into());}
        if b.addr()%256!=0{return Err("graph owning tensor start is not aligned".into());}
        b.addr().checked_add(n).ok_or("graph address extent overflow")?;
    }
    // Every mutable owner may be read and written by different stage helpers.
    // Reject overlap with ALL other arguments, including other mutable owners.
    // Immutable-input aliases are allowed; validated checked ends cannot overflow.
    for i in mutable_start..buffers.len(){
        for j in 0..i{
            let a=&buffers[i];let b=&buffers[j];
            if a.addr()<b.addr()+b.size() && b.addr()<a.addr()+a.size(){
                return Err("mutable graph owner overlaps another argument".into());
            }
        }
    }
    Ok(())
}
pub fn validate_upgate_hidden(b:&Buffers)->Result<()>{validate(b,&expected_upgate_hidden()?,8)}
pub fn validate_down_tail(b:&Buffers)->Result<()>{validate(b,&expected_down_tail()?,7)}

pub fn bind_upgate_hidden(
    input: &HbmTensor<i32, Chip, quant::PackedInput>,
    input_scales: &HbmTensor<f32, Chip, projection::InputScale>,
    weight: &HbmTensor<i4, Chip, projection::JoinedWeight>,
    weight_scales: &HbmTensor<f32, Chip, projection::JoinedBlockScale>,
    up_global: &HbmTensor<f32, Chip, m![1]>,
    gate_global: &HbmTensor<f32, Chip, m![1]>,
    table: &HbmTensor<i32, Chip, m![Mantissa, Record]>,
    shifts: &HbmTensor<i32, Chip, m![L % 8]>,
    joined: &HbmTensor<bf16, Chip, projection::JoinedOutput>,
    hidden: &HbmTensor<bf16, Chip, bridge::HiddenBf16>,
    packed: &HbmTensor<i32, Chip, quant::PackedHidden>,
    scales: &HbmTensor<f32, Chip, quant::HiddenScales>,
    offsets: &HbmTensor<i32, Chip, quant::HiddenScales>,
) -> Result<Buffers> {
    let mut b=Buffers::new();
    DeviceSend::bind(input,&mut b)?;
    DeviceSend::bind(input_scales,&mut b)?;
    DeviceSend::bind(weight,&mut b)?;
    DeviceSend::bind(weight_scales,&mut b)?;
    DeviceSend::bind(up_global,&mut b)?;
    DeviceSend::bind(gate_global,&mut b)?;
    DeviceSend::bind(table,&mut b)?;
    DeviceSend::bind(shifts,&mut b)?;
    DeviceSend::bind(joined,&mut b)?;
    DeviceSend::bind(hidden,&mut b)?;
    DeviceSend::bind(packed,&mut b)?;
    DeviceSend::bind(scales,&mut b)?;
    DeviceSend::bind(offsets,&mut b)?;
    Ok(b)
}

pub async fn upgate_hidden(
    device: &mut Device,
    input: &HbmTensor<i32, Chip, quant::PackedInput>,
    input_scales: &HbmTensor<f32, Chip, projection::InputScale>,
    weight: &HbmTensor<i4, Chip, projection::JoinedWeight>,
    weight_scales: &HbmTensor<f32, Chip, projection::JoinedBlockScale>,
    up_global: &HbmTensor<f32, Chip, m![1]>,
    gate_global: &HbmTensor<f32, Chip, m![1]>,
    table: &HbmTensor<i32, Chip, m![Mantissa, Record]>,
    shifts: &HbmTensor<i32, Chip, m![L % 8]>,
    joined: &mut HbmTensor<bf16, Chip, projection::JoinedOutput>,
    hidden: &mut HbmTensor<bf16, Chip, bridge::HiddenBf16>,
    packed: &mut HbmTensor<i32, Chip, quant::PackedHidden>,
    scales: &mut HbmTensor<f32, Chip, quant::HiddenScales>,
    offsets: &mut HbmTensor<i32, Chip, quant::HiddenScales>,
) -> Result<()> {
    let buffers=bind_upgate_hidden(input, input_scales, weight, weight_scales, up_global, gate_global, table, shifts, joined, hidden, packed, scales, offsets)?;
    validate_upgate_hidden(&buffers)?; // Before lookup or any launch.
    #[cfg(backend="npu")]
    {
        let function=furiosa_opt_std::backend::npu::function(device,kernel::upgate_hidden.path(),&[]).await?;
        function.run(&buffers,&[]).await?;
        Ok(())
    }
    #[cfg(not(backend="npu"))]
    {let _=device;Err("raw three-graph binding is NPU-only".into())}
}

pub fn bind_down_tail(
    input: &HbmTensor<i32, Chip, quant::PackedHidden>,
    input_scales: &HbmTensor<f32, Chip, projection::HiddenScale>,
    weight: &HbmTensor<i4, Chip, projection::DownWeight>,
    weight_scales: &HbmTensor<f32, Chip, projection::DownBlockScale>,
    global: &HbmTensor<f32, Chip, m![1]>,
    post: &HbmTensor<bf16, Chip, m![H]>,
    layer_gate: &HbmTensor<bf16, Chip, m![1#8]>,
    down: &HbmTensor<bf16, Chip, projection::Output>,
    residual: &HbmTensor<bf16, Chip, tail::Residual>,
) -> Result<Buffers> {
    let mut b=Buffers::new();
    DeviceSend::bind(input,&mut b)?;
    DeviceSend::bind(input_scales,&mut b)?;
    DeviceSend::bind(weight,&mut b)?;
    DeviceSend::bind(weight_scales,&mut b)?;
    DeviceSend::bind(global,&mut b)?;
    DeviceSend::bind(post,&mut b)?;
    DeviceSend::bind(layer_gate,&mut b)?;
    DeviceSend::bind(down,&mut b)?;
    DeviceSend::bind(residual,&mut b)?;
    Ok(b)
}

pub async fn down_tail(
    device: &mut Device,
    input: &HbmTensor<i32, Chip, quant::PackedHidden>,
    input_scales: &HbmTensor<f32, Chip, projection::HiddenScale>,
    weight: &HbmTensor<i4, Chip, projection::DownWeight>,
    weight_scales: &HbmTensor<f32, Chip, projection::DownBlockScale>,
    global: &HbmTensor<f32, Chip, m![1]>,
    post: &HbmTensor<bf16, Chip, m![H]>,
    layer_gate: &HbmTensor<bf16, Chip, m![1#8]>,
    down: &mut HbmTensor<bf16, Chip, projection::Output>,
    residual: &mut HbmTensor<bf16, Chip, tail::Residual>,
) -> Result<()> {
    let buffers=bind_down_tail(input, input_scales, weight, weight_scales, global, post, layer_gate, down, residual)?;
    validate_down_tail(&buffers)?; // Before lookup or any launch.
    #[cfg(backend="npu")]
    {
        let function=furiosa_opt_std::backend::npu::function(device,kernel::down_tail.path(),&[]).await?;
        function.run(&buffers,&[]).await?;
        Ok(())
    }
    #[cfg(not(backend="npu"))]
    {let _=device;Err("raw three-graph binding is NPU-only".into())}
}
