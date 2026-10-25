//! Public owning-buffer binding to the already-validated I4 projection image.
//! Only the input Rust owner type differs. No address construction or device copy.
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::{Buffers,DeviceSend};
use crate::{Chip,axes::{P512,H,L},ops_int4_group960_projection as kernel,
    ops_batch_layouts::{P512H960,P512L960}};
pub type Result<T> = std::result::Result<T,Box<dyn std::error::Error>>;
pub type PackedInput=m![P512/256,H/960,P512%256,H%960/8];
pub type PackedHidden=m![P512/256,L/960,P512%256,L%960/8];

fn bytes(elements:usize,bits:usize)->Result<usize> {
    let total=elements.checked_mul(bits).ok_or("tensor bit extent overflow")?;
    if total==0 || total%8!=0 {return Err("non-byte-integral tensor extent".into());}
    Ok(total/8)
}
fn expected_upgate()->Result<[usize;5]> {
    let packed=bytes(PackedInput::SIZE,32)?;
    if packed!=bytes(P512H960::SIZE,4)? {return Err("packed H geometry differs from typed I4 input".into());}
    Ok([packed,bytes(P512::SIZE,32)?,bytes(kernel::JoinedWeight::SIZE,4)?,
        bytes(kernel::JoinedBlockScale::SIZE,32)?,bytes(kernel::JoinedOutput::SIZE,16)?])
}
fn expected_down()->Result<[usize;6]> {
    let packed=bytes(PackedHidden::SIZE,32)?;
    if packed!=bytes(P512L960::SIZE,4)? {return Err("packed L geometry differs from typed I4 input".into());}
    Ok([packed,bytes(P512::SIZE,32)?,bytes(kernel::DownWeight::SIZE,4)?,
        bytes(kernel::DownBlockScale::SIZE,32)?,4,bytes(kernel::Output::SIZE,16)?])
}
fn validate(buffers:&Buffers,expected:&[usize])->Result<()> {
    if buffers.len()!=expected.len() {return Err("projection argument count mismatch".into());}
    for (buffer,&size) in buffers.iter().zip(expected) {
        if buffer.size()!=size {return Err("projection argument exact byte extent mismatch".into());}
        if buffer.addr()%256!=0 {return Err("projection requires aligned owning tensor starts".into());}
        if buffer.addr().checked_add(size).is_none() {return Err("projection address extent overflow".into());}
    }
    let output=buffers.last().ok_or("missing output")?;
    // All ends have been checked above. Immutable-input aliases are harmless;
    // no input may overlap the exclusively borrowed mutable output.
    let end=output.addr()+output.size();
    for input in &buffers[..buffers.len()-1] {
        if input.addr()<end && output.addr()<input.addr()+input.size() {
            return Err("projection output overlaps an immutable input".into());
        }
    }
    Ok(())
}
pub fn validate_upgate(buffers:&Buffers)->Result<()> {validate(buffers,&expected_upgate()?)}
pub fn validate_down(buffers:&Buffers)->Result<()> {validate(buffers,&expected_down()?)}

// These public bind helpers are also used by the standalone negative probe.
// They only clone public Buffer ownership metadata; no lookup or launch occurs.
pub fn bind_upgate(packed:&HbmTensor<i32,Chip,PackedInput>,xscale:&HbmTensor<f32,Chip,m![P512]>,
    weight:&HbmTensor<i4,Chip,kernel::JoinedWeight>,scale:&HbmTensor<f32,Chip,kernel::JoinedBlockScale>,
    output:&HbmTensor<bf16,Chip,kernel::JoinedOutput>)->Result<Buffers> {
    let mut b=Buffers::new();DeviceSend::bind(packed,&mut b)?;DeviceSend::bind(xscale,&mut b)?;
    DeviceSend::bind(weight,&mut b)?;DeviceSend::bind(scale,&mut b)?;DeviceSend::bind(output,&mut b)?;Ok(b)
}
pub fn bind_down(packed:&HbmTensor<i32,Chip,PackedHidden>,xscale:&HbmTensor<f32,Chip,m![P512]>,
    weight:&HbmTensor<i4,Chip,kernel::DownWeight>,scale:&HbmTensor<f32,Chip,kernel::DownBlockScale>,
    global:&HbmTensor<f32,Chip,m![1]>,output:&HbmTensor<bf16,Chip,kernel::Output>)->Result<Buffers> {
    let mut b=Buffers::new();DeviceSend::bind(packed,&mut b)?;DeviceSend::bind(xscale,&mut b)?;
    DeviceSend::bind(weight,&mut b)?;DeviceSend::bind(scale,&mut b)?;DeviceSend::bind(global,&mut b)?;
    DeviceSend::bind(output,&mut b)?;Ok(b)
}

pub async fn project_upgate(device:&mut Device,packed:&HbmTensor<i32,Chip,PackedInput>,xscale:&HbmTensor<f32,Chip,m![P512]>,
    weight:&HbmTensor<i4,Chip,kernel::JoinedWeight>,scale:&HbmTensor<f32,Chip,kernel::JoinedBlockScale>,
    output:&mut HbmTensor<bf16,Chip,kernel::JoinedOutput>)->Result<()> {
    let buffers=bind_upgate(packed,xscale,weight,scale,&*output)?;
    validate_upgate(&buffers)?; // Must precede lookup and every possible launch.
    #[cfg(backend="npu")]
    {
        // The normal typed arm embeds this exact path, with the same argument order.
        // Owners remain borrowed and Buffer clones remain alive through await.
        let f=furiosa_opt_std::backend::npu::function(device,kernel::project_upgate.path(),&[]).await?;
        f.run(&buffers,&[]).await?;
        Ok(())
    }
    #[cfg(not(backend="npu"))]
    {let _=device;Err("checked raw projection binding is NPU-only".into())}
}
pub async fn project_down(device:&mut Device,packed:&HbmTensor<i32,Chip,PackedHidden>,xscale:&HbmTensor<f32,Chip,m![P512]>,
    weight:&HbmTensor<i4,Chip,kernel::DownWeight>,scale:&HbmTensor<f32,Chip,kernel::DownBlockScale>,
    global:&HbmTensor<f32,Chip,m![1]>,output:&mut HbmTensor<bf16,Chip,kernel::Output>)->Result<()> {
    let buffers=bind_down(packed,xscale,weight,scale,global,&*output)?;
    validate_down(&buffers)?;
    #[cfg(backend="npu")]
    {
        let f=furiosa_opt_std::backend::npu::function(device,kernel::project_down.path(),&[]).await?;
        f.run(&buffers,&[]).await?;
        Ok(())
    }
    #[cfg(not(backend="npu"))]
    {let _=device;Err("checked raw projection binding is NPU-only".into())}
}
