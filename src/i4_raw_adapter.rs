//! Standalone checked public raw-buffer adapter. No tensor dtype conversion.
use furiosa_opt_std::prelude::*;
use crate::{Chip, axes::{B,H}, ops_i4_raw_binding as kernel};
use furiosa_opt_std::runtime::{Buffers,DeviceSend};

pub type Result<T> = std::result::Result<T,Box<dyn std::error::Error>>;
#[derive(Clone,Copy,Debug)]
pub struct PackedGeometry { pub rows:usize, pub codes_per_row:usize, pub words_per_row:usize }
impl PackedGeometry {
    pub const B16_H3840:Self=Self{rows:16,codes_per_row:3840,words_per_row:480};
    pub fn check(self,word_count:usize)->Result<()> {
        if self.rows!=B::SIZE || self.codes_per_row!=H::SIZE || self.codes_per_row%16!=0
            || self.words_per_row.checked_mul(8)!=Some(self.codes_per_row)
            || self.rows.checked_mul(self.words_per_row)!=Some(word_count) {
            return Err("packed geometry must be dense16x3840 codes in16x480 words".into());
        }
        Ok(())
    }
}
/// Checks all physical slots, including HBM output disjointness not checked by the runtime.
/// This only reads public Buffer metadata; it never creates or changes an address.
pub fn validate_bound(inputs:&Buffers)->Result<()> {
    if inputs.len()!=3 {return Err("I4 consumer requires three buffers".into());}
    let expected=[B::SIZE*H::SIZE/2,kernel::RawM32::SIZE*H::SIZE/2,kernel::RawM32::SIZE*B::SIZE*4];
    for (buffer,bytes) in inputs.iter().zip(expected) {
        if buffer.size()!=bytes {return Err("packed buffer exact byte extent mismatch".into());}
        if buffer.addr()%256!=0 {return Err("adapter requires whole aligned owning allocations".into());}
        if buffer.addr().checked_add(bytes).is_none() {return Err("buffer extent overflow".into());}
    }
    // Every argument is a whole owning HbmTensor, so no sub-buffer can hide overlap.
    // Still check the public numeric ranges before the normal runtime provenance check.
    let o=&inputs[2];let oe=o.addr()+o.size();
    for i in &inputs[..2] {
        if i.addr()<oe && o.addr()<i.addr()+i.size() {return Err("consumer output overlaps an input".into());}
    }
    Ok(())
}

/// The owner remains read-only until this future completes. Output is a separate
/// placed owner held exclusively. Buffer clones stay local and the runtime drains
/// an already submitted launch on cancellation before releasing those clones.
#[cfg(backend="npu")]
pub async fn run<E:M>(device:&mut Device, packed:&HbmTensor<i32,Chip,E>, geometry:PackedGeometry,
    weight:&HbmTensor<i4,Chip,kernel::Weight>,output:&mut HbmTensor<i32,Chip,kernel::Output>)->Result<()> {
    geometry.check(E::SIZE)?;
    let mut inputs=Buffers::new();
    DeviceSend::bind(packed,&mut inputs)?;
    DeviceSend::bind(weight,&mut inputs)?;
    DeviceSend::bind(&*output,&mut inputs)?;
    validate_bound(&inputs)?;
    // Actual typed control launch embeds this same verified consumer in the binary.
    let function=furiosa_opt_std::backend::npu::function(device,kernel::dot.path(),&[]).await?;
    // Same argument order/in-place output convention as the generated typed wrapper.
    // Runtime checks device ownership, count, memory kind and byte extent unchanged.
    function.run(&inputs,&[]).await?;
    Ok(())
}
#[cfg(not(backend="npu"))]
pub async fn run<E:M>(_:&mut Device,_:&HbmTensor<i32,Chip,E>,_:PackedGeometry,
    _:&HbmTensor<i4,Chip,kernel::Weight>,_:&mut HbmTensor<i32,Chip,kernel::Output>)->Result<()> {
    Err("raw owning-buffer adapter is NPU-only".into())
}

/// Intentional invalid-buffer checks, never a verification override. Short buffers
/// are rejected both by the adapter and by the ordinary runtime before submission.
#[cfg(backend="npu")]
pub async fn reject_bad_bindings<E:M>(device:&mut Device,packed:&HbmTensor<i32,Chip,E>,
 weight:&HbmTensor<i4,Chip,kernel::Weight>,output:&mut HbmTensor<i32,Chip,kernel::Output>)->Result<()> {
    let mut inputs=Buffers::new();DeviceSend::bind(packed,&mut inputs)?;
    DeviceSend::bind(weight,&mut inputs)?;DeviceSend::bind(&*output,&mut inputs)?;
    validate_bound(&inputs)?;
    let short=inputs[0].slice(0..inputs[0].size()-8);inputs[0]=short;
    assert!(validate_bound(&inputs).is_err());
    let function=furiosa_opt_std::backend::npu::function(device,kernel::dot.path(),&[]).await?;
    let error=function.run(&inputs,&[]).await.expect_err("runtime must reject undersized slot");
    assert!(error.to_string().contains("needs"),"unexpected runtime error {error}");
    println!("I4_RAW_REJECT undersized_adapter=true runtime={error}");
    Ok(())
}
#[cfg(not(backend="npu"))]
pub async fn reject_bad_bindings<E:M>(_:&mut Device,_:&HbmTensor<i32,Chip,E>,
 _:&HbmTensor<i4,Chip,kernel::Weight>,_:&mut HbmTensor<i32,Chip,kernel::Output>)->Result<()> {
    Err("raw owning-buffer adapter is NPU-only".into())
}
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn geometry_rejects_wrong_row_stride_and_extent() {
        assert!(PackedGeometry::B16_H3840.check(7680).is_ok());
        for g in [PackedGeometry{rows:8,..PackedGeometry::B16_H3840},PackedGeometry{codes_per_row:3839,..PackedGeometry::B16_H3840},PackedGeometry{words_per_row:481,..PackedGeometry::B16_H3840}] {assert!(g.check(7680).is_err());}
        assert!(PackedGeometry::B16_H3840.check(7679).is_err());
    }
}
