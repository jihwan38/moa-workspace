//! Checked public owning-buffer binding for blocked packed I32 words.
//! No device tensor reinterpretation, fabricated address or private SDK API.
use crate::{
    Chip,
    axes::{H, L, P512},
    ops_int4_pack_p512 as kernel,
};
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::{Buffers, DeviceSend};
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub fn validate_bound(buffers: &Buffers, width: usize) -> Result<()> {
    if ![H::SIZE, L::SIZE].contains(&width) || buffers.len() != 2 {
        return Err("expected two full-P512 H3840 or L15360 owners".into());
    }
    let bytes = P512::SIZE
        .checked_mul(width)
        .and_then(|x| x.checked_div(2))
        .ok_or("packed extent overflow")?;
    for buffer in buffers.iter() {
        if buffer.size() != bytes
            || buffer.addr() % 256 != 0
            || buffer.addr().checked_add(bytes).is_none()
        {
            return Err("raw I4 binding requires exact, aligned whole-owner extents".into());
        }
    }
    let input = &buffers[0];
    let output = &buffers[1];
    if input.addr() < output.addr() + bytes && output.addr() < input.addr() + bytes {
        return Err("raw reader output overlaps immutable packed owner".into());
    }
    Ok(())
}

#[cfg(backend = "npu")]
async fn run<P: M, O: M>(
    device: &mut Device,
    packed: &HbmTensor<i32, Chip, P>,
    output: &mut HbmTensor<i4, Chip, O>,
    width: usize,
    path: &'static str,
) -> Result<()> {
    if P::SIZE.checked_mul(8) != Some(P512::SIZE * width) || O::SIZE != P512::SIZE * width {
        return Err("declared packed/reader mappings have different live extents".into());
    }
    let mut buffers = Buffers::new();
    DeviceSend::bind(packed, &mut buffers)?;
    DeviceSend::bind(&*output, &mut buffers)?;
    validate_bound(&buffers, width)?;
    // Typed probe launch embeds this same consumer. Owners remain borrowed and
    // clones remain live through await; the runtime handles cancellation drain.
    let function = furiosa_opt_std::backend::npu::function(device, path, &[]).await?;
    function.run(&buffers, &[]).await?;
    Ok(())
}

pub async fn read_input(
    device: &mut Device,
    packed: &HbmTensor<i32, Chip, kernel::PackedInput>,
    output: &mut HbmTensor<i4, Chip, m![P512, H]>,
) -> Result<()> {
    #[cfg(backend = "npu")]
    {
        run(device, packed, output, H::SIZE, kernel::read_input.path()).await
    }
    #[cfg(not(backend = "npu"))]
    {
        let _ = (device, packed, output);
        Err("public raw binding is NPU-only".into())
    }
}

pub async fn read_hidden(
    device: &mut Device,
    packed: &HbmTensor<i32, Chip, kernel::PackedHidden>,
    output: &mut HbmTensor<i4, Chip, m![P512, L]>,
) -> Result<()> {
    #[cfg(backend = "npu")]
    {
        run(device, packed, output, L::SIZE, kernel::read_hidden.path()).await
    }
    #[cfg(not(backend = "npu"))]
    {
        let _ = (device, packed, output);
        Err("public raw binding is NPU-only".into())
    }
}
