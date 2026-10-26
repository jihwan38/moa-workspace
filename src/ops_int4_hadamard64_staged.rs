//! Charged H64 low-bit routing through two real 8x8 F32 transposes.
//! No quantizer, model dispatch, or speed claim. Original rejected module is retained.
//! Each stage writes a NEW F32 owner: both old halves remain valid through sum/diff.
use crate::{Chip, axes::{H, P512}};
use furiosa_opt_std::prelude::*;
axes![H64Pair = 2];
type Cluster = m![P512 / 256];
type Rows = m![H / 960, P512 % 256 / 4];
type Local = m![P512 % 4, H % 960];
// Physical order inside each block changes; logical H coordinates do not.
type Staged = m![P512 % 4, H / 64 % 15, H % 8, H / 8 % 8];

/// CPU startup mask only; identical absolute-channel SplitMix policy to the frozen
/// sampled diagnostic. Runtime kernel receives these immutable 15,360 bytes.
pub fn sign_masks() -> Vec<i32> {
    (0..H::SIZE).map(|channel| {
        let mut z = ((channel as u64) ^ 0x243f_6a88_85a3_08d3).wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        if z >> 63 == 0 { 0 } else { i32::MIN }
    }).collect()
}

fn half<E: M, O: M, Time: M, Packet: M, NarrowTime: M, NarrowPacket: M>(
    device: &mut Device,
    low: DmTensorView<'_, f32, Chip, Cluster, Rows, E>,
    high: DmTensorView<'_, f32, Chip, Cluster, Rows, E>,
    output: DmTensorViewMut<'_, f32, Chip, Cluster, Rows, O>,
    op: FpBinaryOp,
) {
    // I is INNERMOST time, so only one8-F32 flit waits in the public unzip cache.
    // begin_interleaved reads both real DM views; the partner route is not a reshape.
    device.main.begin_interleaved::<H64Pair, f32, Chip, Cluster, Rows, E>(low, high)
        .fetch::<m![{Time}, H64Pair], Packet>()
        .collect::<m![{Time}, H64Pair], Packet>()
        .vector_init().vector_intra_slice_unzip::<H64Pair, Time>()
        .vector_narrow_split::<NarrowTime, NarrowPacket>()
        .vector_fp_zip(op)
        .vector_widen_concat::<Time, Packet>()
        .vector_final().commit_trim::<Packet>().commit_view(output);
}

// Both conversions are real post-fetch transpose operations into new owners.
// Packet source is always the source DM's contiguous innermost eight F32s.
fn to_staged(device: &mut Device, input: DmTensor<f32, Chip, Cluster, Rows, Local>)
    -> DmTensor<f32, Chip, Cluster, Rows, Staged> {
    device.main.begin(input.view())
        .fetch::<m![P512 % 4, H / 64 % 15, H / 8 % 8], m![H % 8]>()
        .collect::<m![P512 % 4, H / 64 % 15, H / 8 % 8], m![H % 8]>()
        .transpose::<m![P512 % 4, H / 64 % 15, H % 8], m![H / 8 % 8]>()
        .commit_trim::<m![H / 8 % 8]>().commit()
}
fn from_staged(device: &mut Device, input: DmTensor<f32, Chip, Cluster, Rows, Staged>)
    -> DmTensor<f32, Chip, Cluster, Rows, Local> {
    device.main.begin(input.view())
        .fetch::<m![P512 % 4, H / 64 % 15, H % 8], m![H / 8 % 8]>()
        .collect::<m![P512 % 4, H / 64 % 15, H % 8], m![H / 8 % 8]>()
        .transpose::<m![P512 % 4, H / 64 % 15, H / 8 % 8], m![H % 8]>()
        .commit_trim::<m![H % 8]>().commit()
}

// Stride1 selects a Time bit; all eight packet lanes H/8%8 are contiguous.
fn stage_1(device: &mut Device, input: DmTensor<f32, Chip, Cluster, Rows, Staged>)
    -> DmTensor<f32, Chip, Cluster, Rows, Staged> {
    let mut output = DmTensor::new();
    half::<_, _, m![P512 % 4, H / 64 % 15, H / 2 % 4], m![H / 8 % 8],
        m![P512 % 4, H / 64 % 15, H / 2 % 4, H / 32 % 2], m![H / 8 % 4]>(device,
        input.view().tile::<m![H % 2], 1, m![P512 % 4, H / 64 % 15, H / 2 % 4, H % 2 = 1 # 2, H / 8 % 8]>(0),
        input.view().tile::<m![H % 2], 1, m![P512 % 4, H / 64 % 15, H / 2 % 4, H % 2 = 1 # 2, H / 8 % 8]>(1),
        output.view_mut().tile::<m![H % 2], 1, m![P512 % 4, H / 64 % 15, H / 2 % 4, H % 2 = 1 #{!} 2, H / 8 % 8]>(0), FpBinaryOp::AddF);
    half::<_, _, m![P512 % 4, H / 64 % 15, H / 2 % 4], m![H / 8 % 8],
        m![P512 % 4, H / 64 % 15, H / 2 % 4, H / 32 % 2], m![H / 8 % 4]>(device,
        input.view().tile::<m![H % 2], 1, m![P512 % 4, H / 64 % 15, H / 2 % 4, H % 2 = 1 # 2, H / 8 % 8]>(0),
        input.view().tile::<m![H % 2], 1, m![P512 % 4, H / 64 % 15, H / 2 % 4, H % 2 = 1 # 2, H / 8 % 8]>(1),
        output.view_mut().tile::<m![H % 2], 1, m![P512 % 4, H / 64 % 15, H / 2 % 4, H % 2 = 1 #{!} 2, H / 8 % 8]>(1), FpBinaryOp::SubF);
    output
}

// Stride2 selects a Time bit; all eight packet lanes H/8%8 are contiguous.
fn stage_2(device: &mut Device, input: DmTensor<f32, Chip, Cluster, Rows, Staged>)
    -> DmTensor<f32, Chip, Cluster, Rows, Staged> {
    let mut output = DmTensor::new();
    half::<_, _, m![P512 % 4, H / 64 % 15, H / 4 % 2, H % 2], m![H / 8 % 8],
        m![P512 % 4, H / 64 % 15, H / 4 % 2, H % 2, H / 32 % 2], m![H / 8 % 4]>(device,
        input.view().tile::<m![H / 2 % 2], 1, m![P512 % 4, H / 64 % 15, H / 4 % 2, H / 2 % 2 = 1 # 2, H % 2, H / 8 % 8]>(0),
        input.view().tile::<m![H / 2 % 2], 1, m![P512 % 4, H / 64 % 15, H / 4 % 2, H / 2 % 2 = 1 # 2, H % 2, H / 8 % 8]>(1),
        output.view_mut().tile::<m![H / 2 % 2], 1, m![P512 % 4, H / 64 % 15, H / 4 % 2, H / 2 % 2 = 1 #{!} 2, H % 2, H / 8 % 8]>(0), FpBinaryOp::AddF);
    half::<_, _, m![P512 % 4, H / 64 % 15, H / 4 % 2, H % 2], m![H / 8 % 8],
        m![P512 % 4, H / 64 % 15, H / 4 % 2, H % 2, H / 32 % 2], m![H / 8 % 4]>(device,
        input.view().tile::<m![H / 2 % 2], 1, m![P512 % 4, H / 64 % 15, H / 4 % 2, H / 2 % 2 = 1 # 2, H % 2, H / 8 % 8]>(0),
        input.view().tile::<m![H / 2 % 2], 1, m![P512 % 4, H / 64 % 15, H / 4 % 2, H / 2 % 2 = 1 # 2, H % 2, H / 8 % 8]>(1),
        output.view_mut().tile::<m![H / 2 % 2], 1, m![P512 % 4, H / 64 % 15, H / 4 % 2, H / 2 % 2 = 1 #{!} 2, H % 2, H / 8 % 8]>(1), FpBinaryOp::SubF);
    output
}

// Stride4 selects a Time bit; all eight packet lanes H/8%8 are contiguous.
fn stage_4(device: &mut Device, input: DmTensor<f32, Chip, Cluster, Rows, Staged>)
    -> DmTensor<f32, Chip, Cluster, Rows, Staged> {
    let mut output = DmTensor::new();
    half::<_, _, m![P512 % 4, H / 64 % 15, H % 4], m![H / 8 % 8],
        m![P512 % 4, H / 64 % 15, H % 4, H / 32 % 2], m![H / 8 % 4]>(device,
        input.view().tile::<m![H / 4 % 2], 1, m![P512 % 4, H / 64 % 15, H / 4 % 2 = 1 # 2, H % 4, H / 8 % 8]>(0),
        input.view().tile::<m![H / 4 % 2], 1, m![P512 % 4, H / 64 % 15, H / 4 % 2 = 1 # 2, H % 4, H / 8 % 8]>(1),
        output.view_mut().tile::<m![H / 4 % 2], 1, m![P512 % 4, H / 64 % 15, H / 4 % 2 = 1 #{!} 2, H % 4, H / 8 % 8]>(0), FpBinaryOp::AddF);
    half::<_, _, m![P512 % 4, H / 64 % 15, H % 4], m![H / 8 % 8],
        m![P512 % 4, H / 64 % 15, H % 4, H / 32 % 2], m![H / 8 % 4]>(device,
        input.view().tile::<m![H / 4 % 2], 1, m![P512 % 4, H / 64 % 15, H / 4 % 2 = 1 # 2, H % 4, H / 8 % 8]>(0),
        input.view().tile::<m![H / 4 % 2], 1, m![P512 % 4, H / 64 % 15, H / 4 % 2 = 1 # 2, H % 4, H / 8 % 8]>(1),
        output.view_mut().tile::<m![H / 4 % 2], 1, m![P512 % 4, H / 64 % 15, H / 4 % 2 = 1 #{!} 2, H % 4, H / 8 % 8]>(1), FpBinaryOp::SubF);
    output
}

// Fixed stride8: pair h0,h0+8, h0/8%2=0; no partner crosses64.
fn stage_8(device: &mut Device, input: DmTensor<f32, Chip, Cluster, Rows, Local>)
    -> DmTensor<f32, Chip, Cluster, Rows, Local> {
    let mut output = DmTensor::new();
    half::<_, _, m![P512 % 4, H / 16 % 60], m![H % 8],
        m![P512 % 4, H / 16 % 60, H / 4 % 2], m![H % 4]>(device,
        input.view().tile::<m![H / 8 % 2], 1, m![P512 % 4, H / 16 % 60, H / 8 % 2 = 1 # 2, H % 8]>(0),
        input.view().tile::<m![H / 8 % 2], 1, m![P512 % 4, H / 16 % 60, H / 8 % 2 = 1 # 2, H % 8]>(1),
        output.view_mut().tile::<m![H / 8 % 2], 1, m![P512 % 4, H / 16 % 60, H / 8 % 2 = 1 #{!} 2, H % 8]>(0), FpBinaryOp::AddF);
    half::<_, _, m![P512 % 4, H / 16 % 60], m![H % 8],
        m![P512 % 4, H / 16 % 60, H / 4 % 2], m![H % 4]>(device,
        input.view().tile::<m![H / 8 % 2], 1, m![P512 % 4, H / 16 % 60, H / 8 % 2 = 1 # 2, H % 8]>(0),
        input.view().tile::<m![H / 8 % 2], 1, m![P512 % 4, H / 16 % 60, H / 8 % 2 = 1 # 2, H % 8]>(1),
        output.view_mut().tile::<m![H / 8 % 2], 1, m![P512 % 4, H / 16 % 60, H / 8 % 2 = 1 #{!} 2, H % 8]>(1), FpBinaryOp::SubF);
    output
}

// Fixed stride16: pair h0,h0+16, h0/16%2=0; no partner crosses64.
fn stage_16(device: &mut Device, input: DmTensor<f32, Chip, Cluster, Rows, Local>)
    -> DmTensor<f32, Chip, Cluster, Rows, Local> {
    let mut output = DmTensor::new();
    half::<_, _, m![P512 % 4, H / 32 % 30, H / 8 % 2], m![H % 8],
        m![P512 % 4, H / 32 % 30, H / 8 % 2, H / 4 % 2], m![H % 4]>(device,
        input.view().tile::<m![H / 16 % 2], 1, m![P512 % 4, H / 32 % 30, H / 16 % 2 = 1 # 2, H % 16]>(0),
        input.view().tile::<m![H / 16 % 2], 1, m![P512 % 4, H / 32 % 30, H / 16 % 2 = 1 # 2, H % 16]>(1),
        output.view_mut().tile::<m![H / 16 % 2], 1, m![P512 % 4, H / 32 % 30, H / 16 % 2 = 1 #{!} 2, H % 16]>(0), FpBinaryOp::AddF);
    half::<_, _, m![P512 % 4, H / 32 % 30, H / 8 % 2], m![H % 8],
        m![P512 % 4, H / 32 % 30, H / 8 % 2, H / 4 % 2], m![H % 4]>(device,
        input.view().tile::<m![H / 16 % 2], 1, m![P512 % 4, H / 32 % 30, H / 16 % 2 = 1 # 2, H % 16]>(0),
        input.view().tile::<m![H / 16 % 2], 1, m![P512 % 4, H / 32 % 30, H / 16 % 2 = 1 # 2, H % 16]>(1),
        output.view_mut().tile::<m![H / 16 % 2], 1, m![P512 % 4, H / 32 % 30, H / 16 % 2 = 1 #{!} 2, H % 16]>(1), FpBinaryOp::SubF);
    output
}

// Fixed stride32: pair h0,h0+32, h0/32%2=0; no partner crosses64.
fn stage_32(device: &mut Device, input: DmTensor<f32, Chip, Cluster, Rows, Local>)
    -> DmTensor<f32, Chip, Cluster, Rows, Local> {
    let mut output = DmTensor::new();
    half::<_, _, m![P512 % 4, H / 64 % 15, H / 8 % 4], m![H % 8],
        m![P512 % 4, H / 64 % 15, H / 8 % 4, H / 4 % 2], m![H % 4]>(device,
        input.view().tile::<m![H / 32 % 2], 1, m![P512 % 4, H / 64 % 15, H / 32 % 2 = 1 # 2, H % 32]>(0),
        input.view().tile::<m![H / 32 % 2], 1, m![P512 % 4, H / 64 % 15, H / 32 % 2 = 1 # 2, H % 32]>(1),
        output.view_mut().tile::<m![H / 32 % 2], 1, m![P512 % 4, H / 64 % 15, H / 32 % 2 = 1 #{!} 2, H % 32]>(0), FpBinaryOp::AddF);
    half::<_, _, m![P512 % 4, H / 64 % 15, H / 8 % 4], m![H % 8],
        m![P512 % 4, H / 64 % 15, H / 8 % 4, H / 4 % 2], m![H % 4]>(device,
        input.view().tile::<m![H / 32 % 2], 1, m![P512 % 4, H / 64 % 15, H / 32 % 2 = 1 # 2, H % 32]>(0),
        input.view().tile::<m![H / 32 % 2], 1, m![P512 % 4, H / 64 % 15, H / 32 % 2 = 1 # 2, H % 32]>(1),
        output.view_mut().tile::<m![H / 32 % 2], 1, m![P512 % 4, H / 64 % 15, H / 32 % 2 = 1 #{!} 2, H % 32]>(1), FpBinaryOp::SubF);
    output
}

/// Standalone H3840 resource gate only. Input/output/signs must be separate owners.
/// Sign mask must be sign_masks(); finite-normal-or-zero numerical scope only.
#[device(chip = 1)]
pub fn rotate_input(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![P512, H]>,
    signs: &HbmTensor<i32, Chip, m![H]>,
    output: &mut HbmTensor<bf16, Chip, m![P512, H]>,
) {
    const { assert!(P512::SIZE == 512 && H::SIZE == 3840 && H::SIZE % 960 == 0);
        assert!(Rows::SIZE == 256 && Cluster::SIZE == 2 && 960 % 64 == 0); }
    // Exactly current grouped-quant ownership: p=256c+4r+j, h=960g+u.
    let x: DmTensor<bf16, Chip, Cluster, Rows, Local> = input.to_dm(&mut device.tdma);
    let signs: DmTensor<i32, Chip, Cluster, Rows, m![H % 960]> = signs.to_dm(&mut device.tdma);
    let signs: VrfTensor<i32, Chip, Cluster, Rows, m![H % 960]> = device.sub.begin(signs.view())
        .fetch::<m![H / 8 % 120], m![H % 8]>()
        .collect::<m![H / 8 % 120], m![H % 8]>().to_vrf();
    let x: DmTensor<f32, Chip, Cluster, Rows, Local> = device.main.begin(x.view())
        .fetch::<m![P512 % 4, H / 16 % 60], m![H % 16]>().fetch_cast::<f32>()
        .collect::<m![P512 % 4, H / 8 % 120], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitXor, &signs)
        .vector_reinterpret::<f32>().vector_final().commit_trim::<m![H % 8]>().commit();
    let x = to_staged(device, x);
    let x = stage_1(device, x);
    let x = stage_2(device, x);
    let x = stage_4(device, x);
    let x = from_staged(device, x);
    let x = stage_8(device, x);
    let x = stage_16(device, x);
    let x = stage_32(device, x);
    let x: DmTensor<bf16, Chip, Cluster, Rows, Local> = device.main.begin(x.view())
        .fetch::<m![P512 % 4, H / 8 % 120], m![H % 8]>()
        .collect::<m![P512 % 4, H / 8 % 120], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P512 % 4, H / 4 % 240], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.125f32)
        .vector_widen_concat::<m![P512 % 4, H / 8 % 120], m![H % 8]>()
        .vector_final().cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>().commit();
    x.view().to_hbm_view(&mut device.tdma, output.view_mut());
}
