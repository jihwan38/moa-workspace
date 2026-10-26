//! Canonical BF16 boundaries for a disconnected complete-I4-FFN diagnostic.
//! Existing normalization, global scaling and GeGLU arithmetic are reused.
//! Both canonical layout conversions are real DM/HBM movement, not aliases.
use crate::{Chip, axes::{H, L, P128, P512}};
use crate::ops_ffn_projection_clusters::{FfnProjection, JoinedOutput, global_dm, geglu_tile_into};
use crate::ops_norm_blocked_residual::{Residual, load, store};
use crate::ops_norm_quarters::{Q4, load_projected, load_residual, normalize, residual_add};
use crate::ops_output_p128_h64_blocked::Output;
use furiosa_opt_std::prelude::*;

pub type InputBf16 = m![P512, H];
pub type HiddenBf16 = m![P512, L];
type NormCluster = m![P128 / 64];
type NormShards = m![P128 / 2 % 32, H / 480];
type NormLocal = m![Q4, P128 % 2, H % 480];
type CanonNormCluster = m![P512 / 64 % 2];
type CanonNormShards = m![P512 / 2 % 32, H / 480];
type CanonNormLocal = m![P512 / 128, P512 % 2, H % 480];
type Cluster = m![P512 / 256];
type Shards = m![L / 960, P512 % 256 = 64 / 4];

fn prefix_from_shards(
    device: &mut Device,
    input: &DmTensor<bf16, Chip, NormCluster, NormShards, NormLocal>,
    post: &HbmTensor<bf16, Chip, m![H]>,
    pre: &HbmTensor<bf16, Chip, m![H]>,
    residual: &mut HbmTensor<bf16, Chip, Residual>,
    output: &mut HbmTensor<bf16, Chip, InputBf16>,
) {
    let input = normalize(device, input, post);
    let previous = load(device, residual);
    let updated = residual_add(device, &input, &previous);
    let normalized = normalize(device, &updated, pre);
    // The original stage next calls quantize_shards. Export this BF16 value
    // directly instead; no E4 quantization/dequantization occurs in this path.
    const {
        assert!(Q4::SIZE == 4 && P128::SIZE == 128 && P512::SIZE == 512);
        assert!(H::SIZE == 3840 && Chip::SIZE == 1);
        assert!(NormCluster::SIZE == CanonNormCluster::SIZE);
        assert!(NormShards::SIZE == CanonNormShards::SIZE);
        assert!(NormLocal::SIZE == CanonNormLocal::SIZE);
    };
    // SAFETY: same dtype, chip and every per-memory-level physical extent.
    // At cluster c, slice 8*r+h, local ((2*q+j)*480+k), the old Q4/P128
    // coordinates and new P512 coordinates both identify
    // p=128*q+64*c+2*r+j, channel=480*h+k. No padding or order changes.
    let normalized: DmTensor<bf16, Chip, CanonNormCluster, CanonNormShards, CanonNormLocal> =
        unsafe { normalized.reshape() };
    // This real DMA joins the eight H480 shards and moves Q4/P2 into rows.
    // All 256 slices in each cluster hold one complete H3840 BF16 row.
    let rows: DmTensor<bf16, Chip, CanonNormCluster, m![P512 / 128, P512 % 64], m![H]> =
        normalized.to_dm(&mut device.tdma);
    rows.view().to_hbm_view(&mut device.tdma, output.view_mut());
    // Keep the original output-before-residual persistence order.
    store(device, updated, residual);
}

/// Original sliding post-attention/residual/pre-FFN RMS, before quantization.
/// `output` and `residual` must be distinct HBM owners.
#[device(chip = 1)]
pub fn prefix_sliding_bf16(
    device: &mut Device,
    a: &HbmTensor<bf16, Chip, Output>,
    b: &HbmTensor<bf16, Chip, Output>,
    c: &HbmTensor<bf16, Chip, Output>,
    d: &HbmTensor<bf16, Chip, Output>,
    post: &HbmTensor<bf16, Chip, m![H]>,
    pre: &HbmTensor<bf16, Chip, m![H]>,
    residual: &mut HbmTensor<bf16, Chip, Residual>,
    output: &mut HbmTensor<bf16, Chip, InputBf16>,
) {
    prefix_sliding_bf16_stage(device, a, b, c, d, post, pre, residual, output);
}

pub(crate) fn prefix_sliding_bf16_stage(
    device: &mut Device,
    a: &HbmTensor<bf16, Chip, Output>,
    b: &HbmTensor<bf16, Chip, Output>,
    c: &HbmTensor<bf16, Chip, Output>,
    d: &HbmTensor<bf16, Chip, Output>,
    post: &HbmTensor<bf16, Chip, m![H]>,
    pre: &HbmTensor<bf16, Chip, m![H]>,
    residual: &mut HbmTensor<bf16, Chip, Residual>,
    output: &mut HbmTensor<bf16, Chip, InputBf16>,
) {
    let mut input: DmTensor<
        bf16,
        Chip,
        NormCluster,
        m![P128/2%32,H/960,1#2],
        m![Q4, P128 % 2, H % 960],
    > = DmTensor::new();
    load_projected(device, a, 0, &mut input);
    load_projected(device, b, 1, &mut input);
    load_projected(device, c, 2, &mut input);
    load_projected(device, d, 3, &mut input);
    let input: DmTensor<bf16, Chip, NormCluster, NormShards, NormLocal> = input.to_dm(&mut device.tdma);
    prefix_from_shards(device, &input, post, pre, residual, output);
}

/// Original full-attention post-attention/residual/pre-FFN RMS, before quantization.
#[device(chip = 1)]
pub fn prefix_full_bf16(
    device: &mut Device,
    a: &HbmTensor<bf16, Chip, m![P128, H]>,
    b: &HbmTensor<bf16, Chip, m![P128, H]>,
    c: &HbmTensor<bf16, Chip, m![P128, H]>,
    d: &HbmTensor<bf16, Chip, m![P128, H]>,
    post: &HbmTensor<bf16, Chip, m![H]>,
    pre: &HbmTensor<bf16, Chip, m![H]>,
    residual: &mut HbmTensor<bf16, Chip, Residual>,
    output: &mut HbmTensor<bf16, Chip, InputBf16>,
) {
    let mut input: DmTensor<bf16, Chip, NormCluster, NormShards, NormLocal> = DmTensor::new();
    load_residual(device, a, 0, &mut input);
    load_residual(device, b, 1, &mut input);
    load_residual(device, c, 2, &mut input);
    load_residual(device, d, 3, &mut input);
    prefix_from_shards(device, &input, post, pre, residual, output);
}

/// Original UP/GATE global scaling and BF16 GeGLU, before hidden quantization.
/// Globals are the loader's existing reciprocal scales; do not invert again.
/// JoinedOutput must contain the pre-global BF16 projections, with the same
/// weight dequantization policy as the producer. No extra row scale is added.
#[device(chip = 1)]
pub fn geglu_bf16(
    device: &mut Device,
    joined: &HbmTensor<bf16, Chip, JoinedOutput>,
    up_global: &HbmTensor<f32, Chip, m![1]>,
    gate_global: &HbmTensor<f32, Chip, m![1]>,
    output: &mut HbmTensor<bf16, Chip, HiddenBf16>,
) {
    geglu_bf16_stage(device, joined, up_global, gate_global, output);
}

pub(crate) fn geglu_bf16_stage(
    device: &mut Device,
    joined: &HbmTensor<bf16, Chip, JoinedOutput>,
    up_global: &HbmTensor<f32, Chip, m![1]>,
    gate_global: &HbmTensor<f32, Chip, m![1]>,
    output: &mut HbmTensor<bf16, Chip, HiddenBf16>,
) {
    let up=joined.view().tile::<m![FfnProjection],1,m![FfnProjection=1#2,P512/256,L/240,P512%256,L%240]>(0);
    let gate=joined.view().tile::<m![FfnProjection],1,m![FfnProjection=1#2,P512/256,L/240,P512%256,L%240]>(1);
    let up_global = global_dm(device,up_global);
    let gate_global = global_dm(device,gate_global);
    for chunk in 0..4 {
        let up:DmTensor<bf16,Chip,Cluster,Shards,m![P512%256=64%4,L%960]>=up
            .tile::<m![P512%256],64,m![FfnProjection=1#2,P512/256,L/240,P512%256=64#256,L%240]>(64*chunk).to_dm(&mut device.tdma);
        let gate:DmTensor<bf16,Chip,Cluster,Shards,m![P512%256=64%4,L%960]>=gate
            .tile::<m![P512%256],64,m![FfnProjection=1#2,P512/256,L/240,P512%256=64#256,L%240]>(64*chunk).to_dm(&mut device.tdma);
        let mut hidden:DmTensor<bf16,Chip,Cluster,Shards,m![P512%256=64%4,L%960]>=DmTensor::new();
        for tile in 0..2 {
            geglu_tile_into(device,
                up.view().tile::<m![P512%256=64%4],2,m![P512%256=64%4=2#4,L%960]>(2*tile),
                gate.view().tile::<m![P512%256=64%4],2,m![P512%256=64%4=2#4,L%960]>(2*tile),&up_global,&gate_global,
                hidden.view_mut().tile::<m![P512%256=64%4],2,m![P512%256=64%4=2#{!}4,L%960]>(2*tile));
        }
        // Real same-cluster redistribution: 64 live row leaders, each with
        // all L15360 BF16 channels; 1#4 keeps 256 physical slice positions.
        // This padded allocation and all movement belong to the bridge cost.
        let rows: DmTensor<bf16, Chip, Cluster, m![P512%256=64,1#4], m![L]> =
            hidden.to_dm(&mut device.tdma);
        rows.view().to_hbm_view(&mut device.tdma, output.view_mut()
            .tile::<m![P512%256],64,m![P512/256,P512%256=64#{!}256,L]>(64*chunk));
    }
}
