//! Exact grouped post-FFN normalization with an independent quarter time axis.
//! Whole blocked HBM ingress is redistributed into the original H480 RMS shards.
use crate::ops_native_fp8_p512_downblocked::Output;
use crate::ops_norm_quarters::{Q4, load_residual, normalize, residual_add, store_residual};
use crate::{
    Chip,
    axes::{H, P128, P512},
};
use furiosa_opt_std::prelude::*;
type Cluster = m![P128 / 64];
type Shards = m![P128 / 2 % 32, H / 480];

pub(crate) fn gate(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Shards, m![Q4, P128 % 2, H % 480]>,
    scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
) -> DmTensor<bf16, Chip, Cluster, Shards, m![Q4, P128 % 2, H % 480]> {
    let scalar: DmTensor<bf16, Chip, Cluster, Shards, m![1 # 8]> = scalar.to_dm(&mut device.tdma);
    let scalar: VrfTensor<f32, Chip, Cluster, Shards, m![1 # 8]> = device
        .sub
        .begin(scalar.view())
        .fetch::<m![1], m![1 # 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    device
        .main
        .begin(x.view())
        .fetch::<m![Q4, P128 % 2, H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Q4, P128 % 2, H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Q4, P128 % 2, H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scalar)
        .vector_widen_concat::<m![Q4, P128 % 2, H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

#[device(chip = 1)]
pub fn post_feedforward_four(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, Output>,
    weight: &HbmTensor<bf16, Chip, m![H]>,
    layer_gate: &HbmTensor<bf16, Chip, m![1#8]>,
    a: &mut HbmTensor<bf16, Chip, m![P128, H]>,
    b: &mut HbmTensor<bf16, Chip, m![P128, H]>,
    c: &mut HbmTensor<bf16, Chip, m![P128, H]>,
    d: &mut HbmTensor<bf16, Chip, m![P128, H]>,
) {
    // Real named-axis redistribution happens before this factor-only relabel.
    // P512 = 128*Q4 + P128; cluster bit64 and row slice bits1..5 are unchanged.
    let input: DmTensor<
        bf16,
        Chip,
        m![P512 / 64 % 2],
        m![P512 / 2 % 32, H / 480],
        m![P512 / 128, P512 % 2, H % 480],
    > = input.to_dm(&mut device.tdma);
    let input: DmTensor<bf16, Chip, Cluster, Shards, m![Q4, P128 % 2, H % 480]> =
        unsafe { input.reshape() };
    let normalized = normalize(device, &input, weight);
    let mut previous: DmTensor<bf16, Chip, Cluster, Shards, m![Q4, P128 % 2, H % 480]> =
        DmTensor::new();
    load_residual(device, a, 0, &mut previous);
    load_residual(device, b, 1, &mut previous);
    load_residual(device, c, 2, &mut previous);
    load_residual(device, d, 3, &mut previous);
    let residual = residual_add(device, &normalized, &previous);
    // Preserve BF16 after normalized×weight, residual addition, and layer gate.
    let output = gate(device, &residual, layer_gate);
    store_residual(device, &output, 0, a);
    store_residual(device, &output, 1, b);
    store_residual(device, &output, 2, c);
    store_residual(device, &output, 3, d);
}
