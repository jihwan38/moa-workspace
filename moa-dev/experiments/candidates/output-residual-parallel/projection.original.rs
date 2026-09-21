use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Ds, Dummy2, Gs, H, Ns, Ps, Qs};
use crate::device::layout::{Cluster, Replicated, Slice};

pub(crate) fn project_query(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, m![2], Replicated, m![H]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Gs, Ds]> {
    type QueryCluster = m![Qs / 2048];
    type QueryRows = m![Qs / 8 % 256];


    let x: DmTensorView<'_, bf16, Chip, QueryCluster, QueryRows, m![H]> = unsafe { x.view().reshape() };
    let x_trf: TrfTensor<bf16, Chip, QueryCluster, QueryRows, m![1], m![H]> = device
        .sub
        .begin(x)
        .fetch::<m![1], m![H]>()
        .collect::<m![H / 16], m![H % 16]>()
        .to_trf();

    let weight_f8: DmTensor<f8e4m3, Chip, QueryCluster, QueryRows, m![Qs % 8, H]> = weight.to_dm(&mut device.tdma);
    let contraction: DmTensor<bf16, Chip, QueryCluster, QueryRows, m![Qs % 8]> = device
        .main
        .begin(weight_f8.view())
        .fetch::<m![Qs % 8, H / 16], m![H % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Qs % 8, H / 16], m![H % 16]>()
        .contract_outer::<m![Qs % 8, H / 32], m![H % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs % 8]>()
        .contract_lane::<m![Qs % 8], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Qs / 4 % 2], m![Qs % 4 # 16]>()
        .commit_trim::<m![Qs % 4]>()
        .commit();

    let weight_scale: DmTensor<bf16, Chip, QueryCluster, QueryRows, m![Qs % 8]> = weight_scale.to_dm(&mut device.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, QueryCluster, QueryRows, m![Qs % 8]> = device
        .sub
        .begin(weight_scale.view())
        .fetch::<m![1], m![Qs % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Qs % 8]>()
        .to_vrf();

    let scaled: DmTensor<bf16, Chip, QueryCluster, QueryRows, m![Qs % 8]> = device
        .main
        .begin(contraction.view())
        .fetch::<m![1], m![Qs % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 2], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale_vrf)
        .vector_widen_concat::<m![1], m![Qs % 8]>()
        .vector_final()
        .cast::<bf16, m![Qs % 8 # 16]>()
        .commit_trim::<m![Qs % 8]>()
        .commit();

    let output: DmTensor<bf16, Chip, QueryCluster, Slice, m![Qs % 2048]> = device
        .main
        .begin(scaled.view())
        .fetch::<m![1], m![Qs % 8 # 16]>()
        .switch::<Slice, m![Qs / 8 % 256]>(SwitchConfig::Broadcast1 { slice1: 256, slice0: 1 })
        .collect::<m![Qs / 8 % 256], m![Qs % 8 # 16]>()
        .commit_trim::<m![Qs % 8]>()
        .commit();

    let output: DmTensor<bf16, Chip, Cluster, Slice, m![Qs]> = output.to_dm(&mut device.tdma);
    unsafe { output.reshape() }
}

type KvCluster = m![Ps / 1024];
type KvRows = m![Ps / 4 % 256];

fn project_one_kv_matrix(
    device: &mut Device,
    x_trf: &TrfTensor<bf16, Chip, KvCluster, KvRows, m![1], m![H]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ps]> {
    let weight_f8: DmTensor<f8e4m3, Chip, KvCluster, KvRows, m![Ps % 4, H]> = weight.to_dm(&mut device.tdma);
    let contraction: DmTensor<bf16, Chip, KvCluster, KvRows, m![Ps % 4]> = device
        .main
        .begin(weight_f8.view())
        .fetch::<m![Ps % 4, H / 16], m![H % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Ps % 4, H / 16], m![H % 16]>()
        .contract_outer::<m![Ps % 4, H / 32], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ps % 4]>()
        .contract_lane::<m![Ps % 4], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![1], m![Ps % 4 # 16]>()
        .commit_trim::<m![Ps % 4]>()
        .commit();

    let weight_scale: DmTensor<bf16, Chip, KvCluster, KvRows, m![Ps % 4]> = weight_scale.to_dm(&mut device.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, KvCluster, KvRows, m![Ps % 4 # 8]> = device
        .sub
        .begin(weight_scale.view())
        .fetch::<m![1], m![Ps % 4 # 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Ps % 4 # 8]>()
        .to_vrf();

    let scaled: DmTensor<bf16, Chip, KvCluster, KvRows, m![Ps % 4]> = device
        .main
        .begin(contraction.view())
        .fetch::<m![1], m![Ps % 4 # 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Ps % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![Ps % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale_vrf)
        .vector_widen_pad::<m![Ps % 4 # 8]>()
        .vector_final()
        .cast::<bf16, m![Ps % 4 # 16]>()
        .commit_trim::<m![Ps % 4]>()
        .commit();

    let output: DmTensor<bf16, Chip, KvCluster, Slice, m![Ps % 1024]> = device
        .main
        .begin(scaled.view())
        .fetch::<m![1], m![Ps % 4 # 16]>()
        .switch::<Slice, m![Ps / 4 % 256]>(SwitchConfig::Broadcast1 { slice1: 256, slice0: 1 })
        .collect::<m![Ps / 4 % 256], m![Ps % 4 # 16]>()
        .commit_trim::<m![Ps % 4]>()
        .commit();
    output.to_dm(&mut device.tdma)
}

pub(crate) fn project_key_value(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, m![2], Replicated, m![H]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    v_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]>,
    DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]>,
) {
    let x: DmTensorView<'_, bf16, Chip, KvCluster, KvRows, m![H]> = unsafe { x.view().reshape() };
    let x_trf: TrfTensor<bf16, Chip, KvCluster, KvRows, m![1], m![H]> = device
        .sub
        .begin(x)
        .fetch::<m![H / 16], m![H % 16]>()
        .collect::<m![H / 16], m![H % 16]>()
        .to_trf();

    let k: DmTensor<bf16, Chip, Cluster, Slice, m![Ps]> =
        project_one_kv_matrix(device, &x_trf, k_weight, k_weight_scale);
    let v: DmTensor<bf16, Chip, Cluster, Slice, m![Ps]> =
        project_one_kv_matrix(device, &x_trf, v_weight, v_weight_scale);

    (unsafe { k.reshape() }, unsafe { v.reshape() })
}

pub(crate) fn project_output(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, m![2], Replicated, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    weight_scale: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    type OutputCluster = m![Dummy2];
    type OutputRows = m![H / 8 % 240 # 256];
    let x: DmTensorView<'_, bf16, Chip, OutputCluster, OutputRows, m![Qs]> = unsafe { x.view().reshape() };
    let x_trf: TrfTensor<bf16, Chip, OutputCluster, OutputRows, m![1], m![Qs]> = device
        .sub
        .begin(x)
        .fetch::<m![Qs / 16], m![Qs % 16]>()
        .collect::<m![Qs / 16], m![Qs % 16]>()
        .to_trf();
    let weight: HbmTensorView<'_, f8e4m3, Chip, m![Dummy2, H % 1920, Qs]> = unsafe { weight.view().reshape() };
    let weight: DmTensor<f8e4m3, Chip, OutputCluster, OutputRows, m![H % 8, Qs]> = weight.to_dm(&mut device.tdma);
    let result: DmTensor<bf16, Chip, OutputCluster, OutputRows, m![H % 8]> = device
        .main
        .begin(weight.view())
        .fetch::<m![H % 8, Qs / 16], m![Qs % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![H % 8, Qs / 16], m![Qs % 16]>()
        .contract_outer::<m![H % 8, Qs / 32], m![Qs % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 8]>()
        .contract_lane::<m![H % 8], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H / 4 % 2], m![H % 4 # 16]>()
        .commit_trim::<m![H % 4]>()
        .commit();
    let result: DmTensor<bf16, Chip, OutputCluster, Slice, m![H / 8 % 240 # 256, H % 8]> = device
        .main
        .begin(result.view())
        .fetch::<m![1], m![H % 8 # 16]>()
        .switch::<Slice, m![H / 8 % 240 # 256]>(SwitchConfig::Broadcast1 { slice1: 256, slice0: 1 })
        .collect::<m![H / 8 % 240 # 256], m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    let result: DmTensor<bf16, Chip, OutputCluster, Slice, m![H % 1920]> = result.to_dm(&mut device.tdma);
    let result: DmTensor<bf16, Chip, m![H / 1920], Slice, m![H % 1920]> = unsafe { result.reshape() };
    let result: DmTensor<bf16, Chip, Cluster, HiddenRows, m![H % 120]> = result.to_dm(&mut device.tdma);
    let result = apply_output_channel_scale(device, &result, weight_scale);
    result.to_dm(&mut device.tdma)
}

type HiddenRows = m![H / 120, 1 # 8];

fn apply_output_channel_scale(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, HiddenRows, m![H % 120]>,
    weight_scale: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, HiddenRows, m![H % 120]> {
    let weight_scale: DmTensor<bf16, Chip, Cluster, HiddenRows, m![H % 120]> = weight_scale.to_dm(&mut device.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, Cluster, HiddenRows, m![H % 120]> = device
        .sub
        .begin(weight_scale.view())
        .fetch::<m![1], m![H % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .to_vrf();

    device
        .main
        .begin(x.view())
        .fetch::<m![1], m![H % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 30], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale_vrf)
        .vector_widen_concat::<m![H / 8 % 15], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}
