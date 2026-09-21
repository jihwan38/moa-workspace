use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Ds, Dummy2, Dummy256, Gs, H, Ns, Ps, Qs};
use crate::device::layout::{Cluster, Replicated, Slice};

pub(crate) fn project_query(
    device: &mut Device,
    x: &DmTensor<f8e4m3, Chip, m![2], Replicated, m![Dummy2, H]>,
    x_scale: &DmTensor<f32, Chip, m![2], Replicated, m![1 # 8]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
) -> DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Gs, Ds]> {
    type QueryCluster = m![Qs / 2048];
    type QueryRows = m![Qs / 64 % 32, Qs % 8];


    let x: DmTensorView<'_, f8e4m3, Chip, QueryCluster, QueryRows, m![Dummy2, H]> = unsafe { x.view().reshape() };
    let scale: DmTensorView<'_, f32, Chip, QueryCluster, QueryRows, m![1 # 8]> = unsafe { x_scale.view().reshape() };
    let x_trf: TrfTensor<f8e4m3, Chip, QueryCluster, QueryRows, m![Dummy2], m![H]> = device
        .sub.begin(x)
        .fetch::<m![Dummy2], m![H]>()
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .to_trf();
    let scale_vrf: VrfTensor<f32, Chip, QueryCluster, QueryRows, m![1 # 8]> = device
        .sub.begin(scale).fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>().to_vrf();
    let weight_f8: DmTensor<f8e4m3, Chip, QueryCluster, QueryRows, m![Qs / 8 % 8, H]> = weight.to_dm(&mut device.tdma);
    let contraction: DmTensor<bf16, Chip, QueryCluster, QueryRows, m![Qs / 8 % 8]> = device
        .main.begin(weight_f8.view())
        .fetch::<m![Qs / 8 % 8, H / 32], m![H % 32]>()
        .collect::<m![Qs / 8 % 8, H / 32], m![H % 32]>()
        .contract_outer::<m![Qs / 8 % 8, H / 64], m![H % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs / 8 % 8]>()
        .contract_lane::<m![Qs / 8 % 8, Dummy2], m![1 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_intra_slice_unzip::<Dummy2, m![Qs / 8 % 8]>()
        .vector_narrow_split::<m![Qs / 8 % 8, 1 # 2], m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), (), 0.0625f32)
        .vector_fp_zip(FpBinaryOp::AddF)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &scale_vrf)
        .vector_widen_concat::<m![Qs / 8 % 8], m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Qs / 32 % 2], m![Qs / 8 % 4 # 16]>()
        .commit_trim::<m![Qs / 8 % 4]>().commit();

    // Switch swaps the low slice coordinate with the local row coordinate.
    let contraction: DmTensor<bf16, Chip, QueryCluster, m![Qs / 8 % 256], m![Qs % 8]> = device
        .main.begin(contraction.view())
        .fetch::<m![Qs / 8 % 8], m![1 # 4]>()
        .switch::<m![Qs / 8 % 256], m![Qs % 8]>(SwitchConfig::InterTranspose { slice1: 8, slice0: 1, time0: 1 })
        .collect::<m![Qs % 8], m![1 # 16]>()
        .transpose::<m![Qs / 4 % 2], m![Qs % 4 # 16]>()
        .commit_trim::<m![Qs % 4]>()
        .commit();

    let weight_scale: DmTensor<bf16, Chip, QueryCluster, m![Qs / 8 % 256], m![Qs % 8]> = weight_scale.to_dm(&mut device.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, QueryCluster, m![Qs / 8 % 256], m![Qs % 8]> = device
        .sub
        .begin(weight_scale.view())
        .fetch::<m![1], m![Qs % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Qs % 8]>()
        .to_vrf();

    let scaled: DmTensor<bf16, Chip, QueryCluster, m![Qs / 8 % 256], m![Qs % 8]> = device
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

    let output: DmTensor<bf16, Chip, m![Ns / 4], m![Ns % 4, Gs, Ds / 8], m![Ds % 8]> =
        unsafe { scaled.reshape() };
    output.to_dm(&mut device.tdma)
}

type KvCluster = m![Ps / 1024];
type KvRows = m![Ps / 32 % 32, Ps % 8];

fn project_one_kv_matrix(
    device: &mut Device,
    x_trf: &TrfTensor<f8e4m3, Chip, KvCluster, KvRows, m![Dummy2], m![H]>,
    scale_vrf: &VrfTensor<f32, Chip, KvCluster, KvRows, m![1 # 8]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Ds]> {
    let weight_f8: DmTensor<f8e4m3, Chip, KvCluster, KvRows, m![Ps / 8 % 4, H]> = weight.to_dm(&mut device.tdma);
    let contraction: DmTensor<bf16, Chip, KvCluster, KvRows, m![Ps / 8 % 4]> = device
        .main.begin(weight_f8.view())
        .fetch::<m![Ps / 8 % 4, H / 32], m![H % 32]>()
        .collect::<m![Ps / 8 % 4, H / 32], m![H % 32]>()
        .contract_outer::<m![Ps / 8 % 4, H / 64], m![H % 64], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ps / 8 % 4]>()
        .contract_lane::<m![Ps / 8 % 4, Dummy2], m![1 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_intra_slice_unzip::<Dummy2, m![Ps / 8 % 4]>()
        .vector_narrow_split::<m![Ps / 8 % 4, 1 # 2], m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), (), 0.0625f32)
        .vector_fp_zip(FpBinaryOp::AddF)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), scale_vrf)
        .vector_widen_concat::<m![Ps / 8 % 4], m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![1], m![Ps / 8 % 4 # 16]>()
        .commit_trim::<m![Ps / 8 % 4]>().commit();

    // Four valid local rows are padded to an eight-way Switch group. Invalid
    // row slots stay padded in Slice until the final typed head gather.
    let contraction: DmTensor<bf16, Chip, KvCluster, m![Ps / 32 % 32, Ps / 8 % 4 # 8], m![Ps % 8]> = device
        .main.begin(contraction.view())
        .fetch::<m![Ps / 8 % 4 # 8], m![1 # 4]>()
        .switch::<m![Ps / 32 % 32, Ps / 8 % 4 # 8], m![Ps % 8]>(SwitchConfig::InterTranspose { slice1: 8, slice0: 1, time0: 1 })
        .collect::<m![Ps % 8], m![1 # 16]>()
        .transpose::<m![Ps / 4 % 2], m![Ps % 4 # 16]>()
        .commit_trim::<m![Ps % 4]>()
        .commit();

    let weight_scale: DmTensor<bf16, Chip, KvCluster, m![Ps / 32 % 32, Ps / 8 % 4 # 8], m![Ps % 8]> = weight_scale.to_dm(&mut device.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, KvCluster, m![Ps / 32 % 32, Ps / 8 % 4 # 8], m![Ps % 8]> = device
        .sub
        .begin(weight_scale.view())
        .fetch::<m![1], m![Ps % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Ps % 8]>()
        .to_vrf();

    let scaled: DmTensor<bf16, Chip, KvCluster, m![Ps / 32 % 32, Ps / 8 % 4 # 8], m![Ps % 8]> = device
        .main
        .begin(contraction.view())
        .fetch::<m![1], m![Ps % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Ps % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ps / 4 % 2], m![Ps % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale_vrf)
        .vector_widen_concat::<m![1], m![Ps % 8]>()
        .vector_final()
        .cast::<bf16, m![Ps % 8 # 16]>()
        .commit_trim::<m![Ps % 8]>()
        .commit();

    let output: DmTensor<bf16, Chip, m![Ns / 4], m![Ns % 4, Ds / 32, Ds / 8 % 4 # 8], m![Ds % 8]> =
        unsafe { scaled.reshape() };
    output.to_dm(&mut device.tdma)
}

fn project_value_two_dot(
    device: &mut Device,
    packed: DmTensorView<'_, f8e4m3, Chip, KvCluster, KvRows, m![Dummy2, H]>,
    scale_vrf: &VrfTensor<f32, Chip, KvCluster, KvRows, m![1 # 8]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Ds]> {
    // Select in-slice packed components; no physical-slice tiling or truncation.
    let hi_trf: TrfTensor<f8e4m3, Chip, KvCluster, KvRows, m![1], m![H]> = device
        .sub.begin(packed.tile::<m![Dummy2], 1, m![Dummy2 = 1 # 2, H]>(0))
        .fetch::<m![Dummy2 = 1], m![H]>()
        .collect::<m![H / 32], m![H % 32]>().to_trf();
    let lo_trf: TrfTensor<f8e4m3, Chip, KvCluster, KvRows, m![1], m![H]> = device
        .sub.begin(packed.tile::<m![Dummy2], 1, m![Dummy2 = 1 # 2, H]>(1))
        .fetch::<m![Dummy2 = 1], m![H]>()
        .collect::<m![H / 32], m![H % 32]>().to_trf();
    let weight_f8: DmTensor<f8e4m3, Chip, KvCluster, KvRows, m![Ps / 8 % 4, H]> = weight.to_dm(&mut device.tdma);
    let low_dot: DmTensor<f32, Chip, KvCluster, KvRows, m![Ps / 8 % 4, 1 # 8]> = device
        .main.begin(weight_f8.view())
        .fetch::<m![Ps / 8 % 4, H / 32], m![H % 32]>()
        .collect::<m![Ps / 8 % 4, H / 32], m![H % 32]>()
        .contract_outer::<m![Ps / 8 % 4, H / 64], m![H % 64], _, _, _>(&lo_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ps / 8 % 4]>()
        .contract_lane::<m![Ps / 8 % 4], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0625f32)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .commit_trim::<m![1 # 8]>().commit();
    let low_vrf: VrfTensor<f32, Chip, KvCluster, KvRows, m![Ps / 8 % 4, 1 # 8]> = device
        .sub.begin(low_dot.view())
        .fetch::<m![Ps / 8 % 4], m![1 # 8]>()
        .collect::<m![Ps / 8 % 4], m![1 # 8]>().to_vrf();
    let contraction: DmTensor<bf16, Chip, KvCluster, KvRows, m![Ps / 8 % 4]> = device
        .main.begin(weight_f8.view())
        .fetch::<m![Ps / 8 % 4, H / 32], m![H % 32]>()
        .collect::<m![Ps / 8 % 4, H / 32], m![H % 32]>()
        .contract_outer::<m![Ps / 8 % 4, H / 64], m![H % 64], _, _, _>(&hi_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ps / 8 % 4]>()
        .contract_lane::<m![Ps / 8 % 4], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, &low_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scale_vrf)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![1], m![Ps / 8 % 4 # 16]>()
        .commit_trim::<m![Ps / 8 % 4]>().commit();

    // Four valid local rows are padded to an eight-way Switch group. Invalid
    // row slots stay padded in Slice until the final typed head gather.
    let contraction: DmTensor<bf16, Chip, KvCluster, m![Ps / 32 % 32, Ps / 8 % 4 # 8], m![Ps % 8]> = device
        .main.begin(contraction.view())
        .fetch::<m![Ps / 8 % 4 # 8], m![1 # 4]>()
        .switch::<m![Ps / 32 % 32, Ps / 8 % 4 # 8], m![Ps % 8]>(SwitchConfig::InterTranspose { slice1: 8, slice0: 1, time0: 1 })
        .collect::<m![Ps % 8], m![1 # 16]>()
        .transpose::<m![Ps / 4 % 2], m![Ps % 4 # 16]>()
        .commit_trim::<m![Ps % 4]>()
        .commit();

    let weight_scale: DmTensor<bf16, Chip, KvCluster, m![Ps / 32 % 32, Ps / 8 % 4 # 8], m![Ps % 8]> = weight_scale.to_dm(&mut device.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, KvCluster, m![Ps / 32 % 32, Ps / 8 % 4 # 8], m![Ps % 8]> = device
        .sub
        .begin(weight_scale.view())
        .fetch::<m![1], m![Ps % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Ps % 8]>()
        .to_vrf();

    let scaled: DmTensor<bf16, Chip, KvCluster, m![Ps / 32 % 32, Ps / 8 % 4 # 8], m![Ps % 8]> = device
        .main
        .begin(contraction.view())
        .fetch::<m![1], m![Ps % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Ps % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ps / 4 % 2], m![Ps % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale_vrf)
        .vector_widen_concat::<m![1], m![Ps % 8]>()
        .vector_final()
        .cast::<bf16, m![Ps % 8 # 16]>()
        .commit_trim::<m![Ps % 8]>()
        .commit();

    let output: DmTensor<bf16, Chip, m![Ns / 4], m![Ns % 4, Ds / 32, Ds / 8 % 4 # 8], m![Ds % 8]> =
        unsafe { scaled.reshape() };
    output.to_dm(&mut device.tdma)
}

pub(crate) fn project_key_value(
    device: &mut Device,
    x: &DmTensor<f8e4m3, Chip, m![2], Replicated, m![Dummy2, H]>,
    x_scale: &DmTensor<f32, Chip, m![2], Replicated, m![1 # 8]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    v_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Ds]>,
    DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Ds]>,
) {
    let x_view: DmTensorView<'_, f8e4m3, Chip, KvCluster, KvRows, m![Dummy2, H]> = unsafe { x.view().reshape() };
    let scale: DmTensorView<'_, f32, Chip, KvCluster, KvRows, m![1 # 8]> = unsafe { x_scale.view().reshape() };
    let x_trf: TrfTensor<f8e4m3, Chip, KvCluster, KvRows, m![Dummy2], m![H]> = device
        .sub.begin(x_view)
        .fetch::<m![Dummy2], m![H]>()
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .to_trf();
    let scale_vrf: VrfTensor<f32, Chip, KvCluster, KvRows, m![1 # 8]> = device
        .sub.begin(scale).fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>().to_vrf();
    let k: DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Ds]> =
        project_one_kv_matrix(device, &x_trf, &scale_vrf, k_weight, k_weight_scale);
    let v: DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Ds]> =
        project_value_two_dot(device, unsafe { x.view().reshape() }, &scale_vrf, v_weight, v_weight_scale);

    (k, v)
}

pub(crate) fn project_output(
    device: &mut Device,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    weight_scale: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, OutputEpilogueSlices, m![H % 480]> {
    type OutputRows = m![H / 60 % 32, 1 # 8];
    let x: DmTensor<bf16, Chip, OutputCluster, OutputColumns, m![Qs % 512]> = x.to_dm(&mut device.tdma);
    let (x, input_scale) = prepare_output_fp8_split(device, &x);
    let x_trf: TrfTensor<f8e4m3, Chip, OutputCluster, OutputColumns, m![Dummy2], m![Qs % 512]> = device
        .sub.begin(x.view())
        .fetch::<m![Dummy2, Qs / 32 % 16], m![Qs % 32]>()
        .collect::<m![Dummy2, Qs / 32 % 16], m![Qs % 32]>()
        .to_trf();
    let input_scale_vrf: VrfTensor<f32, Chip, OutputCluster, OutputColumns, m![1 # 8]> = device
        .sub.begin(input_scale.view())
        .fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let weight: HbmTensorView<'_, f8e4m3, Chip, m![Dummy2, H % 1920, Qs]> = unsafe { weight.view().reshape() };
    let weight: DmTensor<f8e4m3, Chip, OutputCluster, OutputColumns, m![H % 60, Qs % 512]> = weight.to_dm(&mut device.tdma);
    let partial: DmTensor<f32, Chip, OutputCluster, OutputColumns, m![H % 60, 1 # 8]> = device
        .main.begin(weight.view())
        .fetch::<m![H % 60, Qs / 32 % 16], m![Qs % 32]>()
        .collect::<m![H % 60, Qs / 32 % 16], m![Qs % 32]>()
        .contract_outer::<m![H % 60, Qs / 64 % 8], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 60]>()
        .contract_lane::<m![H % 60, Dummy2], m![1 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_intra_slice_unzip::<Dummy2, m![H % 60]>()
        .vector_narrow_split::<m![H % 60, 1 # 2], m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), (), 0.0625f32)
        .vector_fp_zip(FpBinaryOp::AddF)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &input_scale_vrf)
        .vector_widen_concat::<m![H % 60], m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>().commit();
    // Pair sum and inter-slice sum share an AddF unit; retain F32 in DM.
    let result: DmTensor<bf16, Chip, OutputCluster, OutputRows, m![H % 60]> = device
        .main.begin(partial.view())
        .fetch::<m![H % 60], m![1 # 8]>()
        .collect::<m![H % 60], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<OutputRows, m![H % 60]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H / 4 % 15], m![H % 4 # 16]>()
        .commit_trim::<m![H % 4]>()
        .commit();
    // Rename only the existing high-row cluster axis; physical slice/padding
    // and all 60 local values stay in place. H=1920*cluster+60*group+local.
    let result: DmTensor<bf16, Chip, m![H / 1920], OutputRows, m![H % 60]> = unsafe { result.reshape() };
    let result: DmTensor<bf16, Chip, Cluster, OutputEpilogueSlices, m![H % 480]> = result.to_dm(&mut device.tdma);
    apply_output_channel_scale(device, &result, weight_scale)
}

pub(crate) type OutputEpilogueSlices = m![1 # 32, H / 480];

fn apply_output_channel_scale(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, OutputEpilogueSlices, m![H % 480]>,
    weight_scale: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, OutputEpilogueSlices, m![H % 480]> {
    let weight_scale: DmTensor<bf16, Chip, Cluster, OutputEpilogueSlices, m![H % 480]> = weight_scale.to_dm(&mut device.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, Cluster, OutputEpilogueSlices, m![H % 480]> = device
        .sub
        .begin(weight_scale.view())
        .fetch::<m![1], m![H % 480]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();

    device
        .main
        .begin(x.view())
        .fetch::<m![1], m![H % 480]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}


pub(crate) fn add_output_residual(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, OutputEpilogueSlices, m![H % 480]>,
    residual: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, OutputEpilogueSlices, m![H % 480]> {
    type ResidualSlices = m![1 # 32, H / 480];
    let residual: DmTensor<bf16, Chip, Cluster, ResidualSlices, m![H % 480]> = residual.to_dm(&mut device.tdma);
    let residual_vrf: VrfTensor<f32, Chip, Cluster, ResidualSlices, m![H % 480]> = device
        .sub
        .begin(residual.view())
        .fetch::<m![1], m![H % 480]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();
    let result: DmTensor<bf16, Chip, Cluster, ResidualSlices, m![H % 480]> = device
        .main
        .begin(x.view())
        .fetch::<m![1], m![H % 480]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &residual_vrf)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    result
}

// O shard-specific quantization: 512 inputs per column, no full-vector broadcast.
type OutputCluster = m![Dummy2];
type OutputColumns = m![H / 60 % 32, Qs / 512];
fn prepare_output_fp8_split(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, OutputCluster, OutputColumns, m![Qs % 512]>,
) -> (
    DmTensor<f8e4m3, Chip, OutputCluster, OutputColumns, m![Dummy2, Qs % 512]>,
    DmTensor<f32, Chip, OutputCluster, OutputColumns, m![1 # 8]>,
) {
    // Independent amax for each 512-value column shard; row replicas agree.
    // i32 reinterpret makes the absolute-value mask a non-NaN immediate.
    let amax: DmTensor<f32, Chip, OutputCluster, OutputColumns, m![1 # 8]> = device
        .main.begin(x.view())
        .fetch::<m![Qs / 16 % 32], m![Qs % 16]>().fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 64], m![Qs % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7fff_ffffi32)
        .vector_reinterpret::<f32>()
        .vector_narrow_split::<m![Qs / 4 % 128], m![Qs % 4]>()
        .vector_intra_slice_reduce::<Qs, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, 1.504632769052528e-36f32)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    // Exponent mask produces 2^floor(log2(amax)); floor 2^-119 guarantees
    // scale >= 2^-126, and finite BF16 inputs imply scale <= 2^120.
    let scale: DmTensor<f32, Chip, OutputCluster, OutputColumns, m![1 # 8]> = device
        .main.begin(amax.view())
        .fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7f80_0000i32)
        .vector_reinterpret::<f32>()
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0078125f32)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .commit_trim::<m![1 # 8]>().commit();
    let quant_scale: DmTensor<f32, Chip, OutputCluster, OutputColumns, m![1 # 8]> = scale.to_dm(&mut device.tdma);
    let scale_vrf: VrfTensor<f32, Chip, OutputCluster, OutputColumns, m![1 # 8]> = device
        .sub.begin(quant_scale.view())
        .fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let x: DmTensor<bf16, Chip, OutputCluster, OutputColumns, m![Qs % 512]> = x.to_dm(&mut device.tdma);
    let mut packed: DmTensor<f8e4m3, Chip, OutputCluster, OutputColumns, m![Dummy2, Qs % 512]> = DmTensor::new();
    device
        .main.begin(x.view())
        .fetch::<m![Qs / 16 % 32], m![Qs % 16]>().fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 64], m![Qs % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 128], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_widen_concat::<m![Qs / 8 % 64], m![Qs % 8]>().vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>()
        .commit_trim::<m![Qs % 8]>()
        .commit_view(packed.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Qs % 512]>(0));
    let hi_vrf: VrfTensor<f32, Chip, OutputCluster, OutputColumns, m![Qs % 512]> = device
        .sub.begin(packed.view().tile::<m![Dummy2], 1, m![Dummy2 = 1 # 2, Qs % 512]>(0))
        .fetch::<m![Qs / 32 % 16], m![Qs % 32]>().fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 64], m![Qs % 8]>().to_vrf();
    device
        .main.begin(x.view())
        .fetch::<m![Qs / 16 % 32], m![Qs % 16]>().fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 64], m![Qs % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 128], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &hi_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 16.0f32)
        .vector_widen_concat::<m![Qs / 8 % 64], m![Qs % 8]>().vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>()
        .commit_trim::<m![Qs % 8]>()
        .commit_view(packed.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Qs % 512]>(1));
    (packed, scale)
}

// Independently derived runtime power-of-two scaling. This is approximate for
// BF16 values below scale / 64 and changes FP32 contraction reduction order.
type QuantSlices = m![1 # 32, H / 480];

fn broadcast_quantized(
    device: &mut Device,
    x: &DmTensor<f8e4m3, Chip, m![2], QuantSlices, m![Dummy2, H % 480]>,
) -> DmTensor<f8e4m3, Chip, m![2], Replicated, m![Dummy2, H]> {
    let x: DmTensor<f8e4m3, Chip, m![2], m![Dummy256 / 32, 1 # 32], m![Dummy2, H]> = x.to_dm(&mut device.tdma);
    let x: DmTensor<f8e4m3, Chip, m![2], m![Dummy256 / 32, 32], m![Dummy2, H]> = device
        .main.begin(x.view())
        .fetch::<m![Dummy2], m![H]>()
        .switch::<m![Dummy256 / 32, 32], m![Dummy2]>(SwitchConfig::CustomBroadcast { ring_size: 32 })
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>().commit();
    // All 512 physical slices now contain the same original H-vector order.
    unsafe { x.reshape() }
}

pub(crate) fn prepare_fp8_split(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, m![2], QuantSlices, m![H % 480]>,
) -> (
    DmTensor<f8e4m3, Chip, m![2], Replicated, m![Dummy2, H]>,
    DmTensor<f32, Chip, m![2], Replicated, m![1 # 8]>,
) {
    // Preserve the whole-H global amax while keeping normalized input sharded.
    // Local floor commutes with max across the eight valid 480-value shards.
    let local_amax: DmTensor<f32, Chip, m![2], QuantSlices, m![1 # 8]> = device
        .main.begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7fff_ffffi32)
        .vector_reinterpret::<f32>()
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, 1.504632769052528e-36f32)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    // Output has exactly one logical scalar per cluster. No padded element
    // is reinterpreted as valid input; this typed reduction must compile.
    let amax: DmTensor<f32, Chip, m![2], Slice, m![1 # 8]> = device
        .main.begin(local_amax.view())
        .fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<Slice, m![1]>(InterSliceReduceOpF32::Max)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    // Exponent mask produces 2^floor(log2(amax)); floor 2^-119 guarantees
    // scale >= 2^-126, and finite BF16 inputs imply scale <= 2^120.
    let scale: DmTensor<f32, Chip, m![2], Slice, m![1 # 8]> = device
        .main.begin(amax.view())
        .fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7f80_0000i32)
        .vector_reinterpret::<f32>()
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0078125f32)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .commit_trim::<m![1 # 8]>().commit();
    let quant_scale: DmTensor<f32, Chip, m![2], QuantSlices, m![1 # 8]> = scale.to_dm(&mut device.tdma);
    let scale_vrf: VrfTensor<f32, Chip, m![2], QuantSlices, m![1 # 8]> = device
        .sub.begin(quant_scale.view())
        .fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let mut packed: DmTensor<f8e4m3, Chip, m![2], QuantSlices, m![Dummy2, H % 480]> = DmTensor::new();
    device
        .main.begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>().vector_final()
        .cast::<f8e4m3, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>()
        .commit_view(packed.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, H % 480]>(0));
    let hi_vrf: VrfTensor<f32, Chip, m![2], QuantSlices, m![H % 480]> = device
        .sub.begin(packed.view().tile::<m![Dummy2], 1, m![Dummy2 = 1 # 2, H % 480]>(0))
        .fetch::<m![H / 32 % 15], m![H % 32]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>().to_vrf();
    device
        .main.begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &hi_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 16.0f32)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>().vector_final()
        .cast::<f8e4m3, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>()
        .commit_view(packed.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, H % 480]>(1));
    (broadcast_quantized(device, &packed), scale.to_dm(&mut device.tdma))
}
