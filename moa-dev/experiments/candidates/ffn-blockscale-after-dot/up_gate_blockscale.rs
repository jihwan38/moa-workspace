// Local block scales apply to 16-element FP32 dot products. FP4*FP8 weight
// products are BF16-exact, but this changes FP32 reduction order; validate on NPU.
fn project_four_rows_scaled_after_dot(
    device: &mut Device,
    x_trf: &TrfTensor<bf16, Chip, Cluster, UpGateRows, m![1], m![H]>,
    weights: &DmTensor<f8e4m3, Chip, Cluster, UpGateRows, m![L % 60, H]>,
    scales: &DmTensor<f8e4m3, Chip, Cluster, UpGateRows, m![L % 60, H / 16]>,
    output: &mut DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]>,
    row_offset: usize,
) {
    let scale_vrf: VrfTensor<f32, Chip, Cluster, UpGateRows, m![L % 60 = 4, H / 16]> = device
        .sub
        .begin(scales.view().tile::<m![L % 60], 4, m![L % 60 = 4 # 60, H / 16]>(row_offset))
        .fetch::<m![L % 60 = 4], m![H / 16]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60 = 4, H / 16 / 8], m![H / 16 % 8]>()
        .to_vrf();

    device
        .main
        .begin(weights.view().tile::<m![L % 60], 4, m![L % 60 = 4 # 60, H]>(row_offset))
        .fetch::<m![L % 60 = 4, H / 32], m![H % 32]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![L % 60 = 4, H / 16], m![H % 16]>()
        // Keep each 16-element scale block separate through the DPE.
        .contract_outer::<m![L % 60 = 4, H / 16], m![H % 16], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![L % 60 = 4, H / 16]>()
        .contract_lane::<m![L % 60 = 4, H / 16], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_intra_slice_reduce::<H, m![L % 60 = 4], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![1], m![L % 60 = 4 # 16]>()
        .commit_trim::<m![L % 60 = 4]>()
        .commit_view(output.view_mut().tile::<m![L % 60], 4, m![L % 60 = 4 #{!} 60]>(row_offset));
}

pub(crate) fn project_up_and_gate(
    device: &mut Device,
    x_trf: &TrfTensor<bf16, Chip, Cluster, UpGateRows, m![1], m![H]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
) -> (
    DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]>,
    DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]>,
) {
    let mut up: DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]> = DmTensor::new();
    let up_weight_packed: DmTensor<f4e2m1, Chip, Cluster, UpGateRows, m![L % 60, H]> =
        up_weight_packed.to_dm(&mut device.tdma);
    let up_weight_packed: DmTensor<f8e4m3, Chip, Cluster, UpGateRows, m![L % 60, H]> = device
        .main
        .begin(up_weight_packed.view())
        .fetch::<m![L % 60], m![H]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![L % 60, H / 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit();
    let up_weight_scale: DmTensor<f8e4m3, Chip, Cluster, UpGateRows, m![L % 60, H / 16]> =
        up_weight_scale.to_dm(&mut device.tdma);

    static_passes!(i, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], {
        project_four_rows_scaled_after_dot(device, x_trf, &up_weight_packed, &up_weight_scale, &mut up, 4 * i);
    });

    let mut gate: DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]> = DmTensor::new();
    let gate_weight_packed: DmTensor<f4e2m1, Chip, Cluster, UpGateRows, m![L % 60, H]> =
        gate_weight_packed.to_dm(&mut device.tdma);
    let gate_weight_packed: DmTensor<f8e4m3, Chip, Cluster, UpGateRows, m![L % 60, H]> = device
        .main
        .begin(gate_weight_packed.view())
        .fetch::<m![L % 60], m![H]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![L % 60, H / 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit();
    let gate_weight_scale: DmTensor<f8e4m3, Chip, Cluster, UpGateRows, m![L % 60, H / 16]> =
        gate_weight_scale.to_dm(&mut device.tdma);

    static_passes!(i, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], {
        project_four_rows_scaled_after_dot(device, x_trf, &gate_weight_packed, &gate_weight_scale, &mut gate, 4 * i);
    });
    (up, gate)
}
