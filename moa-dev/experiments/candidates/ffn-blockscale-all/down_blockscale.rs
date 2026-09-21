// Preserve the local L%1920 column partition and its eight-slice FP32 sum.
fn project_down_four_rows_scaled_after_dot(
    device: &mut Device,
    x_trf: &TrfTensor<bf16, Chip, Cluster, DownRowsByColumns, m![1], m![L % 1920]>,
    weights: &DmTensor<f8e4m3, Chip, Cluster, DownRowsByColumns, m![H % 120, L % 1920]>,
    scales: &DmTensor<f8e4m3, Chip, Cluster, DownRowsByColumns, m![H % 120, L / 16 % 120]>,
    output: &mut DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120]>,
    row_offset: usize,
) {
    let scale_vrf: VrfTensor<f32, Chip, Cluster, DownRowsByColumns, m![H % 120 = 4, L / 16 % 120]> = device
        .sub
        .begin(scales.view().tile::<m![H % 120], 4, m![H % 120 = 4 # 120, L / 16 % 120]>(row_offset))
        .fetch::<m![H % 120 = 4], m![L / 16 % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 120 = 4, L / 16 / 8 % 15], m![L / 16 % 8]>()
        .to_vrf();

    device
        .main
        .begin(weights.view().tile::<m![H % 120], 4, m![H % 120 = 4 # 120, L % 1920]>(row_offset))
        .fetch::<m![H % 120 = 4, L / 32 % 60], m![L % 32]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![H % 120 = 4, L / 16 % 120], m![L % 16]>()
        .contract_outer::<m![H % 120 = 4, L / 16 % 120], m![L % 16], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 4, L / 16 % 120]>()
        .contract_lane::<m![H % 120 = 4, L / 16 % 120], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        // 120 local blocks sum in FP32, then the eight L-column slices sum in FP32.
        .vector_intra_slice_reduce::<L, m![H % 120 = 4], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<DownRows, m![H % 120 = 4]>(InterSliceReduceOpF32::Add)
        .vector_final()
        // Keep the original BF16 boundary AFTER the eight-slice FP32 sum.
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![1], m![H % 120 = 4 # 16]>()
        .commit_trim::<m![H % 120 = 4]>()
        .commit_view(output.view_mut().tile::<m![H % 120], 4, m![H % 120 = 4 #{!} 120]>(row_offset));
}

pub(crate) fn project_down(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, DownRows, m![L]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    let x: DmTensor<bf16, Chip, Cluster, DownRowsByColumns, m![L % 1920]> = x.to_dm(&mut device.tdma);
    let x_trf: TrfTensor<bf16, Chip, Cluster, DownRowsByColumns, m![1], m![L % 1920]> = device
        .sub
        .begin(x.view())
        .fetch::<m![L / 16 % 120], m![L % 16]>()
        .collect::<m![L / 16 % 120], m![L % 16]>()
        .to_trf();
    let mut down: DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120]> = DmTensor::new();
    let down_weight_scale: DmTensor<f8e4m3, Chip, Cluster, DownRowsByColumns, m![H % 120, L / 16 % 120]> =
        down_weight_scale.to_dm(&mut device.tdma);
    let down_weight_packed: DmTensor<f4e2m1, Chip, Cluster, DownRowsByColumns, m![H % 120, L % 1920]> =
        down_weight_packed.to_dm(&mut device.tdma);
    let down_weight_packed: DmTensor<f8e4m3, Chip, Cluster, DownRowsByColumns, m![H % 120, L % 1920]> = device
        .main
        .begin(down_weight_packed.view())
        .fetch::<m![H % 120], m![L % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![H % 120, L / 32 % 60], m![L % 32]>()
        .commit_trim::<m![L % 32]>()
        .commit();

    static_passes!(i, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], {
        project_down_four_rows_scaled_after_dot(device, &x_trf, &down_weight_packed, &down_weight_scale, &mut down, 4 * i);
    });
    down.to_dm(&mut device.tdma)
}
