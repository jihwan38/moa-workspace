pub(crate) type DownCluster = m![Dummy2];
pub(crate) type DownRows = m![H / 60 % 32, 1 # 8];
pub(crate) type DownRowsByColumns = m![H / 60 % 32, L / 1920];

pub(crate) fn project_down(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, UpGateRowsPaired, m![L % 120]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    let x: DmTensor<bf16, Chip, DownCluster, DownRowsByColumns, m![L % 1920]> = x.to_dm(&mut device.tdma);
    let x_trf: TrfTensor<bf16, Chip, DownCluster, DownRowsByColumns, m![1], m![L % 1920]> = device
        .sub
        .begin(x.view())
        .fetch::<m![L / 16 % 120], m![L % 16]>()
        .collect::<m![L / 16 % 120], m![L % 16]>()
        .to_trf();

    let mut down: DmTensor<bf16, Chip, DownCluster, DownRows, m![H % 60]> = DmTensor::new();

    let down_weight_scale: HbmTensorView<'_, f8e4m3, Chip, m![Dummy2, H % 1920, L / 16]> = unsafe { down_weight_scale.view().reshape() };
    let down_weight_scale: DmTensor<f8e4m3, Chip, DownCluster, DownRowsByColumns, m![H % 60, L / 16 % 120]> =
        down_weight_scale.to_dm(&mut device.tdma);

    // The same full-matrix unpack pattern is used by the up/gate path.
    // Keep the column partition during both HBM loading and FP4 expansion.
    let down_weight_packed: HbmTensorView<'_, f4e2m1, Chip, m![Dummy2, H % 1920, L]> = unsafe { down_weight_packed.view().reshape() };
    let down_weight_packed: DmTensor<f4e2m1, Chip, DownCluster, DownRowsByColumns, m![H % 60, L % 1920]> =
        down_weight_packed.to_dm(&mut device.tdma);
    let down_weight_packed: DmTensor<f8e4m3, Chip, DownCluster, DownRowsByColumns, m![H % 60, L % 1920]> = device
        .main
        .begin(down_weight_packed.view())
        .fetch::<m![H % 60], m![L % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![H % 60, L / 32 % 60], m![L % 32]>()
        .commit_trim::<m![L % 32]>()
        .commit();

    static_passes!(i, [0, 1, 2, 3, 4], {
        let down_weight_scale_vrf: VrfTensor<f32, Chip, DownCluster, DownRowsByColumns, m![H % 60 = 12, L / 16 % 120]> = device
            .sub
            .begin(
                down_weight_scale
                    .view()
                    .tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L / 16 % 120]>(12 * i),
            )
            .fetch::<m![H % 60 = 12], m![L / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 12, L / 16 / 8 % 15], m![L / 16 % 8]>()
            .to_vrf();

        let down_weight_tile: DmTensor<bf16, Chip, DownCluster, DownRowsByColumns, m![H % 60 = 12, L % 1920]> = device
            .main
            .begin(
                down_weight_packed
                    .view()
                    .tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L % 1920]>(12 * i),
            )
            .fetch::<m![H % 60 = 12, L / 32 % 60], m![L % 32]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 12, L / 8 % 240], m![L % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![H % 60 = 12, L / 4 % 480], m![L % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_weight_scale_vrf)
            .vector_widen_concat::<m![H % 60 = 12, L / 8 % 240], m![L % 8]>()
            .vector_final()
            .cast::<bf16, m![L % 8 # 16]>()
            .commit_trim::<m![L % 8]>()
            .commit();

        // Canonical 12-row axis lets transpose emit three 4-row packets.
        let down_weight_tile: DmTensor<bf16, Chip, DownCluster, DownRowsByColumns, m![H % 12, L % 1920]> = unsafe { down_weight_tile.reshape() };
        // Preserve per-row reduction while committing three packets of four BF16 rows.
        device
            .main
            .begin(down_weight_tile.view())
            .fetch::<m![H % 12, L / 16 % 120], m![L % 16]>()
            .collect::<m![H % 12, L / 16 % 120], m![L % 16]>()
            .contract_outer::<m![H % 12, L / 32 % 60], m![L % 32], _, _, _>(&x_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![H % 12]>()
            .contract_lane::<m![H % 12], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<DownRows, m![H % 12]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![H / 4 % 3], m![H % 4 # 16]>()
            .commit_trim::<m![H % 4]>()
            .commit_view(down.view_mut().tile::<m![H / 12 % 5], 1, m![H / 12 % 5 = 1 #{!} 5, H % 12]>(i));
    });

    // Compact each cluster before joining its high row axis back into H.
    let down: DmTensor<bf16, Chip, DownCluster, Slice, m![H % 1920]> = down.to_dm(&mut device.tdma);
    let down: DmTensor<bf16, Chip, m![H / 1920], Slice, m![H % 1920]> = unsafe { down.reshape() };
    down.to_dm(&mut device.tdma)
}
