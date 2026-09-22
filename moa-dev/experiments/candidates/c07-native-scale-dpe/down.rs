// Four columns x 64 row groups; all 512 physical slices hold 30 real rows.
pub(crate) type DownCluster = m![Dummy2];
pub(crate) type DownRows = m![H / 30 % 64, 1 # 4];
pub(crate) type DownRowsByColumns = m![H / 30 % 64, L / 3840];
axes![DownQuantK = 3840, DownQuantTile = 8, DownQuantPart = 480];

fn quantize_down(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, DownCluster, DownRowsByColumns, m![DownQuantK]>,
) -> (
    DmTensor<f8e4m3, Chip, DownCluster, DownRowsByColumns, m![FfnTerm, DownQuantK]>,
    DmTensor<f32, Chip, DownCluster, DownRowsByColumns, m![1 # 8]>,
) {
    let amax: DmTensor<f32, Chip, DownCluster, DownRowsByColumns, m![1 # 8]> = device.main
        .begin(x.view()).fetch::<m![DownQuantK / 16], m![DownQuantK % 16]>().fetch_cast::<f32>()
        .collect::<m![DownQuantK / 8], m![DownQuantK % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd, 0x7fff_ffffi32)
        .vector_reinterpret::<f32>()
        .vector_narrow_split::<m![DownQuantK / 4], m![DownQuantK % 4]>()
        .vector_intra_slice_reduce::<DownQuantK, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, 1.504632769052528e-36f32)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    let scale: DmTensor<f32, Chip, DownCluster, DownRowsByColumns, m![1 # 8]> = device.main
        .begin(amax.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd, 0x7f80_0000i32)
        .vector_reinterpret::<f32>().vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0078125f32)
        .vector_widen_pad::<m![1 # 8]>().vector_final().commit_trim::<m![1 # 8]>().commit();
    let scale_vrf: VrfTensor<f32, Chip, DownCluster, DownRowsByColumns, m![1 # 8]> = device.sub
        .begin(scale.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let hi: DmTensor<f8e4m3, Chip, DownCluster, DownRowsByColumns, m![DownQuantK]> = device.main
        .begin(x.view()).fetch::<m![DownQuantK / 16], m![DownQuantK % 16]>().fetch_cast::<f32>()
        .collect::<m![DownQuantK / 8], m![DownQuantK % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![DownQuantK / 4], m![DownQuantK % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_widen_concat::<m![DownQuantK / 8], m![DownQuantK % 8]>().vector_final()
        .cast::<f8e4m3, m![DownQuantK % 8 # 32]>().commit_trim::<m![DownQuantK % 8]>().commit();
    let mut lo: DmTensor<f8e4m3, Chip, DownCluster, DownRowsByColumns, m![DownQuantTile, DownQuantPart]> = DmTensor::new();
    let x_tiled: DmTensorView<'_, bf16, Chip, DownCluster, DownRowsByColumns, m![DownQuantTile, DownQuantPart]> = unsafe { x.view().reshape() };
    let hi_tiled: DmTensorView<'_, f8e4m3, Chip, DownCluster, DownRowsByColumns, m![DownQuantTile, DownQuantPart]> = unsafe { hi.view().reshape() };
    // 480-element residual tiles keep the F32 high-term VRF below 8 KiB.
    static_passes!(i, [0, 1, 2, 3, 4, 5, 6, 7], {
        let hi_vrf: VrfTensor<f32, Chip, DownCluster, DownRowsByColumns, m![DownQuantPart]> = device.sub
            .begin(hi_tiled.tile::<m![DownQuantTile], 1, m![DownQuantTile = 1 # 8, DownQuantPart]>(i))
            .fetch::<m![DownQuantPart / 32], m![DownQuantPart % 32]>().fetch_cast::<f32>()
            .collect::<m![DownQuantPart / 8], m![DownQuantPart % 8]>().to_vrf();
        device.main
            .begin(x_tiled.tile::<m![DownQuantTile], 1, m![DownQuantTile = 1 # 8, DownQuantPart]>(i))
            .fetch::<m![DownQuantPart / 16], m![DownQuantPart % 16]>().fetch_cast::<f32>()
            .collect::<m![DownQuantPart / 8], m![DownQuantPart % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![DownQuantPart / 4], m![DownQuantPart % 4]>()
            .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
            .vector_fp_binary(FpBinaryOp::SubF, &hi_vrf)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 16.0f32)
            .vector_widen_concat::<m![DownQuantPart / 8], m![DownQuantPart % 8]>().vector_final()
            .cast::<f8e4m3, m![DownQuantPart % 8 # 32]>().commit_trim::<m![DownQuantPart % 8]>()
            .commit_view(lo.view_mut().tile::<m![DownQuantTile], 1, m![DownQuantTile = 1 #{!} 8, DownQuantPart]>(i));
    });
    let lo: DmTensor<f8e4m3, Chip, DownCluster, DownRowsByColumns, m![DownQuantK]> = unsafe { lo.reshape() };
    let mut packed: DmTensor<f8e4m3, Chip, DownCluster, DownRowsByColumns, m![FfnTerm, DownQuantK]> = DmTensor::new();
    hi.view().to_dm_view(&mut device.tdma, packed.view_mut().tile::<m![FfnTerm], 1, m![FfnTerm = 1 #{!} 2, DownQuantK]>(0));
    lo.view().to_dm_view(&mut device.tdma, packed.view_mut().tile::<m![FfnTerm], 1, m![FfnTerm = 1 #{!} 2, DownQuantK]>(1));
    (packed, scale)
}

pub(crate) fn project_down(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, UpGateRowsPaired, m![L % 120]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    project_down_sharded(device, x, down_weight_packed, down_weight_scale).to_dm(&mut device.tdma)
}

pub(crate) fn project_down_sharded(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, UpGateRowsPaired, m![L % 120]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
) -> DmTensor<bf16, Chip, Cluster, DownEpilogueSlices, m![H % 480]> {
    let x: DmTensor<bf16, Chip, DownCluster, m![Dummy256 / 16, 1 # 4, L / 3840], m![L % 3840]> = x.to_dm(&mut device.tdma);
    let x: DmTensor<bf16, Chip, DownCluster, m![Dummy256 / 16, 4, L / 3840], m![L % 3840]> = device.main
        .begin(x.view()).fetch::<m![1], m![L % 3840]>()
        .switch::<m![Dummy256 / 16, 4, L / 3840], m![1]>(SwitchConfig::CustomBroadcast { ring_size: 16 })
        .collect::<m![L / 16 % 240], m![L % 16]>().commit_trim::<m![L % 16]>().commit();
    let x: DmTensor<bf16, Chip, DownCluster, DownRowsByColumns, m![DownQuantK]> = unsafe { x.reshape() };
    let (x, x_scale) = quantize_down(device, &x);
    let x: DmTensor<f8e4m3, Chip, DownCluster, DownRowsByColumns, m![FfnTerm, L % 3840]> = unsafe { x.reshape() };
    let x_trf: TrfTensor<f8e4m3, Chip, DownCluster, DownRowsByColumns, m![FfnTerm], m![L % 3840]> = device.sub
        .begin(x.view()).fetch::<m![FfnTerm], m![L % 3840]>()
        .collect::<m![FfnTerm, L / 32 % 120], m![L % 32]>().to_trf();
    let x_scale_vrf: VrfTensor<f32, Chip, DownCluster, DownRowsByColumns, m![1 # 8]> = device.sub
        .begin(x_scale.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let weights: HbmTensorView<'_, f4e2m1, Chip, m![Dummy2, H % 1920, L]> = unsafe { down_weight_packed.view().reshape() };
    let weights: DmTensor<f4e2m1, Chip, DownCluster, DownRowsByColumns, m![H % 30, L % 3840]> = weights.to_dm(&mut device.tdma);
    let scales: HbmTensorView<'_, f8e4m3, Chip, m![Dummy2, H % 1920, L / 16]> = unsafe { down_weight_scale.view().reshape() };
    let scales: DmTensor<f8e4m3, Chip, DownCluster, DownRowsByColumns, m![H % 30, L / 16 % 240]> = scales.to_dm(&mut device.tdma);
    let mut partial: DmTensor<f32, Chip, DownCluster, DownRowsByColumns, m![H % 30, 1 # 2]> = DmTensor::new();
    project_native_down_rows(device, &x_trf, &x_scale_vrf, &weights, &scales, &mut partial);
    let down: DmTensor<f32, Chip, DownCluster, DownRows, m![H % 30]> = device.main
        .begin(partial.view()).fetch::<m![H % 30], m![1 # 8]>().collect::<m![H % 30], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>().vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<DownRows, m![H % 30]>(InterSliceReduceOpF32::Add)
        .vector_final().transpose::<m![H / 2 % 15], m![H % 2 # 8]>().commit_trim::<m![H % 2]>().commit();
    let down: DmTensor<f32, Chip, m![H / 1920], DownRows, m![H % 30]> = unsafe { down.reshape() };
    let down: DmTensor<f32, Chip, Cluster, DownEpilogueSlices, m![H % 480]> = down.to_dm(&mut device.tdma);
    device.main.begin(down.view()).fetch::<m![H / 8 % 60], m![H % 8]>().collect::<m![H / 8 % 60], m![H % 8]>()
        .cast::<bf16, m![H % 8 # 16]>().commit_trim::<m![H % 8]>().commit()
}
