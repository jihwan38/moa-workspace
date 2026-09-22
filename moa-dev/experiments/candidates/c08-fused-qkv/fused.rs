// Four 2048-row matrices: Q first half, Q second half, K, V.
// Public HBM tensors and post-projection RMS/RoPE/cache interfaces are unchanged.
axes![QkvMatrix = 4, QkvRow = 2048];
type FusedCluster = m![QkvRow / 1024];
type FusedSlices = m![QkvRow / 4 % 256];

pub(crate) fn project_qkv_fused(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, m![2], Replicated, m![H]>,
    q_weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    v_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    q_scale: &HbmTensor<bf16, Chip, m![Qs]>,
    k_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Gs, Ds]>,
    DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Ds]>,
    DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Ds]>,
) {
    let mut weights: DmTensor<f8e4m3, Chip, FusedCluster, FusedSlices, m![QkvMatrix, QkvRow % 4, H]> = DmTensor::new();
    let q: HbmTensorView<'_, f8e4m3, Chip, m![QkvMatrix = 2, QkvRow, H]> = unsafe { q_weight.view().reshape() };
    q.to_dm_view(&mut device.tdma, weights.view_mut().tile::<m![QkvMatrix], 2, m![QkvMatrix = 2 #{!} 4, QkvRow % 4, H]>(0));
    let k: HbmTensorView<'_, f8e4m3, Chip, m![QkvMatrix = 1, QkvRow, H]> = unsafe { k_weight.view().reshape() };
    k.to_dm_view(&mut device.tdma, weights.view_mut().tile::<m![QkvMatrix], 1, m![QkvMatrix = 1 #{!} 4, QkvRow % 4, H]>(2));
    let v: HbmTensorView<'_, f8e4m3, Chip, m![QkvMatrix = 1, QkvRow, H]> = unsafe { v_weight.view().reshape() };
    v.to_dm_view(&mut device.tdma, weights.view_mut().tile::<m![QkvMatrix], 1, m![QkvMatrix = 1 #{!} 4, QkvRow % 4, H]>(3));

    let mut scales: DmTensor<bf16, Chip, FusedCluster, FusedSlices, m![QkvMatrix, QkvRow % 4]> = DmTensor::new();
    let q: HbmTensorView<'_, bf16, Chip, m![QkvMatrix = 2, QkvRow]> = unsafe { q_scale.view().reshape() };
    q.to_dm_view(&mut device.tdma, scales.view_mut().tile::<m![QkvMatrix], 2, m![QkvMatrix = 2 #{!} 4, QkvRow % 4]>(0));
    let k: HbmTensorView<'_, bf16, Chip, m![QkvMatrix = 1, QkvRow]> = unsafe { k_scale.view().reshape() };
    k.to_dm_view(&mut device.tdma, scales.view_mut().tile::<m![QkvMatrix], 1, m![QkvMatrix = 1 #{!} 4, QkvRow % 4]>(2));
    let v: HbmTensorView<'_, bf16, Chip, m![QkvMatrix = 1, QkvRow]> = unsafe { v_scale.view().reshape() };
    v.to_dm_view(&mut device.tdma, scales.view_mut().tile::<m![QkvMatrix], 1, m![QkvMatrix = 1 #{!} 4, QkvRow % 4]>(3));

    let x: DmTensorView<'_, bf16, Chip, FusedCluster, FusedSlices, m![H]> = unsafe { x.view().reshape() };
    let x_trf: TrfTensor<bf16, Chip, FusedCluster, FusedSlices, m![1], m![H]> = device.sub
        .begin(x).fetch::<m![H / 16], m![H % 16]>()
        .collect::<m![H / 16], m![H % 16]>().to_trf();
    let scales_vrf: VrfTensor<f32, Chip, FusedCluster, FusedSlices, m![QkvMatrix, QkvRow % 4, 1 # 8]> = device.sub
        .begin(scales.view()).fetch::<m![QkvMatrix, QkvRow % 4], m![1 # 4]>().fetch_cast::<f32>()
        .collect::<m![QkvMatrix, QkvRow % 4], m![1 # 8]>().to_vrf();
    let output: DmTensor<bf16, Chip, FusedCluster, FusedSlices, m![QkvMatrix, QkvRow % 4]> = device.main
        .begin(weights.view()).fetch::<m![QkvMatrix, QkvRow % 4, H / 16], m![H % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![QkvMatrix, QkvRow % 4, H / 16], m![H % 16]>()
        .contract_outer::<m![QkvMatrix, QkvRow % 4, H / 32], m![H % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![QkvMatrix, QkvRow % 4]>()
        .contract_lane::<m![QkvMatrix, QkvRow % 4], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scales_vrf)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![QkvMatrix], m![QkvRow % 4 # 16]>()
        .commit_trim::<m![QkvRow % 4]>().commit();

    let mut q: DmTensor<bf16, Chip, Cluster, m![1 # 32, QkvMatrix = 2, QkvRow / 512], m![QkvRow % 512]> = DmTensor::new();
    output.view().tile::<m![QkvMatrix], 2, m![QkvMatrix = 2 # 4, QkvRow % 4]>(0)
        .to_dm_view(&mut device.tdma, q.view_mut());
    let mut k: DmTensor<bf16, Chip, Cluster, m![1 # 32, QkvMatrix = 1, QkvRow / 256], m![QkvRow % 256]> = DmTensor::new();
    output.view().tile::<m![QkvMatrix], 1, m![QkvMatrix = 1 # 4, QkvRow % 4]>(2)
        .to_dm_view(&mut device.tdma, k.view_mut());
    let mut v: DmTensor<bf16, Chip, Cluster, m![1 # 32, QkvMatrix = 1, QkvRow / 256], m![QkvRow % 256]> = DmTensor::new();
    output.view().tile::<m![QkvMatrix], 1, m![QkvMatrix = 1 # 4, QkvRow % 4]>(3)
        .to_dm_view(&mut device.tdma, v.view_mut());
    (unsafe { q.reshape() }, unsafe { k.reshape() }, unsafe { v.reshape() })
}
