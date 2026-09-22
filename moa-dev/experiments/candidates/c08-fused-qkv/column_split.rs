// Split each dot across eight H480 columns. Column sums remain F32 until
// channel scaling; the existing BF16 head normalization/RoPE follows.
type QueryColumnCluster = m![Qs / 2048];
type QueryColumnSlices = m![Qs / 64 % 32, H / 480];
type QueryReducedSlices = m![Qs / 64 % 32, 1 # 8];
type KvColumnCluster = m![Ps / 1024];
type KvColumnSlices = m![Ps / 32 % 32, H / 480];
type KvReducedSlices = m![Ps / 32 % 32, 1 # 8];

pub(crate) fn project_query_columns8(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, m![2], m![1 # 32, H / 480], m![H % 480]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
) -> DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Gs, Ds]> {
    let x: DmTensor<bf16, Chip, QueryColumnCluster, QueryColumnSlices, m![H % 480]> = x.to_dm(&mut device.tdma);
    let x_trf: TrfTensor<bf16, Chip, QueryColumnCluster, QueryColumnSlices, m![1], m![H % 480]> = device.sub
        .begin(x.view()).fetch::<m![H / 16 % 30], m![H % 16]>()
        .collect::<m![H / 16 % 30], m![H % 16]>().to_trf();
    let weight: DmTensor<f8e4m3, Chip, QueryColumnCluster, QueryColumnSlices, m![Qs % 64, H % 480]> = weight.to_dm(&mut device.tdma);
    let scales: DmTensor<bf16, Chip, QueryColumnCluster, QueryReducedSlices, m![Qs % 64]> = weight_scale.to_dm(&mut device.tdma);
    let scales: VrfTensor<f32, Chip, QueryColumnCluster, QueryReducedSlices, m![Qs % 64, 1 # 8]> = device.sub
        .begin(scales.view()).fetch::<m![Qs % 64], m![1 # 4]>().fetch_cast::<f32>()
        .collect::<m![Qs % 64], m![1 # 8]>().to_vrf();
    let output: DmTensor<bf16, Chip, QueryColumnCluster, QueryReducedSlices, m![Qs % 64]> = device.main
        .begin(weight.view()).fetch::<m![Qs % 64, H / 16 % 30], m![H % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Qs % 64, H / 16 % 30], m![H % 16]>()
        .contract_outer::<m![Qs % 64, H / 32 % 15], m![H % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs % 64]>()
        .contract_lane::<m![Qs % 64], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<QueryReducedSlices, m![Qs % 64]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero).vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scales)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Qs / 4 % 16], m![Qs % 4 # 16]>()
        .commit_trim::<m![Qs % 4]>().commit();
    let output: DmTensor<bf16, Chip, m![Ns / 4], m![Ns % 4, Gs, Ds / 64, 1 # 8], m![Ds % 64]> = unsafe { output.reshape() };
    output.to_dm(&mut device.tdma)
}

fn project_one_kv_columns8(
    device: &mut Device,
    x_trf: &TrfTensor<bf16, Chip, KvColumnCluster, KvColumnSlices, m![1], m![H % 480]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Ds]> {
    let weight: DmTensor<f8e4m3, Chip, KvColumnCluster, KvColumnSlices, m![Ps % 32, H % 480]> = weight.to_dm(&mut device.tdma);
    let scales: DmTensor<bf16, Chip, KvColumnCluster, KvReducedSlices, m![Ps % 32]> = weight_scale.to_dm(&mut device.tdma);
    let scales: VrfTensor<f32, Chip, KvColumnCluster, KvReducedSlices, m![Ps % 32, 1 # 8]> = device.sub
        .begin(scales.view()).fetch::<m![Ps % 32], m![1 # 4]>().fetch_cast::<f32>()
        .collect::<m![Ps % 32], m![1 # 8]>().to_vrf();
    let output: DmTensor<bf16, Chip, KvColumnCluster, KvReducedSlices, m![Ps % 32]> = device.main
        .begin(weight.view()).fetch::<m![Ps % 32, H / 16 % 30], m![H % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Ps % 32, H / 16 % 30], m![H % 16]>()
        .contract_outer::<m![Ps % 32, H / 32 % 15], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ps % 32]>()
        .contract_lane::<m![Ps % 32], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<KvReducedSlices, m![Ps % 32]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero).vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scales)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Ps / 4 % 8], m![Ps % 4 # 16]>()
        .commit_trim::<m![Ps % 4]>().commit();
    let output: DmTensor<bf16, Chip, m![Ns / 4], m![Ns % 4, Ds / 32, 1 # 8], m![Ds % 32]> = unsafe { output.reshape() };
    output.to_dm(&mut device.tdma)
}

pub(crate) fn project_key_value_columns8(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, m![2], m![1 # 32, H / 480], m![H % 480]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    v_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    k_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Ds]>,
    DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Ds]>,
) {
    let x: DmTensor<bf16, Chip, KvColumnCluster, KvColumnSlices, m![H % 480]> = x.to_dm(&mut device.tdma);
    let x_trf: TrfTensor<bf16, Chip, KvColumnCluster, KvColumnSlices, m![1], m![H % 480]> = device.sub
        .begin(x.view()).fetch::<m![H / 16 % 30], m![H % 16]>()
        .collect::<m![H / 16 % 30], m![H % 16]>().to_trf();
    let k = project_one_kv_columns8(device, &x_trf, k_weight, k_scale);
    let v = project_one_kv_columns8(device, &x_trf, v_weight, v_scale);
    (k, v)
}
