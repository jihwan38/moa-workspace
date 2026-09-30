use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Ds, Dummy2, Dummy256, Gs, H, Ns, Ps, Qs};
use crate::device::layout::{Cluster, Replicated};

axes![QkvTerm = 4, QkvGroup = 64, QkvQueryRow = 32, QkvKvRow = 16];
type QkvCluster = m![Dummy2];
type QkvColumns = m![QkvGroup, H / 960];
type QkvRows = m![QkvGroup, 1 # 4];
type QkvHeadSlices = m![1 # 32, Ns];

pub(crate) fn prepare_qkv_terms(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, m![2], QkvColumns, m![H % 960]>,
) -> (DmTensor<f8e4m3, Chip, QkvCluster, QkvColumns, m![QkvTerm, H % 960]>,
      DmTensor<f32, Chip, Cluster, QkvHeadSlices, m![1 # 8]>,
      DmTensor<f32, Chip, QkvCluster, QkvColumns, m![QkvTerm]>) {
    // Each (cluster,row-group) receives identical complete H3840 BF16 input.
    // Its four-column maximum is the same global amax in all128 groups.
    // Local floor commutes with max across the four valid 960-value shards.
    let local_amax: DmTensor<f32, Chip, m![2], QkvColumns, m![1 # 8]> = device
        .main.begin(x.view())
        .fetch::<m![H / 16 % 60], m![H % 16]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 120], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7fff_ffffi32)
        .vector_reinterpret::<f32>()
        .vector_narrow_split::<m![H / 4 % 240], m![H % 4]>()
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, 1.504632769052528e-36f32)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    // Preserve QkvGroup64 row-group ownership and reduce only the four K columns.
    // Exactly q=0 is initialized in each group; padded q slots stay invalid.
    let amax: DmTensor<f32, Chip, m![2], QkvRows, m![1 # 8]> = device
        .main.begin(local_amax.view())
        .fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<QkvRows, m![1]>(InterSliceReduceOpF32::Max)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    // Exponent mask produces 2^floor(log2(amax)); floor 2^-119 guarantees
    // scale >= 2^-126, and finite BF16 inputs imply scale <= 2^120.
    let scale: DmTensor<f32, Chip, m![2], QkvRows, m![1 # 8]> = device
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
    let quant_scale: DmTensor<f32, Chip, m![2], QkvColumns, m![1 # 8]> = scale.to_dm(&mut device.tdma);
    let scale_vrf: VrfTensor<f32, Chip, m![2], QkvColumns, m![1 # 8]> = device
        .sub.begin(quant_scale.view())
        .fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let mut packed: DmTensor<f8e4m3, Chip, m![2], QkvColumns, m![QkvTerm, H % 960]> = DmTensor::new();
    device.main.begin(x.view())
        .fetch::<m![H / 16 % 60], m![H % 16]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 120], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 240], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_widen_concat::<m![H / 8 % 120], m![H % 8]>().vector_final()
        .cast::<f8e4m3, m![H % 8 # 32]>().commit_trim::<m![H % 8]>()
        .commit_view(packed.view_mut().tile::<m![QkvTerm], 1, m![QkvTerm = 1 #{!} 4, H % 960]>(0));
    let sum_vrf: VrfTensor<f32, Chip, m![2], QkvColumns, m![H % 960]> = device.sub
        .begin(packed.view().tile::<m![QkvTerm], 1, m![QkvTerm = 1 # 4, H % 960]>(0))
        .fetch::<m![H / 32 % 30], m![H % 32]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 120], m![H % 8]>().to_vrf();
    // Term 1: residual is x/s minus the F32 cumulative reconstruction.
    device.main.begin(x.view())
        .fetch::<m![H / 16 % 60], m![H % 16]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 120], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 240], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &sum_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 16.0f32)
        .vector_widen_concat::<m![H / 8 % 120], m![H % 8]>().vector_final()
        .cast::<f8e4m3, m![H % 8 # 32]>().commit_trim::<m![H % 8]>()
        .commit_view(packed.view_mut().tile::<m![QkvTerm], 1, m![QkvTerm = 1 #{!} 4, H % 960]>(1));
    let sum: DmTensor<f32, Chip, m![2], QkvColumns, m![H % 960]> = device.main
        .begin(packed.view().tile::<m![QkvTerm], 1, m![QkvTerm = 1 # 4, H % 960]>(1))
        .fetch::<m![H / 32 % 30], m![H % 32]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 120], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 240], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0625f32)
        .vector_fp_binary(FpBinaryOp::AddF, &sum_vrf)
        .vector_widen_concat::<m![H / 8 % 120], m![H % 8]>().vector_final()
        .commit_trim::<m![H % 8]>().commit();
    let sum_vrf: VrfTensor<f32, Chip, m![2], QkvColumns, m![H % 960]> = device.sub
        .begin(sum.view()).fetch::<m![H / 8 % 120], m![H % 8]>()
        .collect::<m![H / 8 % 120], m![H % 8]>().to_vrf();
    // Term 2: residual is x/s minus the F32 cumulative reconstruction.
    device.main.begin(x.view())
        .fetch::<m![H / 16 % 60], m![H % 16]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 120], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 240], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &sum_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 4194304.0f32)
        .vector_widen_concat::<m![H / 8 % 120], m![H % 8]>().vector_final()
        .cast::<f8e4m3, m![H % 8 # 32]>().commit_trim::<m![H % 8]>()
        .commit_view(packed.view_mut().tile::<m![QkvTerm], 1, m![QkvTerm = 1 #{!} 4, H % 960]>(2));
    let sum: DmTensor<f32, Chip, m![2], QkvColumns, m![H % 960]> = device.main
        .begin(packed.view().tile::<m![QkvTerm], 1, m![QkvTerm = 1 # 4, H % 960]>(2))
        .fetch::<m![H / 32 % 30], m![H % 32]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 120], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 240], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0000002384185791015625f32)
        .vector_fp_binary(FpBinaryOp::AddF, &sum_vrf)
        .vector_widen_concat::<m![H / 8 % 120], m![H % 8]>().vector_final()
        .commit_trim::<m![H % 8]>().commit();
    let sum_vrf: VrfTensor<f32, Chip, m![2], QkvColumns, m![H % 960]> = device.sub
        .begin(sum.view()).fetch::<m![H / 8 % 120], m![H % 8]>()
        .collect::<m![H / 8 % 120], m![H % 8]>().to_vrf();
    // Term 3: residual is x/s minus the F32 cumulative reconstruction.
    device.main.begin(x.view())
        .fetch::<m![H / 16 % 60], m![H % 16]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 120], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 240], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &sum_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 134217728.0f32)
        .vector_widen_concat::<m![H / 8 % 120], m![H % 8]>().vector_final()
        .cast::<f8e4m3, m![H % 8 # 32]>().commit_trim::<m![H % 8]>()
        .commit_view(packed.view_mut().tile::<m![QkvTerm], 1, m![QkvTerm = 1 #{!} 4, H % 960]>(3));
    // Constant normalized coefficients: common s is deliberately restored after Term sum.
    // Zero*finite(scale)+constant builds exact bits without an extra public input.
    let mut coefficients: DmTensor<f32, Chip, m![2], QkvRows, m![QkvTerm, 1 # 8]> = DmTensor::new();
    device.main.begin(scale.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0f32)
        .vector_fp_binary(FpBinaryOp::AddF, 1.0f32)
        .vector_widen_pad::<m![1 # 8]>().vector_final().commit_trim::<m![1 # 8]>()
        .commit_view(coefficients.view_mut().tile::<m![QkvTerm], 1, m![QkvTerm = 1 #{!} 4, 1 # 8]>(0));
    device.main.begin(scale.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0f32)
        .vector_fp_binary(FpBinaryOp::AddF, 0.0625f32)
        .vector_widen_pad::<m![1 # 8]>().vector_final().commit_trim::<m![1 # 8]>()
        .commit_view(coefficients.view_mut().tile::<m![QkvTerm], 1, m![QkvTerm = 1 #{!} 4, 1 # 8]>(1));
    device.main.begin(scale.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0f32)
        .vector_fp_binary(FpBinaryOp::AddF, 0.0000002384185791015625f32)
        .vector_widen_pad::<m![1 # 8]>().vector_final().commit_trim::<m![1 # 8]>()
        .commit_view(coefficients.view_mut().tile::<m![QkvTerm], 1, m![QkvTerm = 1 #{!} 4, 1 # 8]>(2));
    device.main.begin(scale.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0f32)
        .vector_fp_binary(FpBinaryOp::AddF, 0.000000007450580596923828125f32)
        .vector_widen_pad::<m![1 # 8]>().vector_final().commit_trim::<m![1 # 8]>()
        .commit_view(coefficients.view_mut().tile::<m![QkvTerm], 1, m![QkvTerm = 1 #{!} 4, 1 # 8]>(3));
    // Spatial Term lanes need consecutive VRF cells. Pack the four scalar
    // time slots with two supported 8B commits, before replication.
    let coefficients: DmTensor<f32, Chip, m![2], QkvRows, m![QkvTerm]> = device.sub
        .begin(coefficients.view()).fetch::<m![QkvTerm], m![1 # 8]>()
        .collect::<m![QkvTerm], m![1 # 8]>()
        .transpose::<m![QkvTerm / 2], m![QkvTerm % 2 # 8]>()
        .commit_trim::<m![QkvTerm % 2]>().commit();
    // Input DMA gives each (c,g) the identical3840 finite BF16 values x[960q+u].
    // K4 Max/floor/exponentmask/pow2scale are deterministic, so all128 scalars
    // are equal. Relabel only the full allocation Hgroup axis as anonymous64;
    // keep physical256 slices and q=0 live mask unchanged. Never shrink to1#256.
    let shared_scale: DmTensor<f32, Chip, m![2], m![64, 1 # 4], m![1 # 8]> = unsafe { scale.reshape() };
    // Packed data is already in the consuming row/K slices. The cluster relabel
    // m![2] -> Dummy2 changes no physical position or byte, and makes no DMA.
    let packed: DmTensor<f8e4m3, Chip, QkvCluster, QkvColumns, m![QkvTerm, H % 960]> = unsafe { packed.reshape() };
    (packed, shared_scale.to_dm(&mut device.tdma), coefficients.to_dm(&mut device.tdma))
}

// One quantizer and one activation TRF serve Q, K and V with identical owners.
pub(crate) fn project_qkv_native(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, m![2], m![1 # 32, H / 480], m![H % 480]>,
    q_weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    v_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    q_weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, Cluster, QkvHeadSlices, m![Gs, Ds]>,
    DmTensor<bf16, Chip, Cluster, QkvHeadSlices, m![Ds]>,
    DmTensor<bf16, Chip, Cluster, QkvHeadSlices, m![Ds]>,
) {
    // Seed four K columns in each physical32-slice ring, then copy only
    // between equivalent output row groups. Each (c,g) still sees H3840.
    let x: DmTensor<bf16, Chip, m![2], m![Dummy256 / 32, 1 # 8, H / 960], m![H % 960]> = x.to_dm(&mut device.tdma);
    let x: DmTensor<bf16, Chip, m![2], m![Dummy256 / 32, 8, H / 960], m![H % 960]> = device.main
        .begin(x.view()).fetch::<m![1], m![H % 960]>()
        .switch::<m![Dummy256 / 32, 8, H / 960], m![1]>(SwitchConfig::CustomBroadcast { ring_size: 32 })
        .collect::<m![H / 16 % 60], m![H % 16]>()
        .commit_trim::<m![H % 16]>().commit();
    // Physical slice32a+4b+q maps to4g+q, g=8a+b. Full allocation
    // relabel changes no wire order, valid mask, K shard or BF16 value.
    let x: DmTensor<bf16, Chip, m![2], QkvColumns, m![H % 960]> = unsafe { x.reshape() };
    let (packed, scale, coefficients) = prepare_qkv_terms(device, &x);
    let x_trf: TrfTensor<f8e4m3, Chip, QkvCluster, QkvColumns, m![QkvTerm], m![H % 960]> = device.sub
        .begin(packed.view()).fetch::<m![QkvTerm, H / 32 % 30], m![H % 32]>()
        .collect::<m![QkvTerm, H / 32 % 30], m![H % 32]>().to_trf();
    let coefficients: VrfTensor<f32, Chip, QkvCluster, QkvColumns, m![QkvTerm # 8]> = device.sub
        .begin(coefficients.view()).fetch::<m![1], m![QkvTerm]>()
        .collect::<m![1], m![QkvTerm # 8]>().to_vrf();
    let q = project_native_query(device, &x_trf, &coefficients, &scale, q_weight, q_weight_scale);
    let k = project_native_key(device, &x_trf, &coefficients, &scale, k_weight, k_weight_scale);
    let v = project_native_value(device, &x_trf, &coefficients, &scale, v_weight, v_weight_scale);
    (q, k, v)
}

fn project_native_query(
    device: &mut Device,
    x: &TrfTensor<f8e4m3, Chip, QkvCluster, QkvColumns, m![QkvTerm], m![H % 960]>,
    coefficients: &VrfTensor<f32, Chip, QkvCluster, QkvColumns, m![QkvTerm # 8]>,
    scale: &DmTensor<f32, Chip, Cluster, QkvHeadSlices, m![1 # 8]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
) -> DmTensor<bf16, Chip, Cluster, QkvHeadSlices, m![Gs, Ds]> {
    // Full compact source wire order: row=2048c+32g+r, h=960q+u.
    let weight: HbmTensorView<'_, f8e4m3, Chip, m![Dummy2, QkvGroup, QkvQueryRow, H]> = unsafe { weight.view().reshape() };
    // Two independent full-H ingress allocations expose first-half consumption.
    // Each source slice retains row low2 ownership; no packed copies or extra output assembly.
    let first: DmTensor<f8e4m3, Chip, QkvCluster, m![QkvGroup, QkvQueryRow = 16 % 4], m![QkvQueryRow = 16 / 4, H]> =
        weight.tile::<m![QkvQueryRow], 16, m![Dummy2, QkvGroup, QkvQueryRow = 16 # 32, H]>(0).to_dm(&mut device.tdma);
    let second: DmTensor<f8e4m3, Chip, QkvCluster, m![QkvGroup, QkvQueryRow = 16 % 4], m![QkvQueryRow = 16 / 4, H]> =
        weight.tile::<m![QkvQueryRow], 16, m![Dummy2, QkvGroup, QkvQueryRow = 16 # 32, H]>(16).to_dm(&mut device.tdma);
    let mut raw: DmTensor<f32, Chip, QkvCluster, QkvRows, m![QkvQueryRow]> = DmTensor::new();
    project_native_query_stage(device, x, coefficients, &first, &mut raw, 0);
    project_native_query_stage(device, x, coefficients, &second, &mut raw, 16);
    // Full-allocation relabel preserves the q=0 live mask; typed DMA merges
    // canonical heads from both row-owning clusters into original head layout.
    let raw: DmTensor<f32, Chip, m![Ns / 4], m![Ns % 4, Gs, Ds / 32, 1 # 4], m![Ds % 32]> = unsafe { raw.reshape() };
    let raw: DmTensor<f32, Chip, Cluster, QkvHeadSlices, m![Gs, Ds]> = raw.to_dm(&mut device.tdma);
    let scale_vrf: VrfTensor<f32, Chip, Cluster, QkvHeadSlices, m![1 # 8]> = device.sub
        .begin(scale.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let weight_scale: HbmTensorView<'_, bf16, Chip, m![Ns, Gs, Ds]> = unsafe { weight_scale.view().reshape() };
    let weight_scale: DmTensor<bf16, Chip, Cluster, QkvHeadSlices, m![Gs, Ds]> = weight_scale.to_dm(&mut device.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, Cluster, QkvHeadSlices, m![Gs, Ds]> = device.sub
        .begin(weight_scale.view()).fetch::<m![1], m![Gs, Ds]>().fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>().to_vrf();
    device.main.begin(raw.view())
        .fetch::<m![Gs, Ds / 8], m![Ds % 8]>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &weight_scale_vrf)
        .vector_widen_concat::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_final().cast::<bf16, m![Ds % 8 # 16]>().commit_trim::<m![Ds % 8]>().commit()
}

fn project_native_query_stage(
    device: &mut Device,
    x: &TrfTensor<f8e4m3, Chip, QkvCluster, QkvColumns, m![QkvTerm], m![H % 960]>,
    coefficients: &VrfTensor<f32, Chip, QkvCluster, QkvColumns, m![QkvTerm # 8]>,
    weight: &DmTensor<f8e4m3, Chip, QkvCluster, m![QkvGroup, QkvQueryRow = 16 % 4], m![QkvQueryRow = 16 / 4, H]>,
    output: &mut DmTensor<f32, Chip, QkvCluster, QkvRows, m![QkvQueryRow]>,
    offset: usize,
) {
    device.main
        .begin(weight.view()).fetch::<m![H / 960, QkvQueryRow = 16 / 4], m![H % 960]>()
        .switch::<QkvColumns, m![QkvQueryRow = 16]>(SwitchConfig::InterTranspose { slice1: 4, slice0: 1, time0: 4 })
        .collect::<m![QkvQueryRow = 16, H / 32 % 30], m![H % 32]>()
        .contract_outer::<m![QkvQueryRow = 16, H / 64 % 15], m![H % 64], _, _, _>(x)
        .contract_packet::<m![1]>().contract_time::<m![QkvQueryRow = 16]>()
        .contract_lane::<m![QkvQueryRow = 16], m![QkvTerm # 8]>(LaneMode::Interleaved)
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![QkvTerm]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), coefficients)
        .vector_intra_slice_reduce::<QkvTerm, m![QkvQueryRow = 16], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<QkvRows, m![QkvQueryRow = 16]>(InterSliceReduceOpF32::Add)
        .vector_final().transpose::<m![QkvQueryRow = 16 / 2], m![QkvQueryRow = 16 % 2 # 8]>()
        .commit_trim::<m![QkvQueryRow = 16 % 2]>().commit_view(output.view_mut().tile::<m![QkvQueryRow], 16, m![QkvQueryRow = 16 #{!} 32]>(offset));
}

fn project_native_key(
    device: &mut Device,
    x: &TrfTensor<f8e4m3, Chip, QkvCluster, QkvColumns, m![QkvTerm], m![H % 960]>,
    coefficients: &VrfTensor<f32, Chip, QkvCluster, QkvColumns, m![QkvTerm # 8]>,
    scale: &DmTensor<f32, Chip, Cluster, QkvHeadSlices, m![1 # 8]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> DmTensor<bf16, Chip, Cluster, QkvHeadSlices, m![Ds]> {
    // Full compact source wire order: row=1024c+16g+r, h=960q+u.
    let weight: HbmTensorView<'_, f8e4m3, Chip, m![Dummy2, QkvGroup, QkvKvRow, H]> = unsafe { weight.view().reshape() };
    let first: DmTensor<f8e4m3, Chip, QkvCluster, m![QkvGroup, QkvKvRow = 8 % 4], m![QkvKvRow = 8 / 4, H]> =
        weight.tile::<m![QkvKvRow], 8, m![Dummy2, QkvGroup, QkvKvRow = 8 # 16, H]>(0).to_dm(&mut device.tdma);
    let second: DmTensor<f8e4m3, Chip, QkvCluster, m![QkvGroup, QkvKvRow = 8 % 4], m![QkvKvRow = 8 / 4, H]> =
        weight.tile::<m![QkvKvRow], 8, m![Dummy2, QkvGroup, QkvKvRow = 8 # 16, H]>(8).to_dm(&mut device.tdma);
    let mut raw: DmTensor<f32, Chip, QkvCluster, QkvRows, m![QkvKvRow]> = DmTensor::new();
    project_native_kv_stage(device, x, coefficients, &first, &mut raw, 0);
    project_native_kv_stage(device, x, coefficients, &second, &mut raw, 8);
    // Full-allocation relabel preserves the q=0 live mask; typed DMA merges
    // canonical heads from both row-owning clusters into original head layout.
    let raw: DmTensor<f32, Chip, m![Ns / 4], m![Ns % 4, Ds / 16, 1 # 4], m![Ds % 16]> = unsafe { raw.reshape() };
    let raw: DmTensor<f32, Chip, Cluster, QkvHeadSlices, m![Ds]> = raw.to_dm(&mut device.tdma);
    let scale_vrf: VrfTensor<f32, Chip, Cluster, QkvHeadSlices, m![1 # 8]> = device.sub
        .begin(scale.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let weight_scale: HbmTensorView<'_, bf16, Chip, m![Ns, Ds]> = unsafe { weight_scale.view().reshape() };
    let weight_scale: DmTensor<bf16, Chip, Cluster, QkvHeadSlices, m![Ds]> = weight_scale.to_dm(&mut device.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, Cluster, QkvHeadSlices, m![Ds]> = device.sub
        .begin(weight_scale.view()).fetch::<m![1], m![Ds]>().fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>().to_vrf();
    device.main.begin(raw.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &weight_scale_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final().cast::<bf16, m![Ds % 8 # 16]>().commit_trim::<m![Ds % 8]>().commit()
}

fn project_native_value(
    device: &mut Device,
    x: &TrfTensor<f8e4m3, Chip, QkvCluster, QkvColumns, m![QkvTerm], m![H % 960]>,
    coefficients: &VrfTensor<f32, Chip, QkvCluster, QkvColumns, m![QkvTerm # 8]>,
    scale: &DmTensor<f32, Chip, Cluster, QkvHeadSlices, m![1 # 8]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> DmTensor<bf16, Chip, Cluster, QkvHeadSlices, m![Ds]> {
    // Full compact source wire order: row=1024c+16g+r, h=960q+u.
    let weight: HbmTensorView<'_, f8e4m3, Chip, m![Dummy2, QkvGroup, QkvKvRow, H]> = unsafe { weight.view().reshape() };
    let weight: DmTensor<f8e4m3, Chip, QkvCluster, m![QkvGroup, QkvKvRow % 4], m![QkvKvRow / 4, H]> = weight.to_dm(&mut device.tdma);
    let raw: DmTensor<f32, Chip, QkvCluster, QkvRows, m![QkvKvRow]> = device.main
        .begin(weight.view()).fetch::<m![H / 960, QkvKvRow / 4], m![H % 960]>()
        .switch::<QkvColumns, m![QkvKvRow]>(SwitchConfig::InterTranspose { slice1: 4, slice0: 1, time0: 4 })
        .collect::<m![QkvKvRow, H / 32 % 30], m![H % 32]>()
        .contract_outer::<m![QkvKvRow, H / 64 % 15], m![H % 64], _, _, _>(x)
        .contract_packet::<m![1]>().contract_time::<m![QkvKvRow]>()
        .contract_lane::<m![QkvKvRow], m![QkvTerm # 8]>(LaneMode::Interleaved)
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![QkvTerm]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), coefficients)
        .vector_intra_slice_reduce::<QkvTerm, m![QkvKvRow], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<QkvRows, m![QkvKvRow]>(InterSliceReduceOpF32::Add)
        .vector_final().transpose::<m![QkvKvRow / 2], m![QkvKvRow % 2 # 8]>()
        .commit_trim::<m![QkvKvRow % 2]>().commit();
    // Full-allocation relabel preserves the q=0 live mask; typed DMA merges
    // canonical heads from both row-owning clusters into original head layout.
    let raw: DmTensor<f32, Chip, m![Ns / 4], m![Ns % 4, Ds / 16, 1 # 4], m![Ds % 16]> = unsafe { raw.reshape() };
    let raw: DmTensor<f32, Chip, Cluster, QkvHeadSlices, m![Ds]> = raw.to_dm(&mut device.tdma);
    let scale_vrf: VrfTensor<f32, Chip, Cluster, QkvHeadSlices, m![1 # 8]> = device.sub
        .begin(scale.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let weight_scale: HbmTensorView<'_, bf16, Chip, m![Ns, Ds]> = unsafe { weight_scale.view().reshape() };
    let weight_scale: DmTensor<bf16, Chip, Cluster, QkvHeadSlices, m![Ds]> = weight_scale.to_dm(&mut device.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, Cluster, QkvHeadSlices, m![Ds]> = device.sub
        .begin(weight_scale.view()).fetch::<m![1], m![Ds]>().fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>().to_vrf();
    device.main.begin(raw.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &weight_scale_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final().cast::<bf16, m![Ds % 8 # 16]>().commit_trim::<m![Ds % 8]>().commit()
}

fn project_native_kv_stage(
    device: &mut Device,
    x: &TrfTensor<f8e4m3, Chip, QkvCluster, QkvColumns, m![QkvTerm], m![H % 960]>,
    coefficients: &VrfTensor<f32, Chip, QkvCluster, QkvColumns, m![QkvTerm # 8]>,
    weight: &DmTensor<f8e4m3, Chip, QkvCluster, m![QkvGroup, QkvKvRow = 8 % 4], m![QkvKvRow = 8 / 4, H]>,
    output: &mut DmTensor<f32, Chip, QkvCluster, QkvRows, m![QkvKvRow]>,
    offset: usize,
) {
    device.main
        .begin(weight.view()).fetch::<m![H / 960, QkvKvRow = 8 / 4], m![H % 960]>()
        .switch::<QkvColumns, m![QkvKvRow = 8]>(SwitchConfig::InterTranspose { slice1: 4, slice0: 1, time0: 2 })
        .collect::<m![QkvKvRow = 8, H / 32 % 30], m![H % 32]>()
        .contract_outer::<m![QkvKvRow = 8, H / 64 % 15], m![H % 64], _, _, _>(x)
        .contract_packet::<m![1]>().contract_time::<m![QkvKvRow = 8]>()
        .contract_lane::<m![QkvKvRow = 8], m![QkvTerm # 8]>(LaneMode::Interleaved)
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![QkvTerm]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), coefficients)
        .vector_intra_slice_reduce::<QkvTerm, m![QkvKvRow = 8], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<QkvRows, m![QkvKvRow = 8]>(InterSliceReduceOpF32::Add)
        .vector_final().transpose::<m![QkvKvRow = 8 / 2], m![QkvKvRow = 8 % 2 # 8]>()
        .commit_trim::<m![QkvKvRow = 8 % 2]>().commit_view(output.view_mut().tile::<m![QkvKvRow], 8, m![QkvKvRow = 8 #{!} 16]>(offset));
}

// E137 uses E134 QKV unchanged; O changes activation representation and native reduction.
use crate::device::layout::Slice;
axes![OutputTerm = 4];
type OutputQuantSlices = m![H / 120 % 16, Qs / 256];
type OutputQuantRows = m![H / 120 % 16, 1 # 16];

// Every consumer row group holds the same sixteen K256 input shards.
// Quantize locally: no packed-term redistribution between row groups.
pub(crate) fn prepare_output_terms(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, m![2], OutputQuantSlices, m![Qs % 256]>,
) -> (DmTensor<f8e4m3, Chip, OutputCluster, OutputColumns, m![OutputTerm, Qs % 256]>,
      DmTensor<f32, Chip, Cluster, OutputEpilogueSlices, m![1 # 8]>,
      DmTensor<f32, Chip, OutputCluster, OutputColumns, m![OutputTerm]>) {
    // Each (cluster,row-group) receives identical complete K4096 BF16 input.
    // Its sixteen-column maximum is the same global amax in all32 groups.
    // Local floor commutes with max across the sixteen valid 256-value shards.
    let local_amax: DmTensor<f32, Chip, m![2], OutputQuantSlices, m![1 # 8]> = device
        .main.begin(x.view())
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>().fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7fff_ffffi32)
        .vector_reinterpret::<f32>()
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_intra_slice_reduce::<Qs, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, 1.504632769052528e-36f32)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    // Preserve H/120 row-group ownership and reduce only the sixteen K columns.
    // Exactly q=0 is initialized in each group; padded q slots stay invalid.
    let amax: DmTensor<f32, Chip, m![2], OutputQuantRows, m![1 # 8]> = device
        .main.begin(local_amax.view())
        .fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<OutputQuantRows, m![1]>(InterSliceReduceOpF32::Max)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    // Exponent mask produces 2^floor(log2(amax)); floor 2^-119 guarantees
    // scale >= 2^-126, and finite BF16 inputs imply scale <= 2^120.
    let scale: DmTensor<f32, Chip, m![2], OutputQuantRows, m![1 # 8]> = device
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
    let quant_scale: DmTensor<f32, Chip, m![2], OutputQuantSlices, m![1 # 8]> = scale.to_dm(&mut device.tdma);
    let scale_vrf: VrfTensor<f32, Chip, m![2], OutputQuantSlices, m![1 # 8]> = device
        .sub.begin(quant_scale.view())
        .fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let mut packed: DmTensor<f8e4m3, Chip, m![2], OutputQuantSlices, m![OutputTerm, Qs % 256]> = DmTensor::new();
    device.main.begin(x.view())
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>().fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>().vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>().commit_trim::<m![Qs % 8]>()
        .commit_view(packed.view_mut().tile::<m![OutputTerm], 1, m![OutputTerm = 1 #{!} 4, Qs % 256]>(0));
    let sum_vrf: VrfTensor<f32, Chip, m![2], OutputQuantSlices, m![Qs % 256]> = device.sub
        .begin(packed.view().tile::<m![OutputTerm], 1, m![OutputTerm = 1 # 4, Qs % 256]>(0))
        .fetch::<m![Qs / 32 % 8], m![Qs % 32]>().fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>().to_vrf();
    // Term 1: residual is x/s minus the F32 cumulative reconstruction.
    device.main.begin(x.view())
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>().fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &sum_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 16.0f32)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>().vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>().commit_trim::<m![Qs % 8]>()
        .commit_view(packed.view_mut().tile::<m![OutputTerm], 1, m![OutputTerm = 1 #{!} 4, Qs % 256]>(1));
    let sum: DmTensor<f32, Chip, m![2], OutputQuantSlices, m![Qs % 256]> = device.main
        .begin(packed.view().tile::<m![OutputTerm], 1, m![OutputTerm = 1 # 4, Qs % 256]>(1))
        .fetch::<m![Qs / 32 % 8], m![Qs % 32]>().fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0625f32)
        .vector_fp_binary(FpBinaryOp::AddF, &sum_vrf)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>().vector_final()
        .commit_trim::<m![Qs % 8]>().commit();
    let sum_vrf: VrfTensor<f32, Chip, m![2], OutputQuantSlices, m![Qs % 256]> = device.sub
        .begin(sum.view()).fetch::<m![Qs / 8 % 32], m![Qs % 8]>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>().to_vrf();
    // Term 2: residual is x/s minus the F32 cumulative reconstruction.
    device.main.begin(x.view())
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>().fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &sum_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 4194304.0f32)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>().vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>().commit_trim::<m![Qs % 8]>()
        .commit_view(packed.view_mut().tile::<m![OutputTerm], 1, m![OutputTerm = 1 #{!} 4, Qs % 256]>(2));
    let sum: DmTensor<f32, Chip, m![2], OutputQuantSlices, m![Qs % 256]> = device.main
        .begin(packed.view().tile::<m![OutputTerm], 1, m![OutputTerm = 1 # 4, Qs % 256]>(2))
        .fetch::<m![Qs / 32 % 8], m![Qs % 32]>().fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0000002384185791015625f32)
        .vector_fp_binary(FpBinaryOp::AddF, &sum_vrf)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>().vector_final()
        .commit_trim::<m![Qs % 8]>().commit();
    let sum_vrf: VrfTensor<f32, Chip, m![2], OutputQuantSlices, m![Qs % 256]> = device.sub
        .begin(sum.view()).fetch::<m![Qs / 8 % 32], m![Qs % 8]>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>().to_vrf();
    // Term 3: residual is x/s minus the F32 cumulative reconstruction.
    device.main.begin(x.view())
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>().fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &sum_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 134217728.0f32)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>().vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>().commit_trim::<m![Qs % 8]>()
        .commit_view(packed.view_mut().tile::<m![OutputTerm], 1, m![OutputTerm = 1 #{!} 4, Qs % 256]>(3));
    // Constant normalized coefficients: common s is deliberately restored after Term sum.
    // Zero*finite(scale)+constant builds exact bits without an extra public input.
    let mut coefficients: DmTensor<f32, Chip, m![2], OutputQuantRows, m![OutputTerm, 1 # 8]> = DmTensor::new();
    device.main.begin(scale.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0f32)
        .vector_fp_binary(FpBinaryOp::AddF, 1.0f32)
        .vector_widen_pad::<m![1 # 8]>().vector_final().commit_trim::<m![1 # 8]>()
        .commit_view(coefficients.view_mut().tile::<m![OutputTerm], 1, m![OutputTerm = 1 #{!} 4, 1 # 8]>(0));
    device.main.begin(scale.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0f32)
        .vector_fp_binary(FpBinaryOp::AddF, 0.0625f32)
        .vector_widen_pad::<m![1 # 8]>().vector_final().commit_trim::<m![1 # 8]>()
        .commit_view(coefficients.view_mut().tile::<m![OutputTerm], 1, m![OutputTerm = 1 #{!} 4, 1 # 8]>(1));
    device.main.begin(scale.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0f32)
        .vector_fp_binary(FpBinaryOp::AddF, 0.0000002384185791015625f32)
        .vector_widen_pad::<m![1 # 8]>().vector_final().commit_trim::<m![1 # 8]>()
        .commit_view(coefficients.view_mut().tile::<m![OutputTerm], 1, m![OutputTerm = 1 #{!} 4, 1 # 8]>(2));
    device.main.begin(scale.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0f32)
        .vector_fp_binary(FpBinaryOp::AddF, 0.000000007450580596923828125f32)
        .vector_widen_pad::<m![1 # 8]>().vector_final().commit_trim::<m![1 # 8]>()
        .commit_view(coefficients.view_mut().tile::<m![OutputTerm], 1, m![OutputTerm = 1 #{!} 4, 1 # 8]>(3));
    // Spatial Term lanes need consecutive VRF cells. Pack the four scalar
    // time slots with two supported 8B commits, before replication.
    let coefficients: DmTensor<f32, Chip, m![2], OutputQuantRows, m![OutputTerm]> = device.sub
        .begin(coefficients.view()).fetch::<m![OutputTerm], m![1 # 8]>()
        .collect::<m![OutputTerm], m![1 # 8]>()
        .transpose::<m![OutputTerm / 2], m![OutputTerm % 2 # 8]>()
        .commit_trim::<m![OutputTerm % 2]>().commit();
    // Input DMA gives each (c,g) the identical4096 finite BF16 values x[256q+u].
    // K16 Max/floor/exponentmask/pow2scale are deterministic, so all32 scalars
    // are equal. Relabel only the full allocation Hgroup axis as anonymous16;
    // keep physical256 slices and q=0 live mask unchanged. Never shrink to1#256.
    let shared_scale: DmTensor<f32, Chip, m![2], m![16, 1 # 16], m![1 # 8]> = unsafe { scale.reshape() };
    // Packed data is already in the consuming row/K slices. The cluster relabel
    // m![2] -> Dummy2 changes no physical position or byte, and makes no DMA.
    let packed: DmTensor<f8e4m3, Chip, OutputCluster, OutputColumns, m![OutputTerm, Qs % 256]> = unsafe { packed.reshape() };
    (packed, shared_scale.to_dm(&mut device.tdma), coefficients.to_dm(&mut device.tdma))
}

pub(crate) fn project_output(
    device: &mut Device,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    weight_scale: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, OutputEpilogueSlices, m![H % 480]> {
    type OutputRows = m![H / 120 % 16, 1 # 16];
    let x: DmTensor<bf16, Chip, m![2], OutputQuantSlices, m![Qs % 256]> = x.to_dm(&mut device.tdma);
    let (x, x_scale, coefficients) = prepare_output_terms(device, &x);
    let x_trf: TrfTensor<f8e4m3, Chip, OutputCluster, OutputColumns, m![OutputTerm], m![Qs % 256]> = device
        .sub.begin(x.view())
        .fetch::<m![OutputTerm, Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![OutputTerm, Qs / 32 % 8], m![Qs % 32]>()
        .to_trf();
    let coefficients: VrfTensor<f32, Chip, OutputCluster, OutputColumns, m![OutputTerm # 8]> = device
        .sub.begin(coefficients.view()).fetch::<m![1], m![OutputTerm]>()
        .collect::<m![1], m![OutputTerm # 8]>().to_vrf();
    let weight: HbmTensorView<'_, f8e4m3, Chip, m![Dummy2, H / 120 % 16, H % 120, Qs]> = unsafe { weight.view().reshape() };
    // Distinct DM allocations expose the second DMA while the first row half computes.
    let first: DmTensor<f8e4m3, Chip, OutputCluster, OutputColumns, m![H % 120 = 60, Qs % 256]> =
        weight.tile::<m![H % 120], 60, m![Dummy2, H / 120 % 16, H % 120 = 60 # 120, Qs]>(0)
            .to_dm(&mut device.tdma);
    let second: DmTensor<f8e4m3, Chip, OutputCluster, OutputColumns, m![H % 120 = 60, Qs % 256]> =
        weight.tile::<m![H % 120], 60, m![Dummy2, H / 120 % 16, H % 120 = 60 # 120, Qs]>(60)
            .to_dm(&mut device.tdma);
    let mut result: DmTensor<f32, Chip, OutputCluster, OutputRows, m![H % 120]> = DmTensor::new();
    project_output_stage(device, &x_trf, &coefficients, &first, &mut result, 0);
    project_output_stage(device, &x_trf, &coefficients, &second, &mut result, 60);
    let result: DmTensor<f32, Chip, m![H / 1920], OutputRows, m![H % 120]> = unsafe { result.reshape() };
    let result: DmTensor<f32, Chip, Cluster, OutputEpilogueSlices, m![H % 480]> = result.to_dm(&mut device.tdma);
    apply_output_channel_scale_f32(device, &result, &x_scale, weight_scale)
}

fn project_output_stage(
    device: &mut Device,
    x: &TrfTensor<f8e4m3, Chip, OutputCluster, OutputColumns, m![OutputTerm], m![Qs % 256]>,
    coefficients: &VrfTensor<f32, Chip, OutputCluster, OutputColumns, m![OutputTerm # 8]>,
    weight: &DmTensor<f8e4m3, Chip, OutputCluster, OutputColumns, m![H % 120 = 60, Qs % 256]>,
    output: &mut DmTensor<f32, Chip, OutputCluster, m![H / 120 % 16, 1 # 16], m![H % 120]>,
    offset: usize,
) {
    device.main.begin(weight.view())
        .fetch::<m![H % 120 = 60, Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![H % 120 = 60, Qs / 32 % 8], m![Qs % 32]>()
        // Four native lanes consume the same streamed original E4 weight packet.
        .contract_outer::<m![H % 120 = 60, Qs / 64 % 4], m![Qs % 64], _, _, _>(x)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 60]>()
        .contract_lane::<m![H % 120 = 60], m![OutputTerm # 8]>(LaneMode::Interleaved)
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![OutputTerm]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), coefficients)
        .vector_intra_slice_reduce::<OutputTerm, m![H % 120 = 60], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<m![H / 120 % 16, 1 # 16], m![H % 120 = 60]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .transpose::<m![H % 120 = 60 / 2], m![H % 120 = 60 % 2 # 8]>()
        .commit_trim::<m![H % 120 = 60 % 2]>()
        .commit_view(output.view_mut().tile::<m![H % 120], 60, m![H % 120 = 60 #{!} 120]>(offset));
}

pub(crate) type OutputEpilogueSlices = m![1 # 32, H / 480];

fn apply_output_channel_scale_f32(
    device: &mut Device,
    x: &DmTensor<f32, Chip, Cluster, OutputEpilogueSlices, m![H % 480]>,
    x_scale: &DmTensor<f32, Chip, Cluster, OutputEpilogueSlices, m![1 # 8]>,
    weight_scale: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, OutputEpilogueSlices, m![H % 480]> {
    let x_scale_vrf: VrfTensor<f32, Chip, Cluster, OutputEpilogueSlices, m![1 # 8]> = device
        .sub.begin(x_scale.view()).fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>().to_vrf();
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
        .fetch::<m![H / 8 % 60], m![H % 8]>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &x_scale_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &weight_scale_vrf)
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

// O projection partitions 4096 BF16 inputs into sixteen 256-column groups.
type OutputCluster = m![Dummy2];
type OutputColumns = m![H / 120 % 16, Qs / 256];
