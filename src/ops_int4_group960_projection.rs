//! Disconnected per960 I4 projection probes. No model or quantizer caller.
//! Each local integer dot is converted exactly before its own group scale.
//! Weight scales describe provided BF16(F4*E4), not native-E4 code amplitudes.
use furiosa_opt_std::prelude::*;
use crate::{Chip, axes::{P512, H, L}};
use crate::ops_batch_layouts::{P512H960, P512L960};
pub use crate::ops_ffn_projection_clusters::{FfnProjection, JoinedWeight, JoinedOutput};
pub use crate::ops_native_fp8_blocked::DownWeight;
pub use crate::ops_native_fp8_p512_downblocked::Output;

pub type JoinedBlockScale = m![FfnProjection,L/240,H/960,L%240];
pub type DownBlockScale = m![H/240,L/960,H%240];
type ProjectionCluster = m![FfnProjection];
type Cluster = m![P512/256];
type UpParts = m![L/240,H/960];
type UpRows = m![L/240,1#4];
type InputRows = m![P512%256=128%4,P512%256=128/8,H/960];
type DownParts = m![H/240,L/960];
type DownRows = m![H/240,1#16];

#[device(chip=1)]
pub fn project_upgate(
    device: &mut Device,
    input: &HbmTensor<i4,Chip,P512H960>,
    xscale: &HbmTensor<f32,Chip,m![P512]>,
    weight: &HbmTensor<i4,Chip,JoinedWeight>,
    block_scale: &HbmTensor<f32,Chip,JoinedBlockScale>,
    output: &mut HbmTensor<bf16,Chip,JoinedOutput>,
) {
    let weight: DmTensor<i4,Chip,ProjectionCluster,UpParts,m![L%240,H%960]> = weight.to_dm(&mut device.tdma);
    let block: DmTensor<f32,Chip,ProjectionCluster,UpParts,m![L%240]> = block_scale.to_dm(&mut device.tdma);
    let block: VrfTensor<f32,Chip,ProjectionCluster,UpParts,m![L%240]> = device.sub.begin(block.view())
        .fetch::<m![L/8%30],m![L%8]>().collect::<m![L/8%30],m![L%8]>().to_vrf();
    macro_rules! chunk { ($chunk:expr) => {{
        let half = $chunk / 2;
        let chunk = $chunk % 2;
        let input: DmTensor<i4,Chip,ProjectionCluster,InputRows,m![P512%256=128%8/4,H%960]> = input.view()
            .tile::<m![P512/256],1,m![P512/256=1#2,H/960,P512%256,H%960]>(half)
            .tile::<m![P512%256],128,m![P512/256=1#2,H/960,P512%256=128#256,H%960]>(128*chunk)
            .to_dm(&mut device.tdma);
        let xs: DmTensor<f32,Chip,ProjectionCluster,UpRows,m![P512%256=128]> = xscale.view()
            .tile::<m![P512/256],1,m![P512/256=1#2,P512%256]>(half)
            .tile::<m![P512%256],128,m![P512/256=1#2,P512%256=128#256]>(128*chunk)
            .to_dm(&mut device.tdma);
        let xs: VrfTensor<f32,Chip,ProjectionCluster,UpRows,m![P512%256=128]> = device.sub.begin(xs.view())
            .fetch::<m![P512%256=128/8],m![P512%256=128%8]>()
            .collect::<m![P512%256=128/8],m![P512%256=128%8]>().to_vrf();
        let x: TrfTensor<i4,Chip,ProjectionCluster,UpParts,m![P512%256=128%8],m![P512%256=128/8,H%960]> = device.sub.begin(input.view())
            .fetch::<m![P512%256=128%8/4],m![H%960]>()
            .switch::<UpParts,m![P512%256=128%8,P512%256=128/8]>(SwitchConfig::CustomBroadcast{ring_size:256})
            .collect::<m![P512%256=128%8,P512%256=128/8,H/64%15],m![H%64]>().to_trf();
        let reduced: DmTensor<f32,Chip,ProjectionCluster,UpRows,m![L%240,P512%256=128]> = device.main.begin(weight.view())
            .fetch::<m![L%240,H/64%15],m![H%64]>()
            .collect::<m![L%240,H/64%15],m![H%64]>()
            .contract_outer::<m![L%240,H/64%15,P512%256=128/8],m![H%64],_,_,_>(&x)
            .contract_packet::<m![1]>().contract_time::<m![L%240,P512%256=128/8]>()
            .contract_lane::<m![L%240,P512%256=128/8],m![P512%256=128%8]>(LaneMode::Interleaved)
            .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_fxp_to_fp(31)
            .vector_narrow_split::<m![L%240,P512%256=128/4],m![P512%256=128%4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&block)
            .vector_widen_concat::<m![L%240,P512%256=128/8],m![P512%256=128%8]>()
            .vector_inter_slice_reduce::<UpRows,m![L%240,P512%256=128/8]>(InterSliceReduceOpF32::Add)
            .vector_final().commit_trim::<m![P512%256=128%8]>().commit();
        // Intra-first reduction ends the SDK vector pass. Retain the real F32
        // DM boundary instead of bypassing that stage constraint.
        let result: DmTensor<bf16,Chip,ProjectionCluster,UpRows,m![L%240,P512%256=128]> = device.main.begin(reduced.view())
            .fetch::<m![L%240,P512%256=128/8],m![P512%256=128%8]>()
            .collect::<m![L%240,P512%256=128/8],m![P512%256=128%8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![L%240,P512%256=128/4],m![P512%256=128%4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&xs)
            .vector_widen_concat::<m![L%240,P512%256=128/8],m![P512%256=128%8]>()
            .vector_final().cast::<bf16,m![P512%256=128%8#16]>()
            .commit_trim::<m![P512%256=128%8]>().commit();
        let result: DmTensor<bf16,Chip,ProjectionCluster,UpRows,m![P512%256=128,L%240]> = device.main.begin(result.view())
            .fetch::<m![P512%256=128/16,L%240],m![P512%256=128%16]>()
            .collect::<m![P512%256=128/16,L%240],m![P512%256=128%16]>()
            .transpose::<m![P512%256=128/16,L/4%60,P512%256=128%16],m![L%4#16]>()
            .commit_trim::<m![L%4]>().commit();
        result.view().to_hbm_view(&mut device.tdma,output.view_mut()
            .tile::<m![P512/256],1,m![FfnProjection,P512/256=1#{!}2,L/240,P512%256,L%240]>(half)
            .tile::<m![P512%256],128,m![FfnProjection,P512/256=1#{!}2,L/240,P512%256=128#{!}256,L%240]>(128*chunk));
    }}; }
    chunk!(0usize); chunk!(1usize); chunk!(2usize); chunk!(3usize);
}

#[device(chip=1)]
pub fn project_down(
    device: &mut Device,
    input: &HbmTensor<i4,Chip,P512L960>,
    xscale: &HbmTensor<f32,Chip,m![P512]>,
    weight: &HbmTensor<i4,Chip,DownWeight>,
    block_scale: &HbmTensor<f32,Chip,DownBlockScale>,
    global_scale: &HbmTensor<f32,Chip,m![1]>,
    output: &mut HbmTensor<bf16,Chip,Output>,
) {
    let weight: DmTensor<i4,Chip,Cluster,DownParts,m![H%240,L%960]> = weight.to_dm(&mut device.tdma);
    let block: DmTensor<f32,Chip,Cluster,DownParts,m![H%240]> = block_scale.to_dm(&mut device.tdma);
    let block: VrfTensor<f32,Chip,Cluster,DownParts,m![H%240]> = device.sub.begin(block.view())
        .fetch::<m![H/8%30],m![H%8]>().collect::<m![H/8%30],m![H%8]>().to_vrf();
    let global: DmTensor<f32,Chip,Cluster,DownRows,m![1#8]> = global_scale.to_dm(&mut device.tdma);
    let global: VrfTensor<f32,Chip,Cluster,DownRows,m![1#8]> = device.sub.begin(global.view())
        .fetch::<m![1],m![1#8]>().collect::<m![1],m![1#8]>().to_vrf();
    macro_rules! chunk { ($chunk:expr) => {{
        let input: DmTensor<i4,Chip,Cluster,DownParts,m![P512%256=128,L%960]> = input.view()
            .tile::<m![P512%256],128,m![P512/256,L/960,P512%256=128#256,L%960]>(128*$chunk)
            .to_dm(&mut device.tdma);
        let xs: DmTensor<f32,Chip,Cluster,DownRows,m![P512%256=128]> = xscale.view()
            .tile::<m![P512%256],128,m![P512/256,P512%256=128#256]>(128*$chunk)
            .to_dm(&mut device.tdma);
        let xs: VrfTensor<f32,Chip,Cluster,DownRows,m![P512%256=128]> = device.sub.begin(xs.view())
            .fetch::<m![P512%256=128/8],m![P512%256=128%8]>()
            .collect::<m![P512%256=128/8],m![P512%256=128%8]>().to_vrf();
        let x: TrfTensor<i4,Chip,Cluster,DownParts,m![P512%256=128%8],m![P512%256=128/8,L%960]> = device.sub.begin(input.view())
            .fetch::<m![P512%256=128%8,P512%256=128/8,L/64%15],m![L%64]>()
            .collect::<m![P512%256=128%8,P512%256=128/8,L/64%15],m![L%64]>().to_trf();
        let reduced: DmTensor<f32,Chip,Cluster,DownRows,m![H%240,P512%256=128]> = device.main.begin(weight.view())
            .fetch::<m![H%240,L/64%15],m![L%64]>()
            .collect::<m![H%240,L/64%15],m![L%64]>()
            .contract_outer::<m![H%240,L/64%15,P512%256=128/8],m![L%64],_,_,_>(&x)
            .contract_packet::<m![1]>().contract_time::<m![H%240,P512%256=128/8]>()
            .contract_lane::<m![H%240,P512%256=128/8],m![P512%256=128%8]>(LaneMode::Interleaved)
            .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_fxp_to_fp(31)
            .vector_narrow_split::<m![H%240,P512%256=128/4],m![P512%256=128%4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&block)
            .vector_widen_concat::<m![H%240,P512%256=128/8],m![P512%256=128%8]>()
            .vector_inter_slice_reduce::<DownRows,m![H%240,P512%256=128/8]>(InterSliceReduceOpF32::Add)
            .vector_final().commit_trim::<m![P512%256=128%8]>().commit();
        let result: DmTensor<bf16,Chip,Cluster,DownRows,m![H%240,P512%256=128]> = device.main.begin(reduced.view())
            .fetch::<m![H%240,P512%256=128/8],m![P512%256=128%8]>()
            .collect::<m![H%240,P512%256=128/8],m![P512%256=128%8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![H%240,P512%256=128/4],m![P512%256=128%4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&xs)
            .vector_widen_concat::<m![H%240,P512%256=128/8],m![P512%256=128%8]>()
            .vector_final().cast::<bf16,m![P512%256=128%8#16]>()
            .commit_trim::<m![P512%256=128%8]>().commit();
        // The original projection BF16 rounding precedes global scaling.
        let result: DmTensor<bf16,Chip,Cluster,DownRows,m![H%240,P512%256=128]> = device.main.begin(result.view())
            .fetch::<m![H%240,P512%256=128/8],m![P512%256=128%8]>().fetch_cast::<f32>()
            .collect::<m![H%240,P512%256=128/8],m![P512%256=128%8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![H%240,P512%256=128/4],m![P512%256=128%4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&global)
            .vector_widen_concat::<m![H%240,P512%256=128/8],m![P512%256=128%8]>()
            .vector_final().cast::<bf16,m![P512%256=128%8#16]>()
            .commit_trim::<m![P512%256=128%8]>().commit();
        let result: DmTensor<bf16,Chip,Cluster,DownRows,m![P512%256=128,H%240]> = device.main.begin(result.view())
            .fetch::<m![P512%256=128/16,H%240],m![P512%256=128%16]>()
            .collect::<m![P512%256=128/16,H%240],m![P512%256=128%16]>()
            .transpose::<m![P512%256=128/16,H/4%60,P512%256=128%16],m![H%4#16]>()
            .commit_trim::<m![H%4]>().commit();
        result.view().to_hbm_view(&mut device.tdma,output.view_mut()
            .tile::<m![P512%256],128,m![P512/256,H/240,P512%256=128#{!}256,H%240]>(128*$chunk));
    }}; }
    chunk!(0usize); chunk!(1usize);
}
