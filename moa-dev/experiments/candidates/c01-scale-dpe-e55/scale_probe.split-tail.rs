//! D01: isolate E33's compact block partial -> padded scale contraction path.
use furiosa_opt_std::prelude::*;
use crate::Chip;

axes![H = 3840, L = 15360, Dummy2 = 2, Block128 = 128];
type Clusters = m![Dummy2];
type Columns = m![H / 60 % 32, L / 1920];
type Rows = m![H / 60 % 32, 1 # 8];
pub type Trace = m![Dummy2, H / 60 % 32, L / 1920, H % 60, Block128 = 120];

macro_rules! static_tiles {
    ($i:ident, [$($value:expr),+], $body:block) => {
        $({ let $i: usize = $value; $body })+
    };
}

#[device(chip = 1)]
pub fn scale_probe(
    device: &mut Device,
    partial: &HbmTensor<f32, Chip, m![H, L / 16]>,
    scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    partial_trace: &mut HbmTensor<bf16, Chip, Trace>,
    scale_trace: &mut HbmTensor<bf16, Chip, Trace>,
    local_trace: &mut HbmTensor<f32, Chip, m![H, L / 1920]>,
    sum_trace: &mut HbmTensor<f32, Chip, m![H]>,
) {
    let p: HbmTensorView<'_, f32, Chip, m![Dummy2, H % 1920, L / 16]> = unsafe { partial.view().reshape() };
    let p: DmTensor<f32, Chip, Clusters, Columns, m![H % 60, L / 16 % 120]> = p.to_dm(&mut device.tdma);
    let s: HbmTensorView<'_, f8e4m3, Chip, m![Dummy2, H % 1920, L / 16]> = unsafe { scale.view().reshape() };
    let s: DmTensor<f8e4m3, Chip, Clusters, Columns, m![H % 60, L / 16 % 120]> = s.to_dm(&mut device.tdma);
    let mut local: DmTensor<f32, Chip, Clusters, Columns, m![H % 60]> = DmTensor::new();
    let mut sums: DmTensor<f32, Chip, Clusters, Rows, m![H % 60]> = DmTensor::new();

    static_tiles!(i, [0, 1, 2], {
        let pv: DmTensorView<'_, f32, Chip, Clusters, Columns, m![H % 60, Block128 = 120]> = unsafe { p.view().reshape() };
        let p128: DmTensor<bf16, Chip, Clusters, Columns, m![H % 60 = 20, Block128 = 120]> = device.main
            .begin(pv.tile::<m![H % 60], 20, m![H % 60 = 20 # 60, Block128 = 120]>(20 * i))
            .fetch::<m![H % 60 = 20, Block128 = 120 / 8], m![Block128 = 120 % 8]>()
            .collect::<m![H % 60 = 20, Block128 = 120 / 8], m![Block128 = 120 % 8]>()
            .cast::<bf16, m![Block128 = 120 % 8 # 16]>()
            .commit_trim::<m![Block128 = 120 % 8]>().commit();
        let sv: DmTensorView<'_, f8e4m3, Chip, Clusters, Columns, m![H % 60, Block128 = 120]> = unsafe { s.view().reshape() };
        let s128: DmTensor<bf16, Chip, Clusters, Columns, m![H % 60 = 20, Block128 = 120]> = device.sub
            .begin(sv.tile::<m![H % 60], 20, m![H % 60 = 20 # 60, Block128 = 120]>(20 * i))
            .fetch::<m![H % 60 = 20, Block128 = 120 / 8], m![Block128 = 120 % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 20, Block128 = 120 / 8], m![Block128 = 120 % 8]>()
            .cast::<bf16, m![Block128 = 120 % 8 # 16]>()
            .commit_trim::<m![Block128 = 120 % 8]>().commit();
        p128.view().to_hbm_view(&mut device.tdma,
            partial_trace.view_mut().tile::<m![H % 60], 20, m![Dummy2, H / 60 % 32, L / 1920, H % 60 = 20 #{!} 60, Block128 = 120]>(20 * i));
        s128.view().to_hbm_view(&mut device.tdma,
            scale_trace.view_mut().tile::<m![H % 60], 20, m![Dummy2, H / 60 % 32, L / 1920, H % 60 = 20 #{!} 60, Block128 = 120]>(20 * i));
        // DPE consumes the first112 blocks, VE handles the remaining8 live values.
        // No padding cell is read, and no compact-to-padded DMA is needed.
        let ts: VrfTensor<f32, Chip, Clusters, Columns, m![H % 60 = 20, Block128 = 8]> = device.sub
            .begin(s128.view().tile::<m![Block128 = 120], 8, m![H % 60 = 20, Block128 = 8 # 120]>(112))
            .fetch::<m![H % 60 = 20], m![Block128 = 8]>().fetch_cast::<f32>()
            .collect::<m![H % 60 = 20], m![Block128 = 8]>().to_vrf();
        let tail: DmTensor<f32, Chip, Clusters, Rows, m![H % 60 = 20, 1 # 8]> = device.main
            .begin(p128.view().tile::<m![Block128 = 120], 8, m![H % 60 = 20, Block128 = 8 # 120]>(112))
            .fetch::<m![H % 60 = 20], m![Block128 = 8]>().fetch_cast::<f32>()
            .collect::<m![H % 60 = 20], m![Block128 = 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![H % 60 = 20, Block128 = 8 / 4], m![Block128 = 8 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &ts)
            .vector_intra_slice_reduce::<Block128, m![H % 60 = 20], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<Rows, m![H % 60 = 20]>(InterSliceReduceOpF32::Add)
            .vector_final().commit_trim::<m![1 # 8]>().commit();
        let tail_vrf: VrfTensor<f32, Chip, Clusters, Rows, m![H % 60 = 20, 1 # 8]> = device.sub
            .begin(tail.view()).fetch::<m![H % 60 = 20], m![1 # 8]>()
            .collect::<m![H % 60 = 20], m![1 # 8]>().to_vrf();
        let mut p112: DmTensor<bf16, Chip, Clusters, Columns, m![H % 60 = 20, Block128 = 112]> = DmTensor::new();
        p128.view().tile::<m![Block128 = 120], 112, m![H % 60 = 20, Block128 = 112 # 120]>(0)
            .to_dm_view(&mut device.tdma, p112.view_mut());
        let mut s112: DmTensor<bf16, Chip, Clusters, Columns, m![H % 60 = 20, Block128 = 112]> = DmTensor::new();
        s128.view().tile::<m![Block128 = 120], 112, m![H % 60 = 20, Block128 = 112 # 120]>(0)
            .to_dm_view(&mut device.tdma, s112.view_mut());
        let st: TrfTensor<bf16, Chip, Clusters, Columns, m![1], m![H % 60 = 20, Block128 = 112]> = device.sub
            .begin(s112.view())
            .fetch::<m![H % 60 = 20, Block128 = 112 / 16], m![Block128 = 112 % 16]>()
            .collect::<m![H % 60 = 20, Block128 = 112 / 16], m![Block128 = 112 % 16]>().to_trf();
        device.main.begin(p112.view())
            .fetch::<m![H % 60 = 20, Block128 = 112 / 16], m![Block128 = 112 % 16]>()
            .collect::<m![H % 60 = 20, Block128 = 112 / 16], m![Block128 = 112 % 16]>()
            .contract_outer::<m![H % 60 = 20, Block128 = 112 / 16], m![Block128 = 112 % 16], _, _, _>(&st)
            .contract_packet::<m![1]>().contract_time::<m![H % 60 = 20]>()
            .contract_lane::<m![H % 60 = 20], m![1 # 8]>(LaneMode::Interleaved)
            .transpose::<m![H % 60 = 20 / 2], m![H % 60 = 20 % 2 # 8]>()
            .commit_trim::<m![H % 60 = 20 % 2]>()
            .commit_view(local.view_mut().tile::<m![H % 60], 20, m![H % 60 = 20 #{!} 60]>(20 * i));
        device.main.begin(p112.view())
            .fetch::<m![H % 60 = 20, Block128 = 112 / 16], m![Block128 = 112 % 16]>()
            .collect::<m![H % 60 = 20, Block128 = 112 / 16], m![Block128 = 112 % 16]>()
            .contract_outer::<m![H % 60 = 20, Block128 = 112 / 16], m![Block128 = 112 % 16], _, _, _>(&st)
            .contract_packet::<m![1]>().contract_time::<m![H % 60 = 20]>()
            .contract_lane::<m![H % 60 = 20], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<Rows, m![H % 60 = 20]>(InterSliceReduceOpF32::Add)
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpF32::Add, &tail_vrf)
            .vector_final()
            .transpose::<m![H % 60 = 20 / 2], m![H % 60 = 20 % 2 # 8]>()
            .commit_trim::<m![H % 60 = 20 % 2]>()
            .commit_view(sums.view_mut().tile::<m![H % 60], 20, m![H % 60 = 20 #{!} 60]>(20 * i));
    });
    let local: DmTensor<f32, Chip, m![H / 1920], Columns, m![H % 60]> = unsafe { local.reshape() };
    local.view().to_hbm_view(&mut device.tdma, local_trace.view_mut());
    let sums: DmTensor<f32, Chip, m![H / 1920], Rows, m![H % 60]> = unsafe { sums.reshape() };
    sums.view().to_hbm_view(&mut device.tdma, sum_trace.view_mut());
}
