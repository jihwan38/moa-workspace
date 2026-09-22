//! D01: isolate E33's compact block partial -> padded scale contraction path.
use furiosa_opt_std::prelude::*;
use crate::Chip;

axes![H = 3840, L = 15360, Dummy2 = 2, Block128 = 128];
type Clusters = m![Dummy2];
type Columns = m![H / 60 % 32, L / 1920];
type Rows = m![H / 60 % 32, 1 # 8];
pub type Trace = m![Dummy2, H / 60 % 32, L / 1920, H % 60, Block128];

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
        let p: DmTensor<bf16, Chip, Clusters, Columns, m![H % 60 = 20, L / 16 % 120]> = device.main
            .begin(p.view().tile::<m![H % 60], 20, m![H % 60 = 20 # 60, L / 16 % 120]>(20 * i))
            .fetch::<m![H % 60 = 20, L / 16 / 8 % 15], m![L / 16 % 8]>()
            .collect::<m![H % 60 = 20, L / 16 / 8 % 15], m![L / 16 % 8]>()
            .cast::<bf16, m![L / 16 % 8 # 16]>()
            .commit_trim::<m![L / 16 % 8]>().commit();
        let mut p128: DmTensor<bf16, Chip, Clusters, Columns, m![H % 60 = 20, Block128]> = DmTensor::new();
        p128.view_mut().memset(const { bf16::from_f32(0.0) }, &mut device.sub);
        let p: DmTensor<bf16, Chip, Clusters, Columns, m![H % 60 = 20, Block128 = 120]> = unsafe { p.reshape() };
        p.view().to_dm_view(&mut device.tdma,
            p128.view_mut().tile::<m![Block128], 120, m![H % 60 = 20, Block128 = 120 #{!} 128]>(0));
        // Explicit F32 zero source: ranged memset is unavailable in SDK 0.8.1.
        let mut zeros: DmTensor<f32, Chip, Clusters, Columns, m![H % 60 = 20, Block128 = 8]> = DmTensor::new();
        zeros.view_mut().memset(0.0f32, &mut device.sub);
        device.sub.begin(zeros.view())
            .fetch::<m![H % 60 = 20], m![Block128 = 8]>()
            .collect::<m![H % 60 = 20], m![Block128 = 8]>()
            .cast::<bf16, m![Block128 = 8 # 16]>()
            .commit_trim::<m![Block128 = 8]>()
            .commit_view(p128.view_mut().tile::<m![Block128], 8, m![H % 60 = 20, Block128 = 8 #{!} 128]>(120));
        let mut s128: DmTensor<f8e4m3, Chip, Clusters, Columns, m![H % 60 = 20, Block128]> = DmTensor::new();
        s128.view_mut().memset(const { f8e4m3::from_f32(0.0) }, &mut device.sub);
        let sv: DmTensorView<'_, f8e4m3, Chip, Clusters, Columns, m![H % 60, Block128 = 120]> = unsafe { s.view().reshape() };
        sv.tile::<m![H % 60], 20, m![H % 60 = 20 # 60, Block128 = 120]>(20 * i)
            .to_dm_view(&mut device.tdma,
                s128.view_mut().tile::<m![Block128], 120, m![H % 60 = 20, Block128 = 120 #{!} 128]>(0));
        let mut s128: DmTensor<bf16, Chip, Clusters, Columns, m![H % 60 = 20, Block128]> = device.main
            .begin(s128.view())
            .fetch::<m![H % 60 = 20, Block128 / 32], m![Block128 % 32]>()
            .fetch_table_lookup::<bf16>()
            .collect::<m![H % 60 = 20, Block128 / 16], m![Block128 % 16]>()
            .commit_trim::<m![Block128 % 16]>().commit();
        device.sub.begin(zeros.view())
            .fetch::<m![H % 60 = 20], m![Block128 = 8]>()
            .collect::<m![H % 60 = 20], m![Block128 = 8]>()
            .cast::<bf16, m![Block128 = 8 # 16]>()
            .commit_trim::<m![Block128 = 8]>()
            .commit_view(s128.view_mut().tile::<m![Block128], 8, m![H % 60 = 20, Block128 = 8 #{!} 128]>(120));
        p128.view().to_hbm_view(&mut device.tdma,
            partial_trace.view_mut().tile::<m![H % 60], 20, m![Dummy2, H / 60 % 32, L / 1920, H % 60 = 20 #{!} 60, Block128]>(20 * i));
        s128.view().to_hbm_view(&mut device.tdma,
            scale_trace.view_mut().tile::<m![H % 60], 20, m![Dummy2, H / 60 % 32, L / 1920, H % 60 = 20 #{!} 60, Block128]>(20 * i));
        let st: TrfTensor<bf16, Chip, Clusters, Columns, m![1], m![H % 60 = 20, Block128]> = device.sub
            .begin(s128.view())
            .fetch::<m![H % 60 = 20, Block128 / 16], m![Block128 % 16]>()
            .collect::<m![H % 60 = 20, Block128 / 16], m![Block128 % 16]>().to_trf();
        device.main.begin(p128.view())
            .fetch::<m![H % 60 = 20, Block128 / 16], m![Block128 % 16]>()
            .collect::<m![H % 60 = 20, Block128 / 16], m![Block128 % 16]>()
            .contract_outer::<m![H % 60 = 20, Block128 / 16], m![Block128 % 16], _, _, _>(&st)
            .contract_packet::<m![1]>().contract_time::<m![H % 60 = 20]>()
            .contract_lane::<m![H % 60 = 20], m![1 # 8]>(LaneMode::Interleaved)
            .transpose::<m![H % 60 = 20 / 2], m![H % 60 = 20 % 2 # 8]>()
            .commit_trim::<m![H % 60 = 20 % 2]>()
            .commit_view(local.view_mut().tile::<m![H % 60], 20, m![H % 60 = 20 #{!} 60]>(20 * i));
        device.main.begin(p128.view())
            .fetch::<m![H % 60 = 20, Block128 / 16], m![Block128 % 16]>()
            .collect::<m![H % 60 = 20, Block128 / 16], m![Block128 % 16]>()
            .contract_outer::<m![H % 60 = 20, Block128 / 16], m![Block128 % 16], _, _, _>(&st)
            .contract_packet::<m![1]>().contract_time::<m![H % 60 = 20]>()
            .contract_lane::<m![H % 60 = 20], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<Rows, m![H % 60 = 20]>(InterSliceReduceOpF32::Add)
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
