use furiosa_opt_std::prelude::*;
use crate::Chip;

axes![R = 4, H = 3840];
type Cl = m![1 # 2];
type Sl = m![1 # 256];

#[device(chip = 1)]
pub fn packet_probe(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    weight: &HbmTensor<bf16, Chip, m![R, H]>,
    scale: &HbmTensor<f8e4m3, Chip, m![R, H / 16]>,
    scale_trace: &mut HbmTensor<f32, Chip, m![R, H / 16]>,
    output: &mut HbmTensor<f32, Chip, m![R]>,
) {
    let x: DmTensor<bf16, Chip, Cl, Sl, m![H]> = x.to_dm(&mut device.tdma);
    let x: TrfTensor<bf16, Chip, Cl, Sl, m![1], m![H]> = device.sub.begin(x.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .collect::<m![H / 16], m![H % 16]>().to_trf();
    let w: DmTensor<bf16, Chip, Cl, Sl, m![R, H]> = weight.to_dm(&mut device.tdma);
    let s: DmTensor<f8e4m3, Chip, Cl, Sl, m![R, H / 16]> = scale.to_dm(&mut device.tdma);
    // Convert contiguous FP8 values first; no padding is requested by fetch_cast.
    let sf: DmTensor<f32, Chip, Cl, Sl, m![R, H / 16]> = device.sub.begin(s.view())
        .fetch::<m![R, H / 128], m![H / 16 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![R, H / 128], m![H / 16 % 8]>()
        .commit_trim::<m![H / 16 % 8]>().commit();
    // Typed DMA materializes two live F32 scales per four-value slot.
    let sp: DmTensor<f32, Chip, Cl, Sl, m![R, H / 32, H / 16 % 2 # 4]> = sf.to_dm(&mut device.tdma);
    sp.view().to_hbm_view(&mut device.tdma, scale_trace.view_mut());
    let sv: VrfTensor<f32, Chip, Cl, Sl, m![R, H / 32, H / 16 % 2 # 4]> = device.sub.begin(sp.view())
        .fetch::<m![R, H / 64], m![H / 32 % 2, H / 16 % 2 # 4]>()
        .collect::<m![R, H / 64], m![H / 32 % 2, H / 16 % 2 # 4]>().to_vrf();
    let y: DmTensor<f32, Chip, Cl, Sl, m![R]> = device.main.begin(w.view())
        .fetch::<m![R, H / 16], m![H % 16]>()
        .collect::<m![R, H / 16], m![H % 16]>()
        .contract_outer::<m![R, H / 32], m![H % 32], _, _, _>(&x)
        .contract_packet::<m![H / 16 % 2]>()
        .contract_time::<m![R, H / 32]>()
        .contract_lane::<m![R, H / 32], m![H / 16 % 2 # 8]>(LaneMode::Sequential)
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![H / 16 % 2 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &sv)
        .vector_intra_slice_reduce::<H, m![R], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .transpose::<m![R / 2], m![R % 2 # 8]>()
        .commit_trim::<m![R % 2]>().commit();
    y.view().to_hbm_view(&mut device.tdma, output.view_mut());
}
