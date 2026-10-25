//! Experimental BF16 full O with output channels in TRF lanes.
//! Keep eight K1024 BF16 partials and the original balanced eight-leaf BF16 addition tree.
//! No activation quantization and no shared mutable inputs.
use crate::{Chip, axes::{H, P128, Qf, Gf, Df}};
use furiosa_opt_std::prelude::*;
axes![Ho = 2, Hr = 240, Ht = 8, Half = 2, KBlock = 4, K1024 = 1024];
type Cluster = m![Ho];
type Parts = m![Hr #256];
type InputParts = m![KBlock,K1024/16];
pub type Output = m![P128,Ho,Hr,Ht];
type Sum = DmTensor<bf16,Chip,Cluster,Parts,m![P128,Ht]>;
type StoreParts = m![Hr/8#32,P128/32,1#2];

fn pair(device: &mut Device,
    partial: &DmTensor<bf16,Chip,Cluster,Parts,m![P128,KBlock,Ht]>,
    offset: usize) -> DmTensor<bf16,Chip,Cluster,Parts,m![P128,Ht]> {
    let left: VrfTensor<f32,Chip,Cluster,Parts,m![P128,Ht]> = device.sub
        .begin(partial.view().tile::<m![KBlock],1,m![P128,1#4,Ht]>(offset))
        .fetch::<m![P128],m![Ht]>().fetch_cast::<f32>()
        .collect::<m![P128],m![Ht]>().to_vrf();
    device.main
        .begin(partial.view().tile::<m![KBlock],1,m![P128,1#4,Ht]>(offset+1))
        .fetch::<m![P128],m![Ht]>().fetch_cast::<f32>()
        .collect::<m![P128],m![Ht]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add,&left).vector_final()
        .cast::<bf16,m![Ht #16]>().commit_trim::<m![Ht]>().commit()
}

fn half_one(device: &mut Device,
    input: &HbmTensor<bf16,Chip,m![P128,Gf,Df]>,
    weight: &TrfTensor<bf16,Chip,Cluster,Parts,m![Ht],m![KBlock,K1024]>,
    half: usize) -> Sum {
    // Named-axis input distribution: each slice owns 16 contiguous K values.
    // The Switch broadcasts these values to output-channel slices without
    // materializing the full P128 x Qs activation separately in every slice.
    let flat: HbmTensorView<'_,bf16,Chip,m![P128,Half,KBlock,K1024]> = unsafe { input.view().reshape() };
    let input: DmTensor<bf16,Chip,Cluster,InputParts,m![P128,K1024%16]> = flat.tile::<m![Half],1,m![P128,Half=1#2,KBlock,K1024]>(half).to_dm(&mut device.tdma);
    let partial: DmTensor<bf16,Chip,Cluster,Parts,m![P128,KBlock,Ht]> = device.main
        .begin(input.view()).fetch::<m![P128],m![K1024%16]>()
        .switch::<Parts,m![P128,KBlock,K1024/16]>(SwitchConfig::CustomBroadcast{ring_size:256})
        .collect::<m![P128,KBlock,K1024/16],m![K1024%16]>()
        .contract_outer::<m![P128,KBlock,K1024/32],m![K1024%32],_,_,_>(weight)
        .contract_packet::<m![1]>().contract_time::<m![P128,KBlock]>()
        .contract_lane::<m![P128,KBlock],m![Ht]>(LaneMode::Interleaved)
        .cast::<bf16,m![Ht #16]>().commit_trim::<m![Ht]>().commit();
    let p01 = pair(device,&partial,0usize);
    let p23 = pair(device,&partial,2usize);
    let p01: VrfTensor<f32,Chip,Cluster,Parts,m![P128,Ht]> = device.sub.begin(p01.view())
        .fetch::<m![P128],m![Ht]>().fetch_cast::<f32>()
        .collect::<m![P128],m![Ht]>().to_vrf();
    let sum: DmTensor<bf16,Chip,Cluster,Parts,m![P128,Ht]> = device.main.begin(p23.view())
        .fetch::<m![P128],m![Ht]>().fetch_cast::<f32>()
        .collect::<m![P128],m![Ht]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add,&p01).vector_final()
        .cast::<bf16,m![Ht #16]>().commit_trim::<m![Ht]>().commit();
    sum
}

fn finish(device: &mut Device, low: &Sum, high: &Sum,
    scale: &VrfTensor<f32,Chip,Cluster,Parts,m![Ht]>, output: HbmTensorViewMut<'_,bf16,Chip,Output>) {
    let low: VrfTensor<f32,Chip,Cluster,Parts,m![P128,Ht]> = device.sub.begin(low.view())
        .fetch::<m![P128],m![Ht]>().fetch_cast::<f32>()
        .collect::<m![P128],m![Ht]>().to_vrf();
    let sum: Sum = device.main.begin(high.view())
        .fetch::<m![P128],m![Ht]>().fetch_cast::<f32>()
        .collect::<m![P128],m![Ht]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add,&low).vector_final()
        .cast::<bf16,m![Ht #16]>().commit_trim::<m![Ht]>().commit();
    let result: DmTensor<bf16,Chip,Cluster,Parts,m![P128,Ht]> = device.main.begin(sum.view())
        .fetch::<m![P128],m![Ht]>().fetch_cast::<f32>()
        .collect::<m![P128],m![Ht]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P128,Ht/4],m![Ht%4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),scale)
        .vector_widen_concat::<m![P128],m![Ht]>().vector_final()
        .cast::<bf16,m![Ht #16]>().commit_trim::<m![Ht]>().commit();
    // Real redistribution joins eight neighboring H8 lanes and splits P32.
    // Canonical row-major output includes all H64 join/store costs in this graph.
    let result: DmTensor<bf16,Chip,Cluster,StoreParts,m![P128%32,Hr%8,Ht]> = result.to_dm(&mut device.tdma);
    result.view().to_hbm_view(&mut device.tdma,output);
}

fn weight_half(device: &mut Device,
    weight: &DmTensor<bf16,Chip,Cluster,Parts,m![Ht,Half,KBlock,K1024]>, half: usize)
    -> TrfTensor<bf16,Chip,Cluster,Parts,m![Ht],m![KBlock,K1024]> {
    device.sub.begin(weight.view().tile::<m![Half],1,m![Ht,Half=1#2,KBlock,K1024]>(half))
        .fetch::<m![Ht,KBlock,K1024/16],m![K1024%16]>()
        .collect::<m![Ht,KBlock,K1024/16],m![K1024%16]>().to_trf()
}

#[device(chip=1)]
pub fn project_full_output_four(device: &mut Device,
    a: &HbmTensor<bf16,Chip,m![P128,Gf,Df]>,
    b: &HbmTensor<bf16,Chip,m![P128,Gf,Df]>,
    c: &HbmTensor<bf16,Chip,m![P128,Gf,Df]>,
    d: &HbmTensor<bf16,Chip,m![P128,Gf,Df]>,
    weight: &HbmTensor<f8e4m3,Chip,m![H,Qf]>,
    scale: &HbmTensor<bf16,Chip,m![H]>,
    oa: HbmTensorViewMut<'_,bf16,Chip,Output>,
    ob: HbmTensorViewMut<'_,bf16,Chip,Output>,
    oc: HbmTensorViewMut<'_,bf16,Chip,Output>,
    od: HbmTensorViewMut<'_,bf16,Chip,Output>) {
    let weight_view: HbmTensorView<'_,f8e4m3,Chip,m![Ho,Hr,Ht,Half,KBlock,K1024]> = unsafe { weight.view().reshape() };
    let packed: DmTensor<f8e4m3,Chip,Cluster,Parts,m![Ht,Half,KBlock,K1024]> = weight_view.to_dm(&mut device.tdma);
    let weight: DmTensor<bf16,Chip,Cluster,Parts,m![Ht,Half,KBlock,K1024]> = device.main.begin(packed.view())
        .fetch::<m![Ht,Half,KBlock,K1024/32],m![K1024%32]>().fetch_table_lookup::<bf16>()
        .collect::<m![Ht,Half,KBlock,K1024/16],m![K1024%16]>().commit_trim::<m![K1024%16]>().commit();
    let (a0,b0,c0,d0) = {
        let w = weight_half(device,&weight,0usize);
        (half_one(device,a,&w,0usize),half_one(device,b,&w,0usize),
         half_one(device,c,&w,0usize),half_one(device,d,&w,0usize))
    };
    let scale_view: HbmTensorView<'_,bf16,Chip,m![Ho,Hr,Ht]> = unsafe { scale.view().reshape() };
    let scale: DmTensor<bf16,Chip,Cluster,Parts,m![Ht]> = scale_view.to_dm(&mut device.tdma);
    let scale: VrfTensor<f32,Chip,Cluster,Parts,m![Ht]> = device.sub.begin(scale.view())
        .fetch::<m![1],m![Ht]>().fetch_cast::<f32>()
        .collect::<m![1],m![Ht]>().to_vrf();
    let w = weight_half(device,&weight,1usize);
    let a1=half_one(device,a,&w,1usize); finish(device,&a0,&a1,&scale,oa);
    let b1=half_one(device,b,&w,1usize); finish(device,&b0,&b1,&scale,ob);
    let c1=half_one(device,c,&w,1usize); finish(device,&c0,&c1,&scale,oc);
    let d1=half_one(device,d,&w,1usize); finish(device,&d0,&d1,&scale,od);
}
