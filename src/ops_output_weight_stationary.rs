//! Experimental BF16 sliding O with output channels in TRF lanes.
//! Keep four K1024 BF16 partials and the original balanced BF16 addition tree.
//! No activation quantization and no shared mutable inputs.
use crate::{Chip, axes::{H, P128, Qs, Ns, Gs, Ds}};
use furiosa_opt_std::prelude::*;
type Cluster = m![H/1920];
type Parts = m![H/8%240#256];
type InputParts = m![Qs/16];
pub type Output = m![P128,H];

fn pair(device: &mut Device,
    partial: &DmTensor<bf16,Chip,Cluster,Parts,m![P128,Qs/1024,H%8]>,
    offset: usize) -> DmTensor<bf16,Chip,Cluster,Parts,m![P128,H%8]> {
    let left: VrfTensor<f32,Chip,Cluster,Parts,m![P128,H%8]> = device.sub
        .begin(partial.view().tile::<m![Qs/1024],1,m![P128,1#4,H%8]>(offset))
        .fetch::<m![P128],m![H%8]>().fetch_cast::<f32>()
        .collect::<m![P128],m![H%8]>().to_vrf();
    device.main
        .begin(partial.view().tile::<m![Qs/1024],1,m![P128,1#4,H%8]>(offset+1))
        .fetch::<m![P128],m![H%8]>().fetch_cast::<f32>()
        .collect::<m![P128],m![H%8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add,&left).vector_final()
        .cast::<bf16,m![H%8#16]>().commit_trim::<m![H%8]>().commit()
}

fn one(device: &mut Device,
    input: &HbmTensor<bf16,Chip,m![P128,Ns,Gs,Ds]>,
    weight: &TrfTensor<bf16,Chip,Cluster,Parts,m![H%8],m![Qs]>,
    scale: &VrfTensor<f32,Chip,Cluster,Parts,m![H%8]>,
    output: &mut HbmTensor<bf16,Chip,Output>) {
    // Named-axis input distribution: each slice owns 16 contiguous K values.
    // The Switch broadcasts these values to output-channel slices without
    // materializing the full P128 x Qs activation separately in every slice.
    let flat: HbmTensorView<'_,bf16,Chip,m![P128,Qs]> = unsafe { input.view().reshape() };
    let input: DmTensor<bf16,Chip,Cluster,InputParts,m![P128,Qs%16]> = flat.to_dm(&mut device.tdma);
    let partial: DmTensor<bf16,Chip,Cluster,Parts,m![P128,Qs/1024,H%8]> = device.main
        .begin(input.view()).fetch::<m![P128],m![Qs%16]>()
        .switch::<Parts,m![P128,Qs/16]>(SwitchConfig::CustomBroadcast{ring_size:256})
        .collect::<m![P128,Qs/16],m![Qs%16]>()
        .contract_outer::<m![P128,Qs/32],m![Qs%32],_,_,_>(weight)
        .contract_packet::<m![1]>().contract_time::<m![P128,Qs/1024]>()
        .contract_lane::<m![P128,Qs/1024],m![H%8]>(LaneMode::Interleaved)
        .cast::<bf16,m![H%8#16]>().commit_trim::<m![H%8]>().commit();
    let p01 = pair(device,&partial,0usize);
    let p23 = pair(device,&partial,2usize);
    let p01: VrfTensor<f32,Chip,Cluster,Parts,m![P128,H%8]> = device.sub.begin(p01.view())
        .fetch::<m![P128],m![H%8]>().fetch_cast::<f32>()
        .collect::<m![P128],m![H%8]>().to_vrf();
    let sum: DmTensor<bf16,Chip,Cluster,Parts,m![P128,H%8]> = device.main.begin(p23.view())
        .fetch::<m![P128],m![H%8]>().fetch_cast::<f32>()
        .collect::<m![P128],m![H%8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add,&p01).vector_final()
        .cast::<bf16,m![H%8#16]>().commit_trim::<m![H%8]>().commit();
    let result: DmTensor<bf16,Chip,Cluster,Parts,m![P128,H%8]> = device.main.begin(sum.view())
        .fetch::<m![P128],m![H%8]>().fetch_cast::<f32>()
        .collect::<m![P128],m![H%8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P128,H/4%2],m![H%4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),scale)
        .vector_widen_concat::<m![P128],m![H%8]>().vector_final()
        .cast::<bf16,m![H%8#16]>().commit_trim::<m![H%8]>().commit();
    result.view().to_hbm_view(&mut device.tdma,output.view_mut());
}

#[device(chip=1)]
pub fn project_sliding_output_four(device: &mut Device,
    a: &HbmTensor<bf16,Chip,m![P128,Ns,Gs,Ds]>,
    b: &HbmTensor<bf16,Chip,m![P128,Ns,Gs,Ds]>,
    c: &HbmTensor<bf16,Chip,m![P128,Ns,Gs,Ds]>,
    d: &HbmTensor<bf16,Chip,m![P128,Ns,Gs,Ds]>,
    weight: &HbmTensor<f8e4m3,Chip,m![H,Qs]>,
    scale: &HbmTensor<bf16,Chip,m![H]>,
    oa: &mut HbmTensor<bf16,Chip,Output>,
    ob: &mut HbmTensor<bf16,Chip,Output>,
    oc: &mut HbmTensor<bf16,Chip,Output>,
    od: &mut HbmTensor<bf16,Chip,Output>) {
    let packed: DmTensor<f8e4m3,Chip,Cluster,Parts,m![H%8,Qs]> = weight.to_dm(&mut device.tdma);
    let weight: DmTensor<bf16,Chip,Cluster,Parts,m![H%8,Qs]> = device.main.begin(packed.view())
        .fetch::<m![H%8,Qs/32],m![Qs%32]>().fetch_table_lookup::<bf16>()
        .collect::<m![H%8,Qs/16],m![Qs%16]>().commit_trim::<m![Qs%16]>().commit();
    let weight: TrfTensor<bf16,Chip,Cluster,Parts,m![H%8],m![Qs]> = device.sub.begin(weight.view())
        .fetch::<m![H%8,Qs/16],m![Qs%16]>()
        .collect::<m![H%8,Qs/16],m![Qs%16]>().to_trf();
    let scale: DmTensor<bf16,Chip,Cluster,Parts,m![H%8]> = scale.to_dm(&mut device.tdma);
    let scale: VrfTensor<f32,Chip,Cluster,Parts,m![H%8]> = device.sub.begin(scale.view())
        .fetch::<m![1],m![H%8]>().fetch_cast::<f32>()
        .collect::<m![1],m![H%8]>().to_vrf();
    one(device,a,&weight,&scale,oa);
    one(device,b,&weight,&scale,ob);
    one(device,c,&weight,&scale,oc);
    one(device,d,&weight,&scale,od);
}
