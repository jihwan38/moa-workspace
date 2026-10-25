//! Experimental original-E4 sliding QKV with output-owned retained weights.
//! Join Q8/K4/V4 inside DM, retain four H960 F32 partials, then preserve
//! xscale -> BF16 -> channel scale -> BF16. No new quantization or HBM copy.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{P128,H,Qs,Ps}};
axes![OutputHalf=2, OutputGroup=256, Joined=16, HBlock=4, H960=960];
type Cluster=m![OutputHalf];
type Parts=m![OutputGroup];
type InputParts=m![HBlock,H960/32,1#2];
type StoreParts=m![OutputGroup/8,P128/32,1#2];
pub type QueryOutput=m![P128/64,OutputHalf,OutputGroup/8,P128%64,OutputGroup%8,Joined=8];
pub type KeyOutput=m![P128/64,OutputHalf,OutputGroup/8,P128%64,OutputGroup%8,Joined=4];

fn one(device:&mut Device,input:&HbmTensor<f8e4m3,Chip,m![P128,H]>,
    xscale:&HbmTensor<f32,Chip,m![P128,1#8]>,
    weight:&TrfTensor<f8e4m3,Chip,Cluster,Parts,m![Joined%8],m![Joined/8,HBlock,H960]>,
    scale:&VrfTensor<f32,Chip,Cluster,Parts,m![Joined]>,
    qout:&mut HbmTensor<bf16,Chip,QueryOutput>,
    kout:&mut HbmTensor<bf16,Chip,KeyOutput>,
    vout:&mut HbmTensor<bf16,Chip,KeyOutput>) {
    let input:HbmTensorView<'_,f8e4m3,Chip,m![P128,HBlock,H960]>=unsafe{input.view().reshape()};
    let input:DmTensor<f8e4m3,Chip,Cluster,InputParts,m![P128,H960%32]>=input.to_dm(&mut device.tdma);
    let xs:DmTensor<f32,Chip,Cluster,Parts,m![P128,1#8]>=xscale.to_dm(&mut device.tdma);
    let xs:DmTensor<f32,Chip,Cluster,Parts,m![P128]>=device.sub.begin(xs.view())
        .fetch::<m![P128],m![1#8]>().collect::<m![P128],m![1#8]>()
        .transpose::<m![P128/2],m![P128%2#8]>().commit_trim::<m![P128%2]>().commit();
    let xs:VrfTensor<f32,Chip,Cluster,Parts,m![P128]>=device.sub.begin(xs.view())
        .fetch::<m![P128/8],m![P128%8]>().collect::<m![P128/8],m![P128%8]>().to_vrf();
    let partial:DmTensor<f32,Chip,Cluster,Parts,m![P128,HBlock,Joined]>=device.main
        .begin(input.view()).fetch::<m![P128],m![H960%32]>()
        .switch::<Parts,m![P128,HBlock,H960/32]>(SwitchConfig::CustomBroadcast{ring_size:256})
        .collect::<m![P128,HBlock,H960/32],m![H960%32]>()
        .contract_outer::<m![P128,HBlock,H960/64,Joined/8],m![H960%64],_,_,_>(weight)
        .contract_packet::<m![1]>().contract_time::<m![P128,HBlock,Joined/8]>()
        .contract_lane::<m![P128,HBlock,Joined/8],m![Joined%8]>(LaneMode::Interleaved)
        .commit_trim::<m![Joined%8]>().commit();
    // The tested local four-way Add matches the original four-slice F32 Add.
    let sum:DmTensor<f32,Chip,Cluster,Parts,m![P128,Joined]>=device.main.begin(partial.view())
        .fetch::<m![P128,HBlock,Joined/8],m![Joined%8]>()
        .collect::<m![P128,HBlock,Joined/8],m![Joined%8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P128,HBlock,Joined/4],m![Joined%4]>()
        .vector_intra_slice_reduce::<HBlock,m![P128,Joined/4],m![Joined%4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_concat::<m![P128,Joined/8],m![Joined%8]>().vector_final()
        .commit_trim::<m![Joined%8]>().commit();
    let dot:DmTensor<bf16,Chip,Cluster,Parts,m![P128,Joined]>=device.main.begin(sum.view())
        .fetch::<m![P128,Joined/8],m![Joined%8]>()
        .collect::<m![P128,Joined/8],m![Joined%8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P128,Joined/4],m![Joined%4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&xs)
        .vector_widen_concat::<m![P128,Joined/8],m![Joined%8]>().vector_final()
        .cast::<bf16,m![Joined%8#16]>().commit_trim::<m![Joined%8]>().commit();
    let result:DmTensor<bf16,Chip,Cluster,Parts,m![P128,Joined]>=device.main.begin(dot.view())
        .fetch::<m![P128],m![Joined]>().fetch_cast::<f32>()
        .collect::<m![P128,Joined/8],m![Joined%8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![P128,Joined/4],m![Joined%4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),scale)
        .vector_widen_concat::<m![P128,Joined/8],m![Joined%8]>().vector_final()
        .cast::<bf16,m![Joined%8#16]>().commit_trim::<m![Joined%8]>().commit();
    let mut q:DmTensor<bf16,Chip,Cluster,StoreParts,m![P128%32,OutputGroup%8,Joined=8]>=DmTensor::new();
    result.view().tile::<m![Joined],8,m![P128,Joined=8#16]>(0).to_dm_view(&mut device.tdma,q.view_mut());
    q.view().to_hbm_view(&mut device.tdma,qout.view_mut());
    let mut k:DmTensor<bf16,Chip,Cluster,StoreParts,m![P128%32,OutputGroup%8,Joined=4]>=DmTensor::new();
    result.view().tile::<m![Joined],4,m![P128,Joined=4#16]>(8).to_dm_view(&mut device.tdma,k.view_mut());
    k.view().to_hbm_view(&mut device.tdma,kout.view_mut());
    let mut v:DmTensor<bf16,Chip,Cluster,StoreParts,m![P128%32,OutputGroup%8,Joined=4]>=DmTensor::new();
    result.view().tile::<m![Joined],4,m![P128,Joined=4#16]>(12).to_dm_view(&mut device.tdma,v.view_mut());
    v.view().to_hbm_view(&mut device.tdma,vout.view_mut());
}

#[device(chip=1)]
pub fn project_sliding_qkv_four(device:&mut Device,
    a:&HbmTensor<f8e4m3,Chip,m![P128,H]>,sa:&HbmTensor<f32,Chip,m![P128,1#8]>,
    b:&HbmTensor<f8e4m3,Chip,m![P128,H]>,sb:&HbmTensor<f32,Chip,m![P128,1#8]>,
    c:&HbmTensor<f8e4m3,Chip,m![P128,H]>,sc:&HbmTensor<f32,Chip,m![P128,1#8]>,
    d:&HbmTensor<f8e4m3,Chip,m![P128,H]>,sd:&HbmTensor<f32,Chip,m![P128,1#8]>,
    qw:&HbmTensor<f8e4m3,Chip,m![Qs,H]>,kw:&HbmTensor<f8e4m3,Chip,m![Ps,H]>,vw:&HbmTensor<f8e4m3,Chip,m![Ps,H]>,
    qs:&HbmTensor<bf16,Chip,m![Qs]>,ks:&HbmTensor<bf16,Chip,m![Ps]>,vs:&HbmTensor<bf16,Chip,m![Ps]>,
    oa:(&mut HbmTensor<bf16,Chip,QueryOutput>,&mut HbmTensor<bf16,Chip,KeyOutput>,&mut HbmTensor<bf16,Chip,KeyOutput>),
    ob:(&mut HbmTensor<bf16,Chip,QueryOutput>,&mut HbmTensor<bf16,Chip,KeyOutput>,&mut HbmTensor<bf16,Chip,KeyOutput>),
    oc:(&mut HbmTensor<bf16,Chip,QueryOutput>,&mut HbmTensor<bf16,Chip,KeyOutput>,&mut HbmTensor<bf16,Chip,KeyOutput>),
    od:(&mut HbmTensor<bf16,Chip,QueryOutput>,&mut HbmTensor<bf16,Chip,KeyOutput>,&mut HbmTensor<bf16,Chip,KeyOutput>)) {
    let (qa,ka,va)=oa;let(qb,kb,vb)=ob;let(qc,kc,vc)=oc;let(qd,kd,vd)=od;
    let qw:HbmTensorView<'_,f8e4m3,Chip,m![OutputHalf,OutputGroup,Joined=8,HBlock,H960]>=unsafe{qw.view().reshape()};
    let kw:HbmTensorView<'_,f8e4m3,Chip,m![OutputHalf,OutputGroup,Joined=4,HBlock,H960]>=unsafe{kw.view().reshape()};
    let vw:HbmTensorView<'_,f8e4m3,Chip,m![OutputHalf,OutputGroup,Joined=4,HBlock,H960]>=unsafe{vw.view().reshape()};
    let mut joined:DmTensor<f8e4m3,Chip,Cluster,Parts,m![Joined,HBlock,H960]>=DmTensor::new();
    qw.to_dm_view(&mut device.tdma,joined.view_mut().tile::<m![Joined],8,m![Joined=8#{!}16,HBlock,H960]>(0));
    kw.to_dm_view(&mut device.tdma,joined.view_mut().tile::<m![Joined],4,m![Joined=4#{!}16,HBlock,H960]>(8));
    vw.to_dm_view(&mut device.tdma,joined.view_mut().tile::<m![Joined],4,m![Joined=4#{!}16,HBlock,H960]>(12));
    let weight:TrfTensor<f8e4m3,Chip,Cluster,Parts,m![Joined%8],m![Joined/8,HBlock,H960]>=device.sub.begin(joined.view())
        .fetch::<m![Joined%8,Joined/8,HBlock,H960/32],m![H960%32]>()
        .collect::<m![Joined%8,Joined/8,HBlock,H960/32],m![H960%32]>().to_trf();
    let qs:HbmTensorView<'_,bf16,Chip,m![OutputHalf,OutputGroup,Joined=8]>=unsafe{qs.view().reshape()};
    let ks:HbmTensorView<'_,bf16,Chip,m![OutputHalf,OutputGroup,Joined=4]>=unsafe{ks.view().reshape()};
    let vs:HbmTensorView<'_,bf16,Chip,m![OutputHalf,OutputGroup,Joined=4]>=unsafe{vs.view().reshape()};
    let mut scales:DmTensor<bf16,Chip,Cluster,Parts,m![Joined]>=DmTensor::new();
    qs.to_dm_view(&mut device.tdma,scales.view_mut().tile::<m![Joined],8,m![Joined=8#{!}16]>(0));
    ks.to_dm_view(&mut device.tdma,scales.view_mut().tile::<m![Joined],4,m![Joined=4#{!}16]>(8));
    vs.to_dm_view(&mut device.tdma,scales.view_mut().tile::<m![Joined],4,m![Joined=4#{!}16]>(12));
    let scales:VrfTensor<f32,Chip,Cluster,Parts,m![Joined]>=device.sub.begin(scales.view())
        .fetch::<m![1],m![Joined]>().fetch_cast::<f32>()
        .collect::<m![Joined/8],m![Joined%8]>().to_vrf();
    one(device,a,sa,&weight,&scales,qa,ka,va);
    one(device,b,sb,&weight,&scales,qb,kb,vb);
    one(device,c,sc,&weight,&scales,qc,kc,vc);
    one(device,d,sd,&weight,&scales,qd,kd,vd);
}
