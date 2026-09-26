use furiosa_opt_std::prelude::*;
use crate::{Chip, axes::{H, L, Dummy2}};
axes![Term = 3, Row = 30, SliceIndex = 256, Projection = 2];
type Clusters = m![Dummy2];
type Slices = m![SliceIndex];

fn materialized_rows(
    device: &mut Device,
    x: &TrfTensor<f8e5m2, Chip, Clusters, Slices, m![1], m![Term, H]>,
    scale: &VrfTensor<f32, Chip, Clusters, Slices, m![1 # 8]>,
    w: &HbmTensor<f4e2m1, Chip, m![L,H]>,
    bs: &HbmTensor<f8e4m3, Chip, m![L,H/16]>,
) -> DmTensor<f32, Chip, Clusters, Slices, m![Row]> {
    let w: HbmTensorView<'_, f4e2m1, Chip, m![Dummy2, SliceIndex, Row, H]> = unsafe { w.view().reshape() };
    let w: DmTensor<f4e2m1, Chip, Clusters, Slices, m![Row, H]> = w.to_dm(&mut device.tdma);
    let w: DmTensor<f8e5m2, Chip, Clusters, Slices, m![Row, H]> = device.main
        .begin(w.view()).fetch::<m![Row], m![H]>()
        .fetch_table_lookup::<f8e5m2>()
        .collect::<m![Row, H / 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>().commit();
    let partials: DmTensor<f32, Chip, Clusters, Slices, m![Row, H / 16]> = device.main
        .begin(w.view()).fetch::<m![Row], m![H]>()
        .collect::<m![Row, H / 32], m![H % 32]>()
        .contract_outer::<m![Row, H / 64, Term], m![H % 64], _, _, _>(x)
        .contract_packet::<m![H / 16 % 4]>()
        .contract_time::<m![Row, H / 64]>()
        .contract_lane::<m![Row, H / 64], m![H / 16 % 4 # 8]>(LaneMode::Sequential)
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![H / 16 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scale)
        .vector_widen_pad::<m![H / 16 % 4 # 8]>().vector_final()
        .commit_trim::<m![H / 16 % 4]>().commit();
    let bs: HbmTensorView<'_, f8e4m3, Chip, m![Dummy2, SliceIndex, Row, H / 16]> = unsafe { bs.view().reshape() };
    let bs: DmTensor<f8e4m3, Chip, Clusters, Slices, m![Row, H / 16]> = bs.to_dm(&mut device.tdma);
    let mut output: DmTensor<f32, Chip, Clusters, Slices, m![Row]> = DmTensor::new();
    macro_rules! reduce_tile { ($i:expr) => {{
        let bs: VrfTensor<f32, Chip, Clusters, Slices, m![Row = 6, H / 16]> = device.sub
            .begin(bs.view().tile::<m![Row], 6, m![Row = 6 # 30, H / 16]>(6*$i))
            .fetch::<m![Row = 6], m![H / 16]>().fetch_cast::<f32>()
            .collect::<m![Row = 6, H / 16 / 8], m![H / 16 % 8]>().to_vrf();
        device.main.begin(partials.view().tile::<m![Row], 6, m![Row = 6 # 30, H / 16]>(6*$i))
            .fetch::<m![Row = 6, H / 16 / 8], m![H / 16 % 8]>()
            .collect::<m![Row = 6, H / 16 / 8], m![H / 16 % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Row = 6, H / 16 / 4], m![H / 16 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &bs)
            .vector_intra_slice_reduce::<H, m![Row = 6], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>().vector_final()
            .transpose::<m![Row = 6 / 2], m![Row = 6 % 2 # 8]>()
            .commit_trim::<m![Row = 6 % 2]>()
            .commit_view(output.view_mut().tile::<m![Row], 6, m![Row = 6 #{!} 30]>(6*$i));
    }}; }
    reduce_tile!(0); reduce_tile!(1); reduce_tile!(2); reduce_tile!(3); reduce_tile!(4);
    output
}

#[device(chip = 1)]
pub fn materialized(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    w: &HbmTensor<f4e2m1, Chip, m![L,H]>,
    g: &HbmTensor<f4e2m1, Chip, m![L,H]>,
    bs: &HbmTensor<f8e4m3, Chip, m![L,H/16]>,
    gs: &HbmTensor<f8e4m3, Chip, m![L,H/16]>,
    y: &mut HbmTensor<f32, Chip, m![L]>,
    z: &mut HbmTensor<f32, Chip, m![L]>,
) {
    let x: DmTensor<bf16, Chip, m![2], crate::device::layout::Slice, m![H]> = x.to_dm(&mut device.tdma);
    let (x, scale) = crate::device::mlp::prepare_ffn_fp8_split(device, &x);
    let x: DmTensor<f8e5m2, Chip, Clusters, Slices, m![Term, H]> = unsafe { x.reshape() };
    let scale: DmTensor<f32, Chip, Clusters, Slices, m![1 # 8]> = unsafe { scale.reshape() };
    let up = {
        let scale_vrf: VrfTensor<f32, Chip, Clusters, Slices, m![1 # 8]> = device.sub
            .begin(scale.view()).fetch::<m![1],m![1 # 8]>().collect::<m![1],m![1 # 8]>().to_vrf();
        let x_trf: TrfTensor<f8e5m2, Chip, Clusters, Slices, m![1],m![Term,H]> = device.sub
            .begin(x.view()).fetch::<m![Term],m![H]>().collect::<m![Term,H/32],m![H%32]>().to_trf();
        materialized_rows(device, &x_trf, &scale_vrf, w, bs)
    };
    let gate = {
        let scale_vrf: VrfTensor<f32, Chip, Clusters, Slices, m![1 # 8]> = device.sub
            .begin(scale.view()).fetch::<m![1],m![1 # 8]>().collect::<m![1],m![1 # 8]>().to_vrf();
        let x_trf: TrfTensor<f8e5m2, Chip, Clusters, Slices, m![1],m![Term,H]> = device.sub
            .begin(x.view()).fetch::<m![Term],m![H]>().collect::<m![Term,H/32],m![H%32]>().to_trf();
        materialized_rows(device, &x_trf, &scale_vrf, g, gs)
    };
    let up: DmTensor<f32, Chip, m![L / 7680], m![L / 30 % 256], m![L % 30]> = unsafe { up.reshape() };
    let gate: DmTensor<f32, Chip, m![L / 7680], m![L / 30 % 256], m![L % 30]> = unsafe { gate.reshape() };
    up.view().to_hbm_view(&mut device.tdma, y.view_mut());
    gate.view().to_hbm_view(&mut device.tdma, z.view_mut());
}

fn reduce_joint_rows(
    device: &mut Device,
    partials: DmTensorView<'_, f32, Chip, Clusters, Slices, m![Projection = 1 # 2, Row, H / 16]>,
    bs: &HbmTensor<f8e4m3, Chip, m![L,H/16]>,
) -> DmTensor<f32, Chip, Clusters, Slices, m![Row]> {
    let bs: HbmTensorView<'_, f8e4m3, Chip, m![Dummy2, SliceIndex, Row, H / 16]> = unsafe { bs.view().reshape() };
    let bs: DmTensor<f8e4m3, Chip, Clusters, Slices, m![Row, H / 16]> = bs.to_dm(&mut device.tdma);
    let mut output: DmTensor<f32, Chip, Clusters, Slices, m![Row]> = DmTensor::new();
    macro_rules! reduce_tile { ($i:expr) => {{
        let bs: VrfTensor<f32, Chip, Clusters, Slices, m![Row = 6, H / 16]> = device.sub
            .begin(bs.view().tile::<m![Row], 6, m![Row = 6 # 30, H / 16]>(6*$i))
            .fetch::<m![Row = 6], m![H / 16]>().fetch_cast::<f32>()
            .collect::<m![Row = 6, H / 16 / 8], m![H / 16 % 8]>().to_vrf();
        device.main.begin(partials.tile::<m![Row], 6, m![Projection = 1 # 2, Row = 6 # 30, H / 16]>(6*$i))
            .fetch::<m![Row = 6, H / 16 / 8], m![H / 16 % 8]>()
            .collect::<m![Row = 6, H / 16 / 8], m![H / 16 % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Row = 6, H / 16 / 4], m![H / 16 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &bs)
            .vector_intra_slice_reduce::<H, m![Row = 6], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>().vector_final()
            .transpose::<m![Row = 6 / 2], m![Row = 6 % 2 # 8]>()
            .commit_trim::<m![Row = 6 % 2]>()
            .commit_view(output.view_mut().tile::<m![Row], 6, m![Row = 6 #{!} 30]>(6*$i));
    }}; }
    reduce_tile!(0); reduce_tile!(1); reduce_tile!(2); reduce_tile!(3); reduce_tile!(4);
    output
}

#[device(chip = 1)]
pub fn fused(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    w: &HbmTensor<f4e2m1, Chip, m![L,H]>,
    g: &HbmTensor<f4e2m1, Chip, m![L,H]>,
    bs: &HbmTensor<f8e4m3, Chip, m![L,H/16]>,
    gs: &HbmTensor<f8e4m3, Chip, m![L,H/16]>,
    y: &mut HbmTensor<f32, Chip, m![L]>,
    z: &mut HbmTensor<f32, Chip, m![L]>,
) {
    let x: DmTensor<bf16, Chip, m![2], crate::device::layout::Slice, m![H]> = x.to_dm(&mut device.tdma);
    let (x, scale) = crate::device::mlp::prepare_ffn_fp8_split(device, &x);
    let x: DmTensor<f8e5m2, Chip, Clusters, Slices, m![Term, H]> = unsafe { x.reshape() };
    let scale: DmTensor<f32, Chip, Clusters, Slices, m![1 # 8]> = unsafe { scale.reshape() };
    let scale: VrfTensor<f32, Chip, Clusters, Slices, m![1 # 8]> = device.sub
        .begin(scale.view()).fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>().to_vrf();
    let x: TrfTensor<f8e5m2, Chip, Clusters, Slices, m![1], m![Term, H]> = device.sub
        .begin(x.view()).fetch::<m![Term], m![H]>()
        .collect::<m![Term, H / 32], m![H % 32]>().to_trf();
    let w: HbmTensorView<'_, f4e2m1, Chip, m![Dummy2, SliceIndex, Row, H]> = unsafe { w.view().reshape() };
    let g: HbmTensorView<'_, f4e2m1, Chip, m![Dummy2, SliceIndex, Row, H]> = unsafe { g.view().reshape() };
    let mut weights: DmTensor<f4e2m1, Chip, Clusters, Slices, m![Projection,Row,H]> = DmTensor::new();
    w.to_dm_view(&mut device.tdma,weights.view_mut().tile::<m![Projection],1,m![Projection=1#{!}2,Row,H]>(0));
    g.to_dm_view(&mut device.tdma,weights.view_mut().tile::<m![Projection],1,m![Projection=1#{!}2,Row,H]>(1));
    let partials: DmTensor<f32, Chip, Clusters, Slices, m![Projection, Row, H/16]> = device.main
        .begin(weights.view())
        .fetch::<m![Projection, Row],m![H]>()
        .fetch_table_lookup::<f8e5m2>()
        .collect::<m![Projection, Row, H / 32], m![H % 32]>()
        .contract_outer::<m![Projection, Row, H / 64, Term],m![H % 64],_,_,_>(&x)
        .contract_packet::<m![H / 16 % 4]>()
        .contract_time::<m![Projection, Row, H / 64]>()
        .contract_lane::<m![Projection, Row, H / 64],m![H / 16 % 4 # 8]>(LaneMode::Sequential)
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![H / 16 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&scale)
        .vector_widen_pad::<m![H / 16 % 4 # 8]>().vector_final()
        .commit_trim::<m![H / 16 % 4]>().commit();
    let up = reduce_joint_rows(device,partials.view().tile::<m![Projection],1,m![Projection=1#2,Row,H/16]>(0),bs);
    let gate = reduce_joint_rows(device,partials.view().tile::<m![Projection],1,m![Projection=1#2,Row,H/16]>(1),gs);
    let up: DmTensor<f32, Chip, m![L / 7680], m![L / 30 % 256], m![L % 30]> = unsafe { up.reshape() };
    let gate: DmTensor<f32, Chip, m![L / 7680], m![L / 30 % 256], m![L % 30]> = unsafe { gate.reshape() };
    up.view().to_hbm_view(&mut device.tdma, y.view_mut());
    gate.view().to_hbm_view(&mut device.tdma, z.view_mut());
}

#[device(chip = 1)]
pub fn decode(
    device: &mut Device,
    w: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    y: &mut HbmTensor<f8e5m2, Chip, m![L, H]>,
) {
    let w: HbmTensorView<'_, f4e2m1, Chip, m![Dummy2, SliceIndex, Row, H]> = unsafe { w.view().reshape() };
    let w: DmTensor<f4e2m1, Chip, Clusters, Slices, m![Row, H]> = w.to_dm(&mut device.tdma);
    let w: DmTensor<f8e5m2, Chip, Clusters, Slices, m![Row, H]> = device.main
        .begin(w.view()).fetch::<m![Row], m![H]>()
        .fetch_table_lookup::<f8e5m2>()
        .collect::<m![Row, H / 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>().commit();
    let w: DmTensor<f8e5m2, Chip, m![L / 7680], m![L / 30 % 256], m![L % 30, H]> = unsafe { w.reshape() };
    w.view().to_hbm_view(&mut device.tdma, y.view_mut());
}


fn first_row_partials(
    device: &mut Device,
    x_dm: &DmTensor<f8e5m2, Chip, Clusters, Slices, m![Term, H]>,
    scale_dm: &DmTensor<f32, Chip, Clusters, Slices, m![1 # 8]>,
    w: &HbmTensor<f4e2m1, Chip, m![L,H]>,
) -> DmTensor<f32, Chip, Clusters, Slices, m![Row = 1, H / 16]> {
    // Compact one valid row from every slice; do not narrow to the observed16 errors.
    let w: HbmTensorView<'_, f4e2m1, Chip, m![Dummy2, SliceIndex, Row, H]> = unsafe { w.view().reshape() };
    let w: DmTensor<f4e2m1, Chip, Clusters, Slices, m![Row = 1, H]> =
        w.tile::<m![Row],1,m![Dummy2, SliceIndex, Row = 1 # 30, H]>(0).to_dm(&mut device.tdma);
    let w: DmTensor<f8e5m2, Chip, Clusters, Slices, m![Row = 1, H]> = device.main
        .begin(w.view()).fetch::<m![Row = 1], m![H]>()
        .fetch_table_lookup::<f8e5m2>()
        .collect::<m![Row = 1, H / 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>().commit();
    // Separate reload after retaining the actual runtime quantized values.
    let x: TrfTensor<f8e5m2, Chip, Clusters, Slices, m![1], m![Term, H]> = device.sub
        .begin(x_dm.view()).fetch::<m![Term],m![H]>()
        .collect::<m![Term,H / 32],m![H % 32]>().to_trf();
    let scale: VrfTensor<f32, Chip, Clusters, Slices, m![1 # 8]> = device.sub
        .begin(scale_dm.view()).fetch::<m![1],m![1 # 8]>()
        .collect::<m![1],m![1 # 8]>().to_vrf();
    device.main.begin(w.view()).fetch::<m![Row = 1],m![H]>()
        .collect::<m![Row = 1,H / 32],m![H % 32]>()
        .contract_outer::<m![Row = 1,H / 64,Term],m![H % 64],_,_,_>(&x)
        .contract_packet::<m![H / 16 % 4]>()
        .contract_time::<m![Row = 1,H / 64]>()
        .contract_lane::<m![Row = 1,H / 64],m![H / 16 % 4 # 8]>(LaneMode::Sequential)
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![H / 16 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&scale)
        .vector_widen_pad::<m![H / 16 % 4 # 8]>().vector_final()
        .commit_trim::<m![H / 16 % 4]>().commit()
}

#[device(chip = 1)]
pub fn hybrid(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    w: &HbmTensor<f4e2m1, Chip, m![L,H]>,
    g: &HbmTensor<f4e2m1, Chip, m![L,H]>,
    bs: &HbmTensor<f8e4m3, Chip, m![L,H/16]>,
    gs: &HbmTensor<f8e4m3, Chip, m![L,H/16]>,
    y: &mut HbmTensor<f32, Chip, m![L]>,
    z: &mut HbmTensor<f32, Chip, m![L]>,
) {
    let x: DmTensor<bf16, Chip, m![2], crate::device::layout::Slice, m![H]> = x.to_dm(&mut device.tdma);
    let (x, scale) = crate::device::mlp::prepare_ffn_fp8_split(device, &x);
    let x_dm: DmTensor<f8e5m2, Chip, Clusters, Slices, m![Term, H]> = unsafe { x.reshape() };
    let scale_dm: DmTensor<f32, Chip, Clusters, Slices, m![1 # 8]> = unsafe { scale.reshape() };
    let scale: VrfTensor<f32, Chip, Clusters, Slices, m![1 # 8]> = device.sub
        .begin(scale_dm.view()).fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>().to_vrf();
    let x: TrfTensor<f8e5m2, Chip, Clusters, Slices, m![1], m![Term, H]> = device.sub
        .begin(x_dm.view()).fetch::<m![Term], m![H]>()
        .collect::<m![Term, H / 32], m![H % 32]>().to_trf();
    let first_row_weight = w;
    let weights = load_joint_compact(device,w,g);
    let mut partials: DmTensor<f32, Chip, Clusters, Slices, m![Projection, Row, H/16]> = device.main
        .begin(weights.view())
        .fetch::<m![Projection, Row],m![H]>()
        .fetch_table_lookup::<f8e5m2>()
        .collect::<m![Projection, Row, H / 32], m![H % 32]>()
        .contract_outer::<m![Projection, Row, H / 64, Term],m![H % 64],_,_,_>(&x)
        .contract_packet::<m![H / 16 % 4]>()
        .contract_time::<m![Projection, Row, H / 64]>()
        .contract_lane::<m![Projection, Row, H / 64],m![H / 16 % 4 # 8]>(LaneMode::Sequential)
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![H / 16 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&scale)
        .vector_widen_pad::<m![H / 16 % 4 # 8]>().vector_final()
        .commit_trim::<m![H / 16 % 4]>().commit();
    let replacement = first_row_partials(device, &x_dm, &scale_dm, first_row_weight);
    // Overwrite960B/slice, not a4B final scalar. Both complete direct projections remain.
    device.main.begin(replacement.view())
        .fetch::<m![H / 128],m![H / 16 % 8]>()
        .collect::<m![H / 128],m![H / 16 % 8]>()
        .commit_trim::<m![H / 16 % 8]>()
        .commit_view(partials.view_mut()
            .tile::<m![Projection],1,m![Projection=1#{!}2,Row,H/16]>(0)
            .tile::<m![Row],1,m![Projection=1#{!}2,Row=1#{!}30,H/16]>(0));
    let up = reduce_joint_rows(device,partials.view().tile::<m![Projection],1,m![Projection=1#2,Row,H/16]>(0),bs);
    let gate = reduce_joint_rows(device,partials.view().tile::<m![Projection],1,m![Projection=1#2,Row,H/16]>(1),gs);
    let up: DmTensor<f32, Chip, m![L / 7680], m![L / 30 % 256], m![L % 30]> = unsafe { up.reshape() };
    let gate: DmTensor<f32, Chip, m![L / 7680], m![L / 30 % 256], m![L % 30]> = unsafe { gate.reshape() };
    up.view().to_hbm_view(&mut device.tdma, y.view_mut());
    gate.view().to_hbm_view(&mut device.tdma, z.view_mut());
}


// Public HBM operands are compact matrices. Never DMA directly into Projection-axis
// padded views: the old path emitted twice the logical per-matrix byte count.
fn load_joint_compact(
    device: &mut Device,
    w: &HbmTensor<f4e2m1, Chip, m![L,H]>,
    g: &HbmTensor<f4e2m1, Chip, m![L,H]>,
) -> DmTensor<f4e2m1, Chip, Clusters, Slices, m![Projection,Row,H]> {
    let w: HbmTensorView<'_, f4e2m1, Chip, m![Dummy2,SliceIndex,Row,H]> = unsafe {w.view().reshape()};
    let g: HbmTensorView<'_, f4e2m1, Chip, m![Dummy2,SliceIndex,Row,H]> = unsafe {g.view().reshape()};
    let w: DmTensor<f4e2m1,Chip,Clusters,Slices,m![Row,H]> = w.to_dm(&mut device.tdma);
    let g: DmTensor<f4e2m1,Chip,Clusters,Slices,m![Row,H]> = g.to_dm(&mut device.tdma);
    let mut weights: DmTensor<f4e2m1,Chip,Clusters,Slices,m![Projection,Row,H]> = DmTensor::new();
    device.main.begin(w.view()).fetch::<m![Row,H/64],m![H%64]>()
        .collect::<m![Row,H/64],m![H%64]>().commit_trim::<m![H%64]>()
        .commit_view(weights.view_mut().tile::<m![Projection],1,m![Projection=1#{!}2,Row,H]>(0));
    device.main.begin(g.view()).fetch::<m![Row,H/64],m![H%64]>()
        .collect::<m![Row,H/64],m![H%64]>().commit_trim::<m![H%64]>()
        .commit_view(weights.view_mut().tile::<m![Projection],1,m![Projection=1#{!}2,Row,H]>(1));
    weights
}

#[device(chip = 1)]
pub fn direct_fixed(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    w: &HbmTensor<f4e2m1, Chip, m![L,H]>,
    g: &HbmTensor<f4e2m1, Chip, m![L,H]>,
    bs: &HbmTensor<f8e4m3, Chip, m![L,H/16]>,
    gs: &HbmTensor<f8e4m3, Chip, m![L,H/16]>,
    y: &mut HbmTensor<f32, Chip, m![L]>,
    z: &mut HbmTensor<f32, Chip, m![L]>,
) {
    let x: DmTensor<bf16, Chip, m![2], crate::device::layout::Slice, m![H]> = x.to_dm(&mut device.tdma);
    let (x, scale) = crate::device::mlp::prepare_ffn_fp8_split(device, &x);
    let x: DmTensor<f8e5m2, Chip, Clusters, Slices, m![Term, H]> = unsafe { x.reshape() };
    let scale: DmTensor<f32, Chip, Clusters, Slices, m![1 # 8]> = unsafe { scale.reshape() };
    let scale: VrfTensor<f32, Chip, Clusters, Slices, m![1 # 8]> = device.sub
        .begin(scale.view()).fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>().to_vrf();
    let x: TrfTensor<f8e5m2, Chip, Clusters, Slices, m![1], m![Term, H]> = device.sub
        .begin(x.view()).fetch::<m![Term], m![H]>()
        .collect::<m![Term, H / 32], m![H % 32]>().to_trf();
    let weights = load_joint_compact(device,w,g);
    let partials: DmTensor<f32, Chip, Clusters, Slices, m![Projection, Row, H/16]> = device.main
        .begin(weights.view())
        .fetch::<m![Projection, Row],m![H]>()
        .fetch_table_lookup::<f8e5m2>()
        .collect::<m![Projection, Row, H / 32], m![H % 32]>()
        .contract_outer::<m![Projection, Row, H / 64, Term],m![H % 64],_,_,_>(&x)
        .contract_packet::<m![H / 16 % 4]>()
        .contract_time::<m![Projection, Row, H / 64]>()
        .contract_lane::<m![Projection, Row, H / 64],m![H / 16 % 4 # 8]>(LaneMode::Sequential)
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![H / 16 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&scale)
        .vector_widen_pad::<m![H / 16 % 4 # 8]>().vector_final()
        .commit_trim::<m![H / 16 % 4]>().commit();
    let up = reduce_joint_rows(device,partials.view().tile::<m![Projection],1,m![Projection=1#2,Row,H/16]>(0),bs);
    let gate = reduce_joint_rows(device,partials.view().tile::<m![Projection],1,m![Projection=1#2,Row,H/16]>(1),gs);
    let up: DmTensor<f32, Chip, m![L / 7680], m![L / 30 % 256], m![L % 30]> = unsafe { up.reshape() };
    let gate: DmTensor<f32, Chip, m![L / 7680], m![L / 30 % 256], m![L % 30]> = unsafe { gate.reshape() };
    up.view().to_hbm_view(&mut device.tdma, y.view_mut());
    gate.view().to_hbm_view(&mut device.tdma, z.view_mut());
}
