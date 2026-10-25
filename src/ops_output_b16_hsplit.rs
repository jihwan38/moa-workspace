//! Exact B16 output projection: K1024 partials run concurrently across slices.
//! Output H is split across both clusters; every partial and sum still rounds to BF16.
//! Gather four contiguous partial planes, then preserve the original BF16 addition tree.
use crate::{
    Chip,
    axes::{B, Df, Ds, Gf, Gs, H, Ns, Qf, Qs},
};
use furiosa_opt_std::prelude::*;
axes![Ho = 2, Hb = 30, Ht = 64];
type Cluster = m![Ho];
type Parts = m![Hb # 32,Qs/1024,1#2];
type FullParts = m![Hb # 32,Qf/1024];
type DownRows = m![Hb # 32,1#8];
fn add_partials_wide(
    device: &mut Device,
    left: &DmTensor<bf16, Chip, Cluster, DownRows, m![Ht, B]>,
    right: &DmTensor<bf16, Chip, Cluster, DownRows, m![Ht, B]>,
) -> DmTensor<bf16, Chip, Cluster, DownRows, m![Ht, B]> {
    let mut result: DmTensor<bf16, Chip, Cluster, DownRows, m![Ht, B]> = DmTensor::new();

    let left: VrfTensor<f32, Chip, Cluster, DownRows, m![Ht, B = 16]> = device
        .sub
        .begin(left.view().tile::<m![B], 16, m![Ht,B = 16 # 16]>(0))
        .fetch::<m![Ht], m![B = 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ht, B = 16 / 8], m![B = 16 % 8]>()
        .to_vrf();
    device
        .main
        .begin(right.view().tile::<m![B], 16, m![Ht,B = 16 # 16]>(0))
        .fetch::<m![Ht], m![B = 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ht, B = 16 / 8], m![B = 16 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &left)
        .vector_final()
        .cast::<bf16, m![B = 16 % 8 # 16]>()
        .commit_trim::<m![B = 16 % 8]>()
        .commit_view(
            result
                .view_mut()
                .tile::<m![B], 16, m![Ht,B = 16 #{!} 16]>(0),
        );

    result
}

fn add_gathered_pair(
    device: &mut Device,
    input: &DmTensor<bf16, Chip, Cluster, DownRows, m![Qs / 1024, Ht, B]>,
    offset: usize,
) -> DmTensor<bf16, Chip, Cluster, DownRows, m![Ht, B]> {
    let mut result: DmTensor<bf16, Chip, Cluster, DownRows, m![Ht, B]> = DmTensor::new();

    let left: VrfTensor<f32, Chip, Cluster, DownRows, m![Ht, B = 16]> = device
        .sub
        .begin(
            input
                .view()
                .tile::<m![Qs / 1024], 1, m![1 # 4,Ht,B]>(offset)
                .tile::<m![B], 16, m![Ht # 256,B = 16 # 16]>(0),
        )
        .fetch::<m![Ht], m![B = 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ht, B = 16 / 8], m![B = 16 % 8]>()
        .to_vrf();
    device
        .main
        .begin(
            input
                .view()
                .tile::<m![Qs / 1024], 1, m![1 # 4,Ht,B]>(offset + 1)
                .tile::<m![B], 16, m![Ht # 256,B = 16 # 16]>(0),
        )
        .fetch::<m![Ht], m![B = 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ht, B = 16 / 8], m![B = 16 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &left)
        .vector_final()
        .cast::<bf16, m![B = 16 % 8 # 16]>()
        .commit_trim::<m![B = 16 % 8]>()
        .commit_view(
            result
                .view_mut()
                .tile::<m![B], 16, m![Ht,B = 16 #{!} 16]>(0),
        );

    result
}

fn scale_output_wide(
    device: &mut Device,
    input: &DmTensor<bf16, Chip, Cluster, DownRows, m![Ht, B]>,
    scale: &HbmTensor<bf16, Chip, m![H]>,
    output: HbmTensorViewMut<'_, bf16, Chip, m![B, Ho, Hb, Ht]>,
) {
    let source: HbmTensorView<'_, bf16, Chip, m![Ho, Hb, Ht]> = unsafe { scale.view().reshape() };
    let scale: DmTensor<bf16, Chip, Cluster, DownRows, m![Ht]> = source.to_dm(&mut device.tdma);
    let scale: VrfTensor<f32, Chip, Cluster, DownRows, m![Ht]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![1], m![Ht]>()
        .fetch_cast::<f32>()
        .collect::<m![Ht / 8], m![Ht % 8]>()
        .to_vrf();
    let result: DmTensor<bf16, Chip, Cluster, DownRows, m![Ht, B]> = device
        .main
        .begin(input.view())
        .fetch::<m![Ht, B / 8], m![B % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Ht, B / 8], m![B % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ht, B / 4], m![B % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_widen_concat::<m![Ht, B / 8], m![B % 8]>()
        .vector_final()
        .cast::<bf16, m![B % 8 # 16]>()
        .commit_trim::<m![B % 8]>()
        .commit();
    let result: DmTensor<bf16, Chip, Cluster, DownRows, m![B, Ht]> = device
        .main
        .begin(result.view())
        .fetch::<m![B / 16, Ht], m![B % 16]>()
        .collect::<m![B / 16, Ht], m![B % 16]>()
        .transpose::<m![B / 16, Ht / 4, B % 16], m![Ht%4 # 16]>()
        .commit_trim::<m![Ht % 4]>()
        .commit();
    result.view().to_hbm_view(&mut device.tdma, output);
}

#[device(chip = 1)]
pub fn project_sliding_output_batch(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    scale: &HbmTensor<bf16, Chip, m![H]>,
    output: HbmTensorViewMut<'_, bf16, Chip, m![B, Ho, Hb, Ht]>,
) {
    let source: HbmTensorView<'_, f8e4m3, Chip, m![Ho, Hb, Ht, Qs]> =
        unsafe { weight.view().reshape() };
    let packed: DmTensor<f8e4m3, Chip, Cluster, Parts, m![Ht, Qs % 1024]> =
        source.to_dm(&mut device.tdma);
    let weight: DmTensor<bf16, Chip, Cluster, Parts, m![Ht, Qs % 1024]> = device
        .main
        .begin(packed.view())
        .fetch::<m![Ht, Qs / 32 % 32], m![Qs % 32]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Ht, Qs / 16 % 64], m![Qs % 16]>()
        .commit_trim::<m![Qs % 16]>()
        .commit();
    let flat: HbmTensorView<'_, bf16, Chip, m![B, Qs]> = unsafe { input.view().reshape() };
    let input: DmTensor<bf16, Chip, Cluster, Parts, m![B, Qs % 1024]> =
        flat.to_dm(&mut device.tdma);
    let mut partial: DmTensor<bf16, Chip, Cluster, Parts, m![Ht, B]> = DmTensor::new();
    for chunk in 0..2 {
        let x: TrfTensor<bf16, Chip, Cluster, Parts, m![B = 8], m![Qs % 1024]> = device
            .sub
            .begin(input.view().tile::<m![B], 8, m![B=8#16,Qs%1024]>(8 * chunk))
            .fetch::<m![B = 8, Qs / 16 % 64], m![Qs % 16]>()
            .collect::<m![B = 8, Qs / 16 % 64], m![Qs % 16]>()
            .to_trf();
        device
            .main
            .begin(weight.view())
            .fetch::<m![Ht, Qs / 16 % 64], m![Qs % 16]>()
            .collect::<m![Ht, Qs / 16 % 64], m![Qs % 16]>()
            .contract_outer::<m![Ht, Qs / 32 % 32], m![Qs % 32], _, _, _>(&x)
            .contract_packet::<m![1]>()
            .contract_time::<m![Ht]>()
            .contract_lane::<m![Ht], m![B = 8]>(LaneMode::Interleaved)
            .cast::<bf16, m![B=8#16]>()
            .commit_trim::<m![B = 8]>()
            .commit_view(
                partial
                    .view_mut()
                    .tile::<m![B], 8, m![Ht,B=8#{!}16]>(8 * chunk),
            );
    }
    let gathered: DmTensor<bf16, Chip, Cluster, DownRows, m![Qs / 1024, Ht, B]> =
        partial.to_dm(&mut device.tdma);
    let p01 = add_gathered_pair(device, &gathered, 0);
    let p23 = add_gathered_pair(device, &gathered, 2);
    let result = add_partials_wide(device, &p01, &p23);
    scale_output_wide(device, &result, scale, output);
}

#[device(chip = 1)]
pub fn project_sliding_partial_dots(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    output: HbmTensorViewMut<'_, bf16, Chip, m![B, Qs / 1024, Ho, Hb, Ht]>,
) {
    let source: HbmTensorView<'_, f8e4m3, Chip, m![Ho, Hb, Ht, Qs]> =
        unsafe { weight.view().reshape() };
    let packed: DmTensor<f8e4m3, Chip, Cluster, Parts, m![Ht, Qs % 1024]> =
        source.to_dm(&mut device.tdma);
    let weight: DmTensor<bf16, Chip, Cluster, Parts, m![Ht, Qs % 1024]> = device
        .main
        .begin(packed.view())
        .fetch::<m![Ht, Qs / 32 % 32], m![Qs % 32]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Ht, Qs / 16 % 64], m![Qs % 16]>()
        .commit_trim::<m![Qs % 16]>()
        .commit();
    let flat: HbmTensorView<'_, bf16, Chip, m![B, Qs]> = unsafe { input.view().reshape() };
    let input: DmTensor<bf16, Chip, Cluster, Parts, m![B, Qs % 1024]> =
        flat.to_dm(&mut device.tdma);
    let mut partial: DmTensor<bf16, Chip, Cluster, Parts, m![Ht, B]> = DmTensor::new();
    for chunk in 0..2 {
        let x: TrfTensor<bf16, Chip, Cluster, Parts, m![B = 8], m![Qs % 1024]> = device
            .sub
            .begin(input.view().tile::<m![B], 8, m![B=8#16,Qs%1024]>(8 * chunk))
            .fetch::<m![B = 8, Qs / 16 % 64], m![Qs % 16]>()
            .collect::<m![B = 8, Qs / 16 % 64], m![Qs % 16]>()
            .to_trf();
        device
            .main
            .begin(weight.view())
            .fetch::<m![Ht, Qs / 16 % 64], m![Qs % 16]>()
            .collect::<m![Ht, Qs / 16 % 64], m![Qs % 16]>()
            .contract_outer::<m![Ht, Qs / 32 % 32], m![Qs % 32], _, _, _>(&x)
            .contract_packet::<m![1]>()
            .contract_time::<m![Ht]>()
            .contract_lane::<m![Ht], m![B = 8]>(LaneMode::Interleaved)
            .cast::<bf16, m![B=8#16]>()
            .commit_trim::<m![B = 8]>()
            .commit_view(
                partial
                    .view_mut()
                    .tile::<m![B], 8, m![Ht,B=8#{!}16]>(8 * chunk),
            );
    }
    partial.view().to_hbm_view(&mut device.tdma, output);
}

fn add_gathered_pair_full(
    device: &mut Device,
    input: &DmTensor<bf16, Chip, Cluster, DownRows, m![Qf / 1024, Ht, B]>,
    offset: usize,
) -> DmTensor<bf16, Chip, Cluster, DownRows, m![Ht, B]> {
    let mut result: DmTensor<bf16, Chip, Cluster, DownRows, m![Ht, B]> = DmTensor::new();

    let left: VrfTensor<f32, Chip, Cluster, DownRows, m![Ht, B = 16]> = device
        .sub
        .begin(
            input
                .view()
                .tile::<m![Qf / 1024], 1, m![1 # 8,Ht,B]>(offset)
                .tile::<m![B], 16, m![Ht # 512,B = 16 # 16]>(0),
        )
        .fetch::<m![Ht], m![B = 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ht, B = 16 / 8], m![B = 16 % 8]>()
        .to_vrf();
    device
        .main
        .begin(
            input
                .view()
                .tile::<m![Qf / 1024], 1, m![1 # 8,Ht,B]>(offset + 1)
                .tile::<m![B], 16, m![Ht # 512,B = 16 # 16]>(0),
        )
        .fetch::<m![Ht], m![B = 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ht, B = 16 / 8], m![B = 16 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &left)
        .vector_final()
        .cast::<bf16, m![B = 16 % 8 # 16]>()
        .commit_trim::<m![B = 16 % 8]>()
        .commit_view(
            result
                .view_mut()
                .tile::<m![B], 16, m![Ht,B = 16 #{!} 16]>(0),
        );

    result
}

#[device(chip = 1)]
pub fn project_full_output_batch(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![B, Gf, Df]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qf]>,
    scale: &HbmTensor<bf16, Chip, m![H]>,
    output: HbmTensorViewMut<'_, bf16, Chip, m![B, Ho, Hb, Ht]>,
) {
    let source: HbmTensorView<'_, f8e4m3, Chip, m![Ho, Hb, Ht, Qf]> =
        unsafe { weight.view().reshape() };
    let packed: DmTensor<f8e4m3, Chip, Cluster, FullParts, m![Ht, Qf % 1024]> =
        source.to_dm(&mut device.tdma);
    let weight: DmTensor<bf16, Chip, Cluster, FullParts, m![Ht, Qf % 1024]> = device
        .main
        .begin(packed.view())
        .fetch::<m![Ht, Qf / 32 % 32], m![Qf % 32]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Ht, Qf / 16 % 64], m![Qf % 16]>()
        .commit_trim::<m![Qf % 16]>()
        .commit();
    let flat: HbmTensorView<'_, bf16, Chip, m![B, Qf]> = unsafe { input.view().reshape() };
    let input: DmTensor<bf16, Chip, Cluster, FullParts, m![B, Qf % 1024]> =
        flat.to_dm(&mut device.tdma);
    let mut partial: DmTensor<bf16, Chip, Cluster, FullParts, m![Ht, B]> = DmTensor::new();
    for chunk in 0..2 {
        let x: TrfTensor<bf16, Chip, Cluster, FullParts, m![B = 8], m![Qf % 1024]> = device
            .sub
            .begin(input.view().tile::<m![B], 8, m![B=8#16,Qf%1024]>(8 * chunk))
            .fetch::<m![B = 8, Qf / 16 % 64], m![Qf % 16]>()
            .collect::<m![B = 8, Qf / 16 % 64], m![Qf % 16]>()
            .to_trf();
        device
            .main
            .begin(weight.view())
            .fetch::<m![Ht, Qf / 16 % 64], m![Qf % 16]>()
            .collect::<m![Ht, Qf / 16 % 64], m![Qf % 16]>()
            .contract_outer::<m![Ht, Qf / 32 % 32], m![Qf % 32], _, _, _>(&x)
            .contract_packet::<m![1]>()
            .contract_time::<m![Ht]>()
            .contract_lane::<m![Ht], m![B = 8]>(LaneMode::Interleaved)
            .cast::<bf16, m![B=8#16]>()
            .commit_trim::<m![B = 8]>()
            .commit_view(
                partial
                    .view_mut()
                    .tile::<m![B], 8, m![Ht,B=8#{!}16]>(8 * chunk),
            );
    }
    let gathered: DmTensor<bf16, Chip, Cluster, DownRows, m![Qf / 1024, Ht, B]> =
        partial.to_dm(&mut device.tdma);
    let p01 = add_gathered_pair_full(device, &gathered, 0);
    let p23 = add_gathered_pair_full(device, &gathered, 2);
    let p45 = add_gathered_pair_full(device, &gathered, 4);
    let p67 = add_gathered_pair_full(device, &gathered, 6);
    let p0123 = add_partials_wide(device, &p01, &p23);
    let p4567 = add_partials_wide(device, &p45, &p67);
    let result = add_partials_wide(device, &p0123, &p4567);
    scale_output_wide(device, &result, scale, output);
}

#[device(chip = 1)]
pub fn project_full_partial_dots(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![B, Gf, Df]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qf]>,
    output: HbmTensorViewMut<'_, bf16, Chip, m![B, Qf / 1024, Ho, Hb, Ht]>,
) {
    let source: HbmTensorView<'_, f8e4m3, Chip, m![Ho, Hb, Ht, Qf]> =
        unsafe { weight.view().reshape() };
    let packed: DmTensor<f8e4m3, Chip, Cluster, FullParts, m![Ht, Qf % 1024]> =
        source.to_dm(&mut device.tdma);
    let weight: DmTensor<bf16, Chip, Cluster, FullParts, m![Ht, Qf % 1024]> = device
        .main
        .begin(packed.view())
        .fetch::<m![Ht, Qf / 32 % 32], m![Qf % 32]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Ht, Qf / 16 % 64], m![Qf % 16]>()
        .commit_trim::<m![Qf % 16]>()
        .commit();
    let flat: HbmTensorView<'_, bf16, Chip, m![B, Qf]> = unsafe { input.view().reshape() };
    let input: DmTensor<bf16, Chip, Cluster, FullParts, m![B, Qf % 1024]> =
        flat.to_dm(&mut device.tdma);
    let mut partial: DmTensor<bf16, Chip, Cluster, FullParts, m![Ht, B]> = DmTensor::new();
    for chunk in 0..2 {
        let x: TrfTensor<bf16, Chip, Cluster, FullParts, m![B = 8], m![Qf % 1024]> = device
            .sub
            .begin(input.view().tile::<m![B], 8, m![B=8#16,Qf%1024]>(8 * chunk))
            .fetch::<m![B = 8, Qf / 16 % 64], m![Qf % 16]>()
            .collect::<m![B = 8, Qf / 16 % 64], m![Qf % 16]>()
            .to_trf();
        device
            .main
            .begin(weight.view())
            .fetch::<m![Ht, Qf / 16 % 64], m![Qf % 16]>()
            .collect::<m![Ht, Qf / 16 % 64], m![Qf % 16]>()
            .contract_outer::<m![Ht, Qf / 32 % 32], m![Qf % 32], _, _, _>(&x)
            .contract_packet::<m![1]>()
            .contract_time::<m![Ht]>()
            .contract_lane::<m![Ht], m![B = 8]>(LaneMode::Interleaved)
            .cast::<bf16, m![B=8#16]>()
            .commit_trim::<m![B = 8]>()
            .commit_view(
                partial
                    .view_mut()
                    .tile::<m![B], 8, m![Ht,B=8#{!}16]>(8 * chunk),
            );
    }
    partial.view().to_hbm_view(&mut device.tdma, output);
}
