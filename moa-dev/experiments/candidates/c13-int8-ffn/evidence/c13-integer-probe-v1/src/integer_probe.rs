//! Diagnostic gate, not a competition kernel. Input digits are supplied by the
//! diagnostic caller; activation quantization belongs to the full C13 design.
use furiosa_opt_std::prelude::*;
use crate::Chip;
axes![Rows = 4, K = 64, Term = 2];
type Cluster = m![1 # 2];
type Slice = m![1 # 256];

#[device(chip = 1)]
pub fn integer_blocks(
    device: &mut Device,
    weight: &HbmTensor<f4e2m1, Chip, m![Rows, K]>,
    digits: &HbmTensor<i8, Chip, m![Term, K]>,
    decoded: &mut HbmTensor<i8, Chip, m![Rows, K]>,
    sums: &mut HbmTensor<f32, Chip, m![Rows, K / 16]>,
) {
    let packed: DmTensor<f4e2m1, Chip, Cluster, Slice, m![Rows, K]> = weight.to_dm(&mut device.tdma);
    // The fixed-point converter's fraction width is 31-int_width. Width 30
    // therefore computes round(2*w), exact for all sixteen FP4 E2M1 codes.
    let integer: DmTensor<i8, Chip, Cluster, Slice, m![Rows, K]> = device.main
        .begin(packed.view())
        .fetch::<m![Rows], m![K]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![Rows, K / 8], m![K % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fp_to_fxp(30)
        .vector_final()
        .cast::<i8, m![K % 8 # 32]>()
        .commit_trim::<m![K % 8]>()
        .commit();
    integer.view().to_hbm_view(&mut device.tdma, decoded.view_mut());
    let digits: DmTensor<i8, Chip, Cluster, Slice, m![Term, K]> = digits.to_dm(&mut device.tdma);
    let trf: TrfTensor<i8, Chip, Cluster, Slice, m![Term], m![K]> = device.sub
        .begin(digits.view())
        .fetch::<m![Term, K / 32], m![K % 32]>()
        .collect::<m![Term, K / 32], m![K % 32]>()
        .to_trf();
    let combined: DmTensor<f32, Chip, Cluster, Slice, m![Rows, K / 16]> = device.main
        .begin(integer.view())
        .fetch::<m![Rows, K / 32], m![K % 32]>()
        .collect::<m![Rows, K / 32], m![K % 32]>()
        .contract_outer::<m![Rows, K / 64], m![K % 64], _, _, _>(&trf)
        .contract_packet::<m![K / 16 % 4]>()
        .contract_time::<m![Rows, K / 64]>()
        .contract_lane::<m![Rows, K / 64, Term], m![K / 16 % 4 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_intra_slice_unzip::<Term, m![Rows, K / 64]>()
        .vector_fxp(FxpBinaryOp::LeftShift, 7i32, ())
        .vector_fxp_zip(FxpBinaryOp::AddFxp)
        .vector_fxp_to_fp(31)
        .vector_final()
        .commit_trim::<m![K / 16 % 4]>()
        .commit();
    combined.view().to_hbm_view(&mut device.tdma, sums.view_mut());
}
