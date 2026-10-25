//! Finite-BF16 exact-rational symmetric A4, exposed as ordinary I32 codes.
//! Code selection uses no floating division. This is an isolated foundation,
//! not the older rounded-DivF quantizer and not a model dispatch path.
use furiosa_opt_std::prelude::*;
use crate::{Chip, axes::{B, H}};

axes![Threshold = 8];
type Cluster = m![B / 8];
type Row = m![B % 8 # 256];
type Shards = m![B % 8, H / 960, 1 # 8];
type Comparisons = m![B % 8, H / 960, Threshold];

/// `odds` must contain exactly [1,3,5,7,9,11,13,15] in the live lane of
/// each padded eight-I32 record (aligned per-slice HBM transfer). The last threshold
/// cannot pass when A>=abs(x), so all eight lanes can use the same reduction.
#[device(chip = 1)]
pub fn quantize_row(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![B, H]>,
    odds: &HbmTensor<i32, Chip, m![Threshold, 1 # 8]>,
    output: &mut HbmTensor<i32, Chip, m![B, H]>,
    scales: &mut HbmTensor<f32, Chip, m![B]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, Row, m![H]> = input.to_dm(&mut device.tdma);
    // Established finite absolute-Max reduction; the floor is exactly 2^-112.
    let amax: DmTensor<f32, Chip, Cluster, Row, m![1 # 8]> = device.main.begin(x.view())
        .fetch::<m![H / 16], m![H % 16]>().fetch_cast::<f32>()
        .collect::<m![H / 8], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd, 0x7fff_ffffi32)
        .vector_reinterpret::<f32>()
        .vector_narrow_split::<m![H / 4], m![H % 4]>()
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, 1.925929944387236e-34f32)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    let scale: DmTensor<f32, Chip, Cluster, Row, m![1 # 8]> = device.main.begin(amax.view())
        .fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.1428571492433548f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    let exponent: DmTensor<i32, Chip, Cluster, Row, m![1 # 8]> = device.main.begin(amax.view())
        .fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::LogicRightShift, 23i32)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    let mantissa: DmTensor<i32, Chip, Cluster, Row, m![1 # 8]> = device.main.begin(amax.view())
        .fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::LogicRightShift, 16i32)
        .vector_logic(LogicBinaryOpI32::BitAnd, 127i32)
        .vector_logic(LogicBinaryOpI32::BitOr, 128i32)
        .vector_final().commit_trim::<m![1 # 8]>().commit();

    let xs: DmTensor<bf16, Chip, Cluster, Shards, m![H % 960]> = x.to_dm(&mut device.tdma);
    let es: DmTensor<i32, Chip, Cluster, Shards, m![1 # 8]> = exponent.to_dm(&mut device.tdma);
    let ea: VrfTensor<i32, Chip, Cluster, Shards, m![1 # 8]> = device.sub.begin(es.view())
        .fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let delta: DmTensor<i32, Chip, Cluster, Shards, m![H % 960]> = device.main.begin(xs.view())
        .fetch::<m![H % 960 / 16], m![H % 16]>().fetch_cast::<f32>()
        .collect::<m![H % 960 / 8], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::LogicRightShift, 23i32)
        .vector_logic(LogicBinaryOpI32::BitAnd, 255i32)
        .vector_fxp_with_mode(FxpBinaryOp::SubFxp, BinaryArgMode::Mode10, &ea)
        .vector_clip(ClipBinaryOpI32::Min, 5i32)
        .vector_final().commit_trim::<m![H % 8]>().commit();
    let numerator: DmTensor<i32, Chip, Cluster, Shards, m![H % 960]> = device.main.begin(xs.view())
        .fetch::<m![H % 960 / 16], m![H % 16]>().fetch_cast::<f32>()
        .collect::<m![H % 960 / 8], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::LogicRightShift, 16i32)
        .vector_logic(LogicBinaryOpI32::BitAnd, 127i32)
        .vector_logic(LogicBinaryOpI32::BitOr, 128i32)
        .vector_fxp(FxpBinaryOp::MulInt, 14i32)
        .vector_clip(ClipBinaryOpI32::AddFxp, 1i32)
        .vector_final().commit_trim::<m![H % 8]>().commit();

    // Real replication into eight independent threshold slices, not a dtype or
    // physical-layout reinterpretation. Two H960 VRFs plus two scalar VRFs use
    // 7,744 bytes per PE in this pass.
    let nd: DmTensor<i32, Chip, Cluster, Comparisons, m![H % 960]> = numerator.to_dm(&mut device.tdma);
    let dd: DmTensor<i32, Chip, Cluster, Comparisons, m![H % 960]> = delta.to_dm(&mut device.tdma);
    let md: DmTensor<i32, Chip, Cluster, Comparisons, m![1 # 8]> = mantissa.to_dm(&mut device.tdma);
    let od: DmTensor<i32, Chip, Cluster, Comparisons, m![1 # 8]> = odds.to_dm(&mut device.tdma);
    let n: VrfTensor<i32, Chip, Cluster, Comparisons, m![H % 960]> = device.sub.begin(nd.view())
        .fetch::<m![H % 960 / 8], m![H % 8]>().collect::<m![H % 960 / 8], m![H % 8]>().to_vrf();
    let d: VrfTensor<i32, Chip, Cluster, Comparisons, m![H % 960]> = device.sub.begin(dd.view())
        .fetch::<m![H % 960 / 8], m![H % 8]>().collect::<m![H % 960 / 8], m![H % 8]>().to_vrf();
    let m: VrfTensor<i32, Chip, Cluster, Comparisons, m![1 # 8]> = device.sub.begin(md.view())
        .fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let odd: VrfTensor<i32, Chip, Cluster, Comparisons, m![1 # 8]> = device.sub.begin(od.view())
        .fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let negative_magnitude: DmTensor<i32, Chip, Cluster, Shards, m![H % 960]> = device.main.begin(nd.view())
        .fetch::<m![H % 960 / 8], m![H % 8]>().collect::<m![H % 960 / 8], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_logic_with_mode(LogicBinaryOpI32::BitOr, BinaryArgMode::Mode11, &m)
        .vector_logic(LogicBinaryOpI32::LeftShift, &d)
        .vector_fxp(FxpBinaryOp::MulInt, &odd)
        .vector_fxp(FxpBinaryOp::SubFxp, &n)
        .vector_fxp(FxpBinaryOp::ArithRightShift, 31i32)
        .vector_inter_slice_reduce::<Shards, m![H % 960 / 8]>(InterSliceReduceOpI32::Add)
        .vector_final().commit_trim::<m![H % 8]>().commit();

    // Flip -magnitude only for positive inputs. Signed zero produces integer0.
    let flip: VrfTensor<i32, Chip, Cluster, Shards, m![H % 960]> = device.sub.begin(xs.view())
        .fetch::<m![H % 960 / 16], m![H % 16]>().fetch_cast::<f32>()
        .collect::<m![H % 960 / 8], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::ArithRightShift, 31i32)
        .vector_logic(LogicBinaryOpI32::BitXor, -1i32)
        .vector_final().to_vrf();
    let codes: DmTensor<i32, Chip, Cluster, Shards, m![H % 960]> = device.main.begin(negative_magnitude.view())
        .fetch::<m![H % 960 / 8], m![H % 8]>().collect::<m![H % 960 / 8], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpI32::BitXor, &flip)
        .vector_fxp(FxpBinaryOp::SubFxp, &flip)
        .vector_final().commit_trim::<m![H % 8]>().commit();
    codes.view().to_hbm_view(&mut device.tdma, output.view_mut());
    scale.view().to_hbm_view(&mut device.tdma, scales.view_mut());
}
