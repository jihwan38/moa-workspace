"""Alternatives that remove block BF16 rounding or extend FP8 input range."""
from pathlib import Path


def f32_blocks(source, down=False):
    function = "fn project_native_down_rows(" if down else "fn project_rows_with_scale_contraction("
    a = source.index(function)
    b = source.index("type FfnQuantSlices" if down else "pub(crate) fn project_up_and_gate(", a)
    helper = source[a:b]
    helper = helper.replace(
        "    // This deliberately adds BF16 rounding of each 16-element block dot.\n"
        "    // The second contraction multiplies block scales and reduces the block axis.\n",
        "    // Preserve F32 block partials; the following VE passes apply block scales\n"
        "    // and reduce the block axis before the final projection BF16 cast.\n")
    helper = helper.replace("let partials: DmTensor<bf16,", "let partials: DmTensor<f32,", 1)
    cast = "        .cast::<bf16, m![L / 16 % 4 # 16]>()\n" if down else "        .cast::<bf16, m![H / 16 % 4 # 16]>()\n"
    assert helper.count(cast) == 1
    helper = helper.replace(cast, "", 1)
    cut = helper.index("    let scales: DmTensor<bf16")
    row, block, cluster, slices = ("H % 30", "L / 16 % 240", "DownCluster", "DownRowsByColumns") if down else ("FfnRow", "H / 16", "UpGateCluster", "UpGateSlices")
    # Keep exactly the F32 block dot; only final projection values round to BF16.
    body = """    static_passes!(i, [0, 1, 2, 3, 4], {
        let scale_vrf: VrfTensor<f32, Chip, CLUSTER, SLICES, m![ROW = 6, BLOCK]> = device.sub
            .begin(scales.view().tile::<m![ROW], 6, m![ROW = 6 # 30, BLOCK]>(6 * i))
            .fetch::<m![ROW = 6], m![BLOCK]>().fetch_cast::<f32>()
            .collect::<m![ROW = 6, BLOCK / 8], m![BLOCK % 8]>().to_vrf();
        device.main
            .begin(partials.view().tile::<m![ROW], 6, m![ROW = 6 # 30, BLOCK]>(6 * i))
            .fetch::<m![ROW = 6, BLOCK / 8], m![BLOCK % 8]>()
            .collect::<m![ROW = 6, BLOCK / 8], m![BLOCK % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![ROW = 6, BLOCK / 4], m![BLOCK % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
            .vector_intra_slice_reduce::<REDUCE, m![ROW = 6], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>().vector_final()
COMMIT
    });
}

"""
    if down:
        commit = """            .commit_trim::<m![1 # 2]>()
            .commit_view(output.view_mut().tile::<m![ROW], 6, m![ROW = 6 #{!} 30, 1 # 2]>(6 * i));"""
    else:
        commit = """            .transpose::<m![ROW = 6 / 2], m![ROW = 6 % 2 # 8]>().commit_trim::<m![ROW = 6 % 2]>()
            .commit_view(output.view_mut().tile::<m![ROW], 6, m![ROW = 6 #{!} 30]>(6 * i));"""
    body = body.replace("COMMIT", commit)
    for old, new in {"CLUSTER":cluster, "SLICES":slices, "ROW":row, "BLOCK":block, "REDUCE":"L" if down else "H"}.items():
        body = body.replace(old, new)
    return source[:a] + helper[:cut] + body + source[b:]


def e5_three_terms(source):
    source = source.replace("FfnTerm = 2", "FfnTerm = 3")
    source = source.replace("TrfTensor<f8e4m3", "TrfTensor<f8e5m2")
    source = source.replace(".fetch_table_lookup::<f8e4m3>()", ".fetch_table_lookup::<f8e5m2>()", 1)
    a = source.index("    // Quantize the actual")
    b = source.index("    let x = geglu", a)
    source = source[:a] + source[a:b].replace("f8e4m3", "f8e5m2") + source[b:]
    a = source.index("type FfnQuantSlices")
    quant = source[a:].replace("f8e4m3", "f8e5m2").replace("0.0078125f32", "0.00006103515625f32")
    # The exponent floor gives a finite normal scale after dividing by 2^14.
    quant = quant.replace("1.504632769052528e-36f32", "1.925929944387236e-34f32")
    quant = quant.replace("floor 2^-119 guarantees", "floor 2^-112 guarantees")
    quant = quant.replace("scale <= 2^120", "scale <= 2^113")
    start = quant.index("    let lo: DmTensor")
    end = quant.index("    let mut packed:", start)
    third = quant[start:end].replace("let lo:", "let third:")
    third = third.replace("&hi_vrf", "&sum_vrf")
    # Add the first two terms on Sub; Main then needs one subtraction.
    prep = """    let sum_terms: DmTensor<f32, Chip, m![2], FfnQuantSlices, m![H % 480]> = device.sub
        .begin(lo.view()).fetch::<m![H / 32 % 15], m![H % 32]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, &hi_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>().vector_final().commit_trim::<m![H % 8]>().commit();
    let sum_vrf: VrfTensor<f32, Chip, m![2], FfnQuantSlices, m![H % 480]> = device.sub
        .begin(sum_terms.view()).fetch::<m![H / 8 % 60], m![H % 8]>()
        .collect::<m![H / 8 % 60], m![H % 8]>().to_vrf();
"""
    quant = quant[:end] + prep + third + quant[end:]
    quant = quant.replace("FfnTerm = 1 #{!} 2", "FfnTerm = 1 #{!} 3")
    marker = "    (broadcast_ffn_quantized(device, &packed), scale.to_dm(&mut device.tdma))"
    quant = quant.replace(marker, """    third.view().to_dm_view(&mut device.tdma,
        packed.view_mut().tile::<m![FfnTerm], 1, m![FfnTerm = 1 #{!} 3, H % 480]>(2));
""" + marker)
    return source[:a] + quant


def compensated_scale_dpe(source):
    """Preserve an F32 block sum as BF16 high/residual terms before scale-DPE."""
    source = source.replace("axes![FfnTerm", "axes![FfnPartial = 2, FfnTerm", 1)
    a = source.index("    let partials: DmTensor<bf16")
    b = source.index("    let scales: DmTensor<bf16", a)
    first = source[a:b].replace("let partials: DmTensor<bf16", "let partials: DmTensor<f32", 1)
    first = first.replace("        .cast::<bf16, m![H / 16 % 4 # 16]>()\n", "")
    split = """    let hi: DmTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![FfnRow, H / 16]> = device.sub
        .begin(partials.view()).fetch::<m![FfnRow, H / 128], m![H / 16 % 8]>()
        .collect::<m![FfnRow, H / 128], m![H / 16 % 8]>()
        .cast::<bf16, m![H / 16 % 8 # 16]>().commit_trim::<m![H / 16 % 8]>().commit();
    let mut lo: DmTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![FfnRow, H / 16]> = DmTensor::new();
    static_passes!(i, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], {
        let hi_vrf: VrfTensor<f32, Chip, UpGateCluster, UpGateSlices, m![FfnRow = 3, H / 16]> = device.sub
            .begin(hi.view().tile::<m![FfnRow], 3, m![FfnRow = 3 # 30, H / 16]>(3 * i))
            .fetch::<m![FfnRow = 3, H / 256], m![H / 16 % 16]>().fetch_cast::<f32>()
            .collect::<m![FfnRow = 3, H / 128], m![H / 16 % 8]>().to_vrf();
        device.main
            .begin(partials.view().tile::<m![FfnRow], 3, m![FfnRow = 3 # 30, H / 16]>(3 * i))
            .fetch::<m![FfnRow = 3, H / 128], m![H / 16 % 8]>()
            .collect::<m![FfnRow = 3, H / 128], m![H / 16 % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![FfnRow = 3, H / 64], m![H / 16 % 4]>()
            .vector_fp_binary(FpBinaryOp::SubF, &hi_vrf)
            .vector_widen_concat::<m![FfnRow = 3, H / 128], m![H / 16 % 8]>().vector_final()
            .cast::<bf16, m![H / 16 % 8 # 16]>().commit_trim::<m![H / 16 % 8]>()
            .commit_view(lo.view_mut().tile::<m![FfnRow], 3, m![FfnRow = 3 #{!} 30, H / 16]>(3 * i));
    });
    let mut compensated: DmTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![FfnPartial, FfnRow, H / 16]> = DmTensor::new();
    hi.view().to_dm_view(&mut device.tdma, compensated.view_mut().tile::<m![FfnPartial], 1, m![FfnPartial = 1 #{!} 2, FfnRow, H / 16]>(0));
    lo.view().to_dm_view(&mut device.tdma, compensated.view_mut().tile::<m![FfnPartial], 1, m![FfnPartial = 1 #{!} 2, FfnRow, H / 16]>(1));
"""
    source = source[:a] + first + split + source[b:]
    a = source.index("    device.main.begin(partials.view())", a)
    b = source.index("        .contract_packet::<m![1]>()", a)
    source = source[:a] + """    device.main.begin(compensated.view())
        .fetch::<m![FfnRow, FfnPartial, H / 256], m![H / 16 % 16]>()
        .collect::<m![FfnRow, FfnPartial, H / 256], m![H / 16 % 16]>()
        .contract_outer::<m![FfnRow, FfnPartial, H / 256], m![H / 16 % 16], _, _, _>(&scales_trf)
""" + source[b:]
    return source


def reload_projection_input(source):
    """Each matrix loads its own TRF/scale from the retained quantized DM."""
    a = source.index("    let x_trf: TrfTensor", source.index("    // Quantize the actual"))
    b = source.index("    let (up, gate) = project_up_and_gate(", a)
    loading = source[a:b].replace(".sub.begin(x)", ".sub.begin(x.view())").replace(".sub.begin(x_scale)", ".sub.begin(x_scale.view())")
    source = source[:a] + source[b:]
    call = "        &x_trf,\n        &x_scale_vrf,\n"
    assert source.count(call) == 1
    source = source.replace(call, "        &x,\n        &x_scale,\n")
    source = source.replace("let x: DmTensorView<'_, f8e5m2, Chip, UpGateCluster, UpGateSlices, m![FfnTerm, H]> = unsafe { x.view().reshape() };", "let x: DmTensor<f8e5m2, Chip, UpGateCluster, UpGateSlices, m![FfnTerm, H]> = unsafe { x.reshape() };")
    source = source.replace("let x_scale: DmTensorView<'_, f32, Chip, UpGateCluster, UpGateSlices, m![1 # 8]> = unsafe { x_scale.view().reshape() };", "let x_scale: DmTensor<f32, Chip, UpGateCluster, UpGateSlices, m![1 # 8]> = unsafe { x_scale.reshape() };")
    a = source.index("pub(crate) fn project_up_and_gate(")
    b = source.index("    up_weight_packed:", a)
    source = source[:a] + """pub(crate) fn project_up_and_gate(
    device: &mut Device,
    x: &DmTensor<f8e5m2, Chip, UpGateCluster, UpGateSlices, m![FfnTerm, H]>,
    x_scale: &DmTensor<f32, Chip, UpGateCluster, UpGateSlices, m![1 # 8]>,
""" + source[b:]
    for which in ("up", "gate"):
        old = f"    project_rows_with_scale_contraction(device, x_trf, x_scale_vrf, &{which}_weight_packed, &{which}_weight_scale, &mut {which});"
        new = loading + old.replace("device, x_trf, x_scale_vrf,", "device, &x_trf, &x_scale_vrf,")
        assert source.count(old) == 1
        source = source.replace(old, new)
    return source


def materialize_fp8(source):
    """Keep the native FP8 arithmetic while separating unpack from contraction."""
    # Only the contraction helper's weight operand changes; public packed inputs stay FP4.
    source = source.replace("weights: &DmTensor<f4e2m1,", "weights: &DmTensor<f8e5m2,", 1)
    source = source.replace("        .fetch_table_lookup::<f8e5m2>()\n", "", 1)
    for which in ("up", "gate"):
        marker = f"    let {which}_weight_scale: HbmTensorView"
        a = source.index(marker)
        unpack = f"""    let {which}_weight_packed: DmTensor<f8e5m2, Chip, UpGateCluster, UpGateSlices, m![FfnRow, H]> = device.main
        .begin({which}_weight_packed.view()).fetch::<m![FfnRow], m![H]>()
        .fetch_table_lookup::<f8e5m2>()
        .collect::<m![FfnRow, H / 32], m![H % 32]>().commit_trim::<m![H % 32]>().commit();
"""
        source = source[:a] + unpack + source[a:]
    return source
