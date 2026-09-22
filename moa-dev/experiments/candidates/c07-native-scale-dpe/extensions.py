"""Independent C07 alternatives: two-column down and DPE-time term summation."""


def time_down(source):
    start = source.index("pub(crate) type DownCluster")
    end = source.index("type FfnQuantSlices", start)
    s = source[start:end]
    s = s.replace("m![FfnTerm], m![L % 3840]>", "m![1], m![FfnTerm, L % 3840]>")
    s = s.replace(".fetch::<m![1], m![FfnTerm, L % 3840]>()", ".fetch::<m![FfnTerm], m![L % 3840]>()")
    s = s.replace("            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 16.0f32)\n", "")
    a = s.index("        .contract_outer::<m![H % 30, L / 64 % 60]")
    b = s.index("        .cast::<bf16, m![L / 16 % 4 # 16]>()", a)
    s = s[:a] + """        .contract_outer::<m![H % 30, L / 64 % 60, FfnTerm], m![L % 64], _, _, _>(x_trf)
        .contract_packet::<m![L / 16 % 4]>()
        .contract_time::<m![H % 30, L / 64 % 60]>()
        .contract_lane::<m![H % 30, L / 64 % 60], m![L / 16 % 4 # 8]>(LaneMode::Sequential)
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![L / 16 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), x_scale_vrf)
        .vector_widen_pad::<m![L / 16 % 4 # 8]>().vector_final()
""" + s[b:]
    return source[:start] + s + source[end:]


def two_columns(source):
    start = source.index("pub(crate) type DownCluster")
    end = source.index("type FfnQuantSlices", start)
    s = source[start:end]
    replacements = {
        "m![H / 30 % 64, 1 # 4]": "m![H / 30 % 64, H % 2, 1 # 2]",
        "m![H / 30 % 64, L / 3840]": "m![H / 30 % 64, H % 2, L / 7680]",
        "DownQuantK = 3840": "DownQuantK = 7680", "DownQuantTile = 8": "DownQuantTile = 16",
        "DownQuantTile = 1 # 8": "DownQuantTile = 1 # 16", "DownQuantTile = 1 #{!} 8": "DownQuantTile = 1 #{!} 16",
        "[0, 1, 2, 3, 4, 5, 6, 7]": "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]",
        "1 # 4, L / 3840": "1 # 8, L / 7680", "4, L / 3840": "8, L / 7680",
        "H % 30": "H / 2 % 15", "L / 16 % 240": "L / 16 % 480", "L / 32 % 120": "L / 32 % 240",
        "L / 64 % 60": "L / 64 % 120", "L / 128 % 30": "L / 128 % 60", "L / 256 % 15": "L / 256 % 30",
        "L / 3840": "L / 7680", "L % 3840": "L % 7680",
    }
    import re
    pattern = "|".join(re.escape(k) for k in sorted(replacements, key=len, reverse=True))
    s = re.sub(pattern, lambda m: replacements[m.group()], s)
    a = s.index("    let down: DmTensor<f32, Chip, DownCluster, DownRows")
    b = s.index("    let down: DmTensor<f32, Chip, Cluster, DownEpilogueSlices", a)
    s = s[:a] + """    let down: DmTensor<f32, Chip, DownCluster, DownRows, m![H / 2 % 15, 1 # 2]> = device.main
        .begin(partial.view()).fetch::<m![H / 2 % 15], m![1 # 8]>().collect::<m![H / 2 % 15], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>().vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<DownRows, m![H / 2 % 15]>(InterSliceReduceOpF32::Add)
        .vector_final().commit_trim::<m![1 # 2]>().commit();
    let down: DmTensor<f32, Chip, DownCluster, m![H / 30 % 64, 1 # 4], m![H % 30]> = device.main
        .begin(down.view()).fetch::<m![H / 2 % 15], m![1 # 8]>()
        .switch::<m![H / 30 % 64, 1 # 4], m![H / 2 % 15, H % 2]>(SwitchConfig::CustomBroadcast { ring_size: 4 })
        .collect::<m![H / 2 % 15, H % 2], m![1 # 8]>()
        .transpose::<m![H / 2 % 15], m![H % 2 # 8]>().commit_trim::<m![H % 2]>().commit();
    let down: DmTensor<f32, Chip, m![H / 1920], m![H / 30 % 64, 1 # 4], m![H % 30]> = unsafe { down.reshape() };
""" + s[b:]
    # An odd 15-row result is stored in scalar slots until the two row lanes join.
    s = s.replace("output: &mut DmTensor<f32, Chip, DownCluster, DownRowsByColumns, m![H / 2 % 15, 1 # 2]>",
                  "output: &mut DmTensor<f32, Chip, DownCluster, DownRowsByColumns, m![H / 2 % 15, 1 # 2]>")
    return source[:start] + s + source[end:]
