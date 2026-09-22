#!/usr/bin/env python3
"""Build two F32-preserving dataflows without a materialized BF16 weight matrix.

The public shell is reused from E55; it is not an architectural requirement.
`direct` decodes each row tile straight into dot/scale/reduction.
`partials` streams the whole matrix into compact F32 block dots and uses a
separate vector pass for scale/reduction. Neither rounds block dots to BF16.
"""
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = "873f64ed94229c88cd18d2731bdcca7eea0d6365"


def replace(s, before, after, count=1):
    assert s.count(before) == count, (before, s.count(before), count)
    return s.replace(before, after)


def partials(down=False):
    row, block, axis, local, cluster, slices, weights, x = (
        ("H % 60", "L / 16 % 120", "L", "L % 1920", "DownCluster", "DownRowsByColumns", "down_weight_fp8", "&x_trf")
        if down else
        ("L % 32", "H / 16", "H", "H", "UpGateCluster", "UpGateSlices", "weights", "x_trf"))
    pairs = "L / 32 % 60" if down else "H / 32"
    return f"""    // Decode only into the DPE stream. The stored tensor has one F32 per block.
    let partials: DmTensor<f32, Chip, {cluster}, {slices}, m![{row}, {block}]> = device.main
        .begin({weights}.view())
        .fetch::<m![{row}], m![{local}]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![{row}, {block}], m![{axis} % 16]>()
        .contract_outer::<m![{row}, {block}], m![{axis} % 16], _, _, _>({x})
        .contract_packet::<m![1]>()
        .contract_time::<m![{row}, {block}]>()
        .contract_lane::<m![{row}, {block}], m![1 # 8]>(LaneMode::Interleaved)
        .transpose::<m![{row}, {pairs}], m![{axis} / 16 % 2 # 8]>()
        .commit_trim::<m![{axis} / 16 % 2]>()
        .commit();
"""


def transform(original, mode, which):
    s = original
    if which in ("upgate", "combined"):
        for name in ("up", "gate"):
            start = s.index("    // Materialize each unscaled BF16 matrix once;")
            end = s.index(f"    let {name}_weight_scale:", start)
            s = s[:start] + s[end:]
            s = replace(s, f"&{name}_weight_bf16,", f"&{name}_weight_packed,")
        s = replace(s, "weights: &DmTensor<bf16, Chip, UpGateCluster", "weights: &DmTensor<f8e4m3, Chip, UpGateCluster")
        s = s.replace("project_eight_rows_scaled_after_dot", "project_streamed_rows" if mode == "direct" else "reduce_eight_rows_f32")
        start = s.index("    device\n        .main\n        .begin(weights.view().tile")
        stop = s.index("        .vector_init()", start)
        if mode == "direct":
            s = s[:start] + s[start:stop].replace(
                ".fetch::<m![L % 32 = 8, H / 16], m![H % 16]>()",
                ".fetch::<m![L % 32 = 8], m![H]>()\n        .fetch_table_lookup::<bf16>()") + s[stop:]
        else:
            s = s[:start] + """    device.main
        .begin(weights.view().tile::<m![L % 32], 8, m![L % 32 = 8 # 32, H / 16]>(row_offset))
        .fetch::<m![L % 32 = 8, H / 128], m![H / 16 % 8]>()
        .collect::<m![L % 32 = 8, H / 128], m![H / 16 % 8]>()
""" + s[stop:]
            s = replace(s, "weights: &DmTensor<f8e4m3, Chip, UpGateCluster, UpGateSlices, m![L % 32, H]>",
                "weights: &DmTensor<f32, Chip, UpGateCluster, UpGateSlices, m![L % 32, H / 16]>")
            # Main packets now contain eight contiguous block dots, so narrow-split.
            first, rest = s.split("pub(crate) fn project_up_and_gate", 1)
            first = replace(first, ".vector_narrow_trim::<m![1 # 4]>()", ".vector_narrow_split::<m![L % 32 = 8, H / 64], m![H / 16 % 4]>()")
            s = first + "pub(crate) fn project_up_and_gate" + rest
            for name in ("up", "gate"):
                marker = "    static_passes!(i, [0, 1, 2, 3], {\n        reduce_eight_rows_f32(device, x_trf, &" + name + "_weight_packed,"
                replacement = partials().replace("weights.view()", name + "_weight_packed.view()") + "\n" + marker.replace("&" + name + "_weight_packed,", "&partials,")
                s = replace(s, marker, replacement)
    if which in ("down", "combined"):
        start = s.index("    let down_weight_bf16:")
        stop = s.index("    static_passes!(i, [0, 1, 2, 3, 4],", start)
        s = s[:start] + (partials(True) + "\n" if mode == "partials" else "") + s[stop:]
        start = s.index("        device\n            .main\n            .begin(down_weight_bf16")
        stop = s.index("            .vector_init()", start)
        if mode == "direct":
            s = s[:start] + s[start:stop].replace("down_weight_bf16.view()", "down_weight_fp8.view()").replace(
                ".fetch::<m![H % 60 = 12, L / 16 % 120], m![L % 16]>()",
                ".fetch::<m![H % 60 = 12], m![L % 1920]>()\n            .fetch_table_lookup::<bf16>()") + s[stop:]
        else:
            s = s[:start] + """        device.main
            .begin(partials.view().tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L / 16 % 120]>(12 * i))
            .fetch::<m![H % 60 = 12, L / 128 % 15], m![L / 16 % 8]>()
            .collect::<m![H % 60 = 12, L / 128 % 15], m![L / 16 % 8]>()
""" + s[stop:]
            s = s[:start] + replace(s[start:], ".vector_narrow_trim::<m![1 # 4]>()", ".vector_narrow_split::<m![H % 60 = 12, L / 64 % 30], m![L / 16 % 4]>()")
    return s


if __name__ == "__main__":
    original = subprocess.check_output(["git", "show", f"{BASE}:furiosa-opt-gemma4-12B/src/device/shared/mlp.rs"], cwd=ROOT, text=True)
    for mode in ("direct", "partials"):
        for which in ("combined", "upgate", "down"):
            out = HERE / (mode if which == "combined" else mode + "-" + which)
            out.mkdir(exist_ok=True)
            (out / "mlp.rs").write_text(transform(original, mode, which))
    # With lookup/DPE and VE no longer sharing a stream, try the reduction on
    # Sub. Actual engine overlap must be established from the compiled schedule.
    s = transform(original, "partials", "combined")
    s = replace(s, "device.main\n        .begin(weights.view().tile", "device.sub\n        .begin(weights.view().tile")
    s = replace(s, "device.main\n            .begin(partials.view().tile", "device.sub\n            .begin(partials.view().tile")
    out = HERE / "partials-sub"
    out.mkdir(exist_ok=True)
    (out / "mlp.rs").write_text(s)
