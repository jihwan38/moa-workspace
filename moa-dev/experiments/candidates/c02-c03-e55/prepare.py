#!/usr/bin/env python3
"""Construct isolated C02/C03 sources from the immutable E55 source commit."""
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = "873f64ed94229c88cd18d2731bdcca7eea0d6365"
CRATE = "furiosa-opt-gemma4-12B"


def base(path):
    return subprocess.check_output(["git", "show", f"{BASE}:{CRATE}/{path}"], cwd=ROOT, text=True)


def replace(text, before, after, count=1):
    assert text.count(before) == count, (before, text.count(before), count)
    return text.replace(before, after)


def c03():
    out = HERE / "c03-cluster-heads"
    out.mkdir(exist_ok=True)
    projection = base("src/device/sliding/projection.rs")
    prefix, suffix = projection.split("pub(crate) fn project_output(", 1)
    prefix = replace(prefix, "pub(crate) fn project_query(",
        "pub(crate) type HeadCluster = m![Ns / 4];\npub(crate) type HeadSlices = m![1 # 64, Ns % 4];\n\npub(crate) fn project_query(")
    prefix = prefix.replace("Chip, Cluster, m![1 # 32, Ns]", "Chip, HeadCluster, HeadSlices")
    (out / "projection.rs").write_text(prefix + "pub(crate) fn project_output(" + suffix)
    rms = base("src/device/sliding/rmsnorm.rs")
    rms = replace(rms, "use crate::device::layout::Cluster;", "use super::projection::{HeadCluster, HeadSlices};")
    rms = replace(rms, "type HeadSlices = m![1 # 32, Ns];\n", "")
    rms = rms.replace("Chip, Cluster, HeadSlices", "Chip, HeadCluster, HeadSlices")
    rms = rms.replace("{Ds, Gs, Ns}", "{Ds, Gs}")
    (out / "rmsnorm.rs").write_text(rms)
    rope = base("src/device/sliding/rope.rs")
    rope = replace(rope, "type KvHeadsAcrossSlices = m![1 # 32, Ns];", "use super::projection::{HeadCluster, HeadSlices};")
    rope = rope.replace("Chip, Cluster, KvHeadsAcrossSlices", "Chip, HeadCluster, HeadSlices")
    rope = rope.replace("Chip, Cluster, Slice, m![Ns,", "Chip, HeadCluster, Slice, m![Ns % 4,")
    rope = rope.replace(".switch::<Slice, m![Ns]>(SwitchConfig::Broadcast1 { slice1: 8, slice0: 1 })",
                        ".switch::<Slice, m![Ns % 4]>(SwitchConfig::Broadcast1 { slice1: 4, slice0: 1 })")
    rope = rope.replace(".collect::<m![Ns,", ".collect::<m![Ns % 4,")
    (out / "rope.rs").write_text(rope)
    ops = base("src/ops.rs")
    start = ops.index("pub fn sliding_project_qkv(")
    stop = ops.index("#[device(chip = 1)]", start)
    qkv = ops[start:stop]
    qkv = qkv.replace("Chip, Cluster, m![1 # 32, Ns]", "Chip, sliding::projection::HeadCluster, sliding::projection::HeadSlices")
    qkv = qkv.replace("Chip, Cluster, Slice, m![Ns, Ds]", "Chip, sliding::projection::HeadCluster, Slice, m![Ns % 4, Ds]")
    # Keep the final V gather as typed DMA in this first candidate; no new arithmetic.
    (out / "ops.rs").write_text(ops[:start] + qkv + ops[stop:])


def packet_body(down=False):
    """Two independent 16-element F32 dots; compact before the F32 reduction."""
    row, tile, cluster, slices, axis = (("H % 60 = 4", "H % 60", "DownCluster", "DownRowsByColumns", "L")
        if down else ("L % 32 = 4", "L % 32", "UpGateCluster", "UpGateSlices", "H"))
    width = 60 if down else 32
    local = "L % 1920" if down else "H"
    block = "L / 16 % 120" if down else "H / 16"
    pair = "L / 32 % 60" if down else "H / 32"
    eight = "L / 128 % 15" if down else "H / 128"
    four = "L / 64 % 30" if down else "H / 64"
    w, s, x, output = ("down_weight_bf16", "down_weight_scale", "&x_trf", "down") if down else ("weights", "scales", "x_trf", "output")
    offset = "4 * i" if down else "row_offset"
    inter = f".vector_inter_slice_reduce::<DownRows, m![{row}]>(InterSliceReduceOpF32::Add)" if down else ""
    return f"""    // FP8 conversion is contiguous; a separate F32 sequencer pass creates padded slots.
    let sf: DmTensor<f32, Chip, {cluster}, {slices}, m![{row}, {block}]> = device.sub
        .begin({s}.view().tile::<m![{tile}], 4, m![{row} # {width}, {block}]>({offset}))
        .fetch::<m![{row}, {eight}], m![{axis} / 16 % 8]>().fetch_cast::<f32>()
        .collect::<m![{row}, {eight}], m![{axis} / 16 % 8]>()
        .commit_trim::<m![{axis} / 16 % 8]>().commit();
    let sp: DmTensor<f32, Chip, {cluster}, {slices}, m![{row}, {pair}, {axis} / 16 % 2 # 4]> = device.main.begin(sf.view())
        .fetch::<m![{row}, {pair}], m![{axis} / 16 % 2 # 8]>()
        .collect::<m![{row}, {pair}], m![{axis} / 16 % 2 # 8]>()
        .commit_trim::<m![{axis} / 16 % 2 # 4]>().commit();
    let sv: VrfTensor<f32, Chip, {cluster}, {slices}, m![{row}, {pair}, {axis} / 16 % 2 # 4]> = device.sub.begin(sp.view())
        .fetch::<m![{row}, {four}], m![{axis} / 32 % 2, {axis} / 16 % 2 # 4]>()
        .collect::<m![{row}, {four}], m![{axis} / 32 % 2, {axis} / 16 % 2 # 4]>().to_vrf();
    let partial: DmTensor<f32, Chip, {cluster}, {slices}, m![{row}, {block}]> = device.main
        .begin({w}.view().tile::<m![{tile}], 4, m![{row} # {width}, {local}]>({offset}))
        .fetch::<m![{row}, {block}], m![{axis} % 16]>()
        .collect::<m![{row}, {block}], m![{axis} % 16]>()
        .contract_outer::<m![{row}, {pair}], m![{axis} % 32], _, _, _>({x})
        .contract_packet::<m![{axis} / 16 % 2]>()
        .contract_time::<m![{row}, {pair}]>()
        .contract_lane::<m![{row}, {pair}], m![{axis} / 16 % 2 # 8]>(LaneMode::Sequential)
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![{axis} / 16 % 2 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &sv)
        .vector_widen_pad::<m![{axis} / 16 % 2 # 8]>().vector_final()
        .commit_trim::<m![{axis} / 16 % 2]>().commit();
    // SDK 0.8.1 rejects reducing a padded packet axis. No BF16 partial cast.
    device.main.begin(partial.view())
        .fetch::<m![{row}, {eight}], m![{axis} / 16 % 8]>()
        .collect::<m![{row}, {eight}], m![{axis} / 16 % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![{row}, {four}], m![{axis} / 16 % 4]>()
        .vector_intra_slice_reduce::<{axis}, m![{row}], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        {inter}
        .vector_final().cast::<bf16, m![1 # 16]>()
        .transpose::<m![{row} / 4], m![{row} % 4 # 16]>()
        .commit_trim::<m![{row} % 4]>()
        .commit_view({output}.view_mut().tile::<m![{tile}], 4, m![{row} #{{!}} {width}]>({offset}));
"""


def c02():
    original = base("src/device/shared/mlp.rs")
    for variant in ("upgate", "down", "combined"):
        s = original
        if variant in ("upgate", "combined"):
            start = s.index("    let scale_vrf:", s.index("fn project_eight_rows_scaled_after_dot("))
            stop = s.index("\n}\n\npub(crate) fn project_up_and_gate", start)
            s = s[:start] + packet_body() + s[stop:]
            s = s.replace("project_eight_rows_scaled_after_dot", "project_four_rows_packet_scaled")
            s = replace(s, "static_passes!(i, [0, 1, 2, 3],", "static_passes!(i, [0, 1, 2, 3, 4, 5, 6, 7],", 2)
            s = replace(s, "&mut up, 8 * i", "&mut up, 4 * i")
            s = replace(s, "&mut gate, 8 * i", "&mut gate, 4 * i")
        if variant in ("down", "combined"):
            start = s.index("    static_passes!(i, [0, 1, 2, 3, 4],")
            stop = s.index("\n    // Only rename the existing high-row cluster axis.", start)
            s = s[:start] + "    static_passes!(i, [" + ", ".join(map(str, range(15))) + "], {\n" + packet_body(True) + "    });\n" + s[stop:]
        out = HERE / f"c02-{variant}"
        out.mkdir(exist_ok=True)
        (out / "mlp.rs").write_text(s)


if __name__ == "__main__":
    c03()
    c02()
