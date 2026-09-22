#!/usr/bin/env python3
"""Generate isolated memory-layout variants from the immutable E55 checkpoint."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARTIFACTS = ROOT / "moa-dev/experiments/artifacts/memory-layout-e55"
PREFIX = "furiosa-opt-gemma4-12B/"
VARIANTS = ("down-packed", "down-scale", "down-both", "up-gate-packed",
            "down-packed-split", "down-scale-split", "down-both-split",
            "up-gate-bf16-skew", "down-bf16-skew",
            "up-gate-bf16-skew32", "down-bf16-skew32",
            "qkv-sixteen-rows", "output-column-major", "output-column-major-staged")


def source_path(variant):
    return "src/device/sliding/projection.rs" if variant.startswith(("qkv-", "output-")) else "src/device/shared/mlp.rs"


def kernel_name(variant):
    if variant.startswith("qkv-"):
        return "sliding_project_qkv"
    if variant.startswith("output-"):
        return "sliding_attention_output"
    return "decoder_feedforward"


def projection_source(base, variant):
    if variant == "output-column-major-staged":
        return replace_once(base,
            "let weight: DmTensor<f8e4m3, Chip, OutputCluster, OutputColumns, m![H % 120, Qs % 256]> = weight.to_dm(&mut device.tdma);",
            """// Load with column-major slice order, then route to the innermost
    // column partition required by the inter-slice reduction engine.
    let weight: DmTensor<f8e4m3, Chip, OutputCluster, m![Qs / 256, H / 120 % 16], m![H % 120, Qs % 256]> = weight.to_dm(&mut device.tdma);
    let weight: DmTensor<f8e4m3, Chip, OutputCluster, OutputColumns, m![H % 120, Qs % 256]> = weight.to_dm(&mut device.tdma);""")
    if variant == "output-column-major":
        text = replace_once(base, "type OutputRows = m![H / 120 % 16, 1 # 16];", "type OutputRows = m![1 # 16, H / 120 % 16];")
        return replace_once(text, "type OutputColumns = m![H / 120 % 16, Qs / 256];", "type OutputColumns = m![Qs / 256, H / 120 % 16];")
    assert variant == "qkv-sixteen-rows"
    text, output = base.split("pub(crate) fn project_output(", 1)
    for old, new in (
        ("Qs / 64 % 32, Qs % 8", "Qs / 128 % 16, Qs % 16"),
        ("Ps / 32 % 32, Ps % 8", "Ps / 64 % 16, Ps % 16"),
        ("Qs / 8 % 256", "Qs / 128 % 16, Qs / 16 % 8 # 16"),
        ("Ps / 32 % 32, Ps / 8 % 4 # 8", "Ps / 64 % 16, Ps / 16 % 4 # 16"),
        ("Qs / 8 % 8", "Qs / 16 % 8"),
        ("Qs / 8 % 4", "Qs / 16 % 4"),
        ("Qs / 32 % 2", "Qs / 64 % 2"),
        ("Ps / 8 % 4", "Ps / 16 % 4"),
        ("Ps / 16 % 4 # 8", "Ps / 16 % 4 # 16"),
        ("Qs % 8", "Qs % 16"),
        ("Ps % 8", "Ps % 16"),
        ("Qs / 4 % 2", "Qs / 4 % 4"),
        ("Ps / 4 % 2", "Ps / 4 % 4"),
        ("slice1: 8", "slice1: 16"),
        ("m![Ns % 4, Gs, Ds / 8], m![Ds % 8]", "m![Ns % 4, Gs, Ds / 128, Ds / 16 % 8 # 16], m![Ds % 16]"),
        ("m![Ns % 4, Ds / 32, Ds / 8 % 4 # 8], m![Ds % 8]", "m![Ns % 4, Ds / 64, Ds / 16 % 4 # 16], m![Ds % 16]"),
    ):
        assert old in text, old
        text = text.replace(old, new)
    # Q has eight live local rows inside a sixteen-slot transpose group.
    # Padding is carried through the scale VRF and output Switch, never summed.
    for old, new in (
        ("QueryRows, m![Qs / 16 % 8, 1 # 8]>", "QueryRows, m![Qs / 16 % 8 # 16, 1 # 8]>"),
        (".switch::<QueryRows, m![Qs / 16 % 8]>", ".switch::<QueryRows, m![Qs / 16 % 8 # 16]>"),
        (".collect::<m![Qs / 16 % 8], m![1 # 8]>()", ".collect::<m![Qs / 16 % 8 # 16], m![1 # 8]>()"),
        (".fetch::<m![Qs / 16 % 8], m![1 # 4]>()", ".fetch::<m![Qs / 16 % 8 # 16], m![1 # 4]>()"),
    ):
        text = replace_once(text, old, new)
    text = text.replace("Four valid local rows are padded to an eight-way Switch group", "Four valid local rows are padded to a sixteen-way Switch group")
    return text + "pub(crate) fn project_output(" + output


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new, 1)


def source_for(base, variant):
    if variant.startswith(("qkv-", "output-")):
        return projection_source(base, variant)
    if variant.endswith("skew32"):
        # Retain the 32-byte BF16 commit packet as an innermost run.
        # The attempted 8-byte skew cannot satisfy that engine constraint.
        return source_for(base, variant.removesuffix("32")).replace("H # 3844", "H # 3856").replace("L % 1920 # 1924", "L % 1920 # 1936")
    if variant.endswith("-split"):
        # Name the existing 8 x 1920 (or 8 x 120) factors separately only
        # across DMA. Padding a modulo of the unsplit L axis is rejected by
        # SDK 0.8.1. The subsequent reshape renames these factors in place;
        # it preserves both the physical padding and every live coordinate.
        text = source_for(base, variant.removesuffix("-split"))
        text = replace_once(text, "const INVSQRT2: f32", "mod down_dma_axes {\n    use furiosa_opt_std::prelude::*;\n    axes![DownShard = 8, DownElement = 1920, DownBlock = 120];\n}\nuse down_dma_axes::{DownShard, DownElement, DownBlock};\n\nconst INVSQRT2: f32")
        for kind, dtype, axis, original, padded in (
            ("packed", "f4e2m1", "DownElement", "L", "L % 1920 # 2048"),
            ("scale", "f8e4m3", "DownBlock", "L / 16", "L / 16 % 120 # 128"),
        ):
            if variant not in (f"down-{kind}-split", "down-both-split"):
                continue
            name = f"down_weight_{kind}"
            pitch = 2048 if kind == "packed" else 128
            old = f"""    let {name}: HbmTensorView<'_, {dtype}, Chip, m![Dummy2, H % 1920, {original}]> = unsafe {{ {name}.view().reshape() }};
    let {name}: DmTensor<{dtype}, Chip, DownCluster, DownRowsByColumns, m![H % 60, {padded}]> =
        {name}.to_dm(&mut device.tdma);"""
            new = f"""    // Same HBM order: h = 1920*c + 60*r + row, column = {1920 if kind == 'packed' else 120}*shard + local.
    let {name}: HbmTensorView<'_, {dtype}, Chip, m![Dummy2, H % 1920, DownShard, {axis}]> = unsafe {{ {name}.view().reshape() }};
    let {name}: DmTensor<{dtype}, Chip, DownCluster, m![H / 60 % 32, DownShard], m![H % 60, {axis} # {pitch}]> =
        {name}.to_dm(&mut device.tdma);
    // Rename only; the row stride and inaccessible-to-math padding stay intact.
    let {name}: DmTensor<{dtype}, Chip, DownCluster, DownRowsByColumns, m![H % 60, {padded}]> = unsafe {{ {name}.reshape() }};"""
            text = replace_once(text, old, new)
        return text
    text = base
    if variant in ("down-packed", "down-both"):
        text = replace_once(text,
            "let down_weight_packed: DmTensor<f4e2m1, Chip, DownCluster, DownRowsByColumns, m![H % 60, L % 1920]>",
            "let down_weight_packed: DmTensor<f4e2m1, Chip, DownCluster, DownRowsByColumns, m![H % 60, L % 1920 # 2048]>")
    if variant in ("down-scale", "down-both"):
        text = replace_once(text,
            "let down_weight_scale: DmTensor<f8e4m3, Chip, DownCluster, DownRowsByColumns, m![H % 60, L / 16 % 120]>",
            "let down_weight_scale: DmTensor<f8e4m3, Chip, DownCluster, DownRowsByColumns, m![H % 60, L / 16 % 120 # 128]>")
        text = replace_once(text,
            ".begin(down_weight_scale.view().tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L / 16 % 120]>(12 * i))",
            ".begin(down_weight_scale.view().tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L / 16 % 120 # 128]>(12 * i))")
    if variant == "up-gate-packed":
        for name in ("up", "gate"):
            text = replace_once(text,
                f"let {name}_weight_packed: DmTensor<f4e2m1, Chip, UpGateCluster, UpGateSlices, m![L % 32, H]>",
                f"let {name}_weight_packed: DmTensor<f4e2m1, Chip, UpGateCluster, UpGateSlices, m![L % 32, H # 4096]>")
    if variant == "up-gate-bf16-skew":
        old = "DmTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![L % 32, H]>"
        assert text.count(old) == 3  # Two matrices and their common dot helper.
        text = text.replace(old, "DmTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![L % 32, H # 3844]>")
        text = replace_once(text,
            ".begin(weights.view().tile::<m![L % 32], 8, m![L % 32 = 8 # 32, H]>(row_offset))",
            ".begin(weights.view().tile::<m![L % 32], 8, m![L % 32 = 8 # 32, H # 3844]>(row_offset))")
    if variant == "down-bf16-skew":
        text = replace_once(text,
            "let down_weight_bf16: DmTensor<bf16, Chip, DownCluster, DownRowsByColumns, m![H % 60, L % 1920]>",
            "let down_weight_bf16: DmTensor<bf16, Chip, DownCluster, DownRowsByColumns, m![H % 60, L % 1920 # 1924]>")
        text = replace_once(text,
            ".begin(down_weight_bf16.view().tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L % 1920]>(12 * i))",
            ".begin(down_weight_bf16.view().tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L % 1920 # 1924]>(12 * i))")
    # Reversing only the new padding annotations recovers the exact source.
    restored = text.replace("L % 1920 # 2048", "L % 1920").replace("L / 16 % 120 # 128", "L / 16 % 120").replace("H # 4096", "H").replace("H # 3844", "H").replace("L % 1920 # 1924", "L % 1920")
    assert restored == base
    assert text != base
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=VARIANTS)
    args = parser.parse_args()
    files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", "baseline-e55", "--", PREFIX + "src", PREFIX + "Cargo.toml", PREFIX + "rust-toolchain.toml"], cwd=ROOT, text=True).splitlines()
    out = HERE / args.variant
    crate = ARTIFACTS / args.variant / "crate"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in files:
        rel = name.removeprefix(PREFIX)
        data = subprocess.check_output(["git", "show", f"baseline-e55:{name}"], cwd=ROOT)
        if rel == source_path(args.variant):
            base = data.decode()
            candidate = source_for(base, args.variant)
            data = candidate.encode()
            (out / Path(rel).name).write_bytes(data)
            (out / "candidate.patch").write_text("".join(difflib.unified_diff(base.splitlines(True), candidate.splitlines(True), fromfile="a/" + rel, tofile="b/" + rel)))
        path = crate / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        manifest[rel] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    lock = (ROOT / "moa-dev/locks/gemma4-Cargo.lock").read_bytes()
    assert hashlib.sha256(lock).hexdigest() == "aaff8dd41f9c643dd8c9faa358b9bc780e88a42c09172c1a5d7a0726d07efe09"
    (crate / "Cargo.lock").write_bytes(lock)
    manifest["Cargo.lock"] = {"bytes": len(lock), "sha256": hashlib.sha256(lock).hexdigest()}
    assert len(manifest) == 46
    (out / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(crate)


if __name__ == "__main__":
    main()
