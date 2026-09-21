#!/usr/bin/env python3
"""Stage two 16-row up-weight chunks while retaining eight-row BF16 dots."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT / "furiosa-opt-gemma4-12B"
CRATE = ROOT / "moa-dev/experiments/artifacts/ffn-up-sixteen-row-e55/crate"
SOURCE = Path("src/device/shared/mlp.rs")


def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after, 1)


def main():
    original = subprocess.check_output(
        ["git", "show", "baseline-e55:furiosa-opt-gemma4-12B/src/device/shared/mlp.rs"],
        cwd=ROOT,
    ).decode()
    assert (BASE / SOURCE).read_text() == original
    start = original.index("fn project_eight_rows_scaled_after_dot(")
    end = original.index("pub(crate) fn project_up_and_gate(")
    helper = original[start:end]
    helper = replace_once(helper, "project_eight_rows_scaled_after_dot", "project_eight_rows_from_sixteen")
    helper = replace_once(helper,
        "weights: &DmTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![L % 32, H]>",
        "weights: &DmTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![L % 32 = 16, H]>")
    helper = replace_once(helper, "    row_offset: usize,", "    row_offset: usize,\n    local_offset: usize,")
    helper = replace_once(helper,
        ".begin(weights.view().tile::<m![L % 32], 8, m![L % 32 = 8 # 32, H]>(row_offset))",
        ".begin(weights.view().tile::<m![L % 32 = 16], 8, m![L % 32 = 16 = 8 # 16, H]>(local_offset))")
    helpers = helper + """// Tile the row-major HBM view before copying it to each slice.
fn load_up_sixteen_rows(
    device: &mut Device,
    source: &HbmTensorView<'_, f4e2m1, Chip, m![Dummy2, L / 32 % 240, L % 32, H]>,
    row_offset: usize,
) -> DmTensor<f4e2m1, Chip, UpGateCluster, UpGateSlices, m![L % 32 = 16, H]> {
    source.tile::<m![L % 32], 16, m![Dummy2, L / 32 % 240, L % 32 = 16 # 32, H]>(row_offset)
        .to_dm(&mut device.tdma)
}

fn decode_up_sixteen_rows(
    device: &mut Device,
    packed: &DmTensor<f4e2m1, Chip, UpGateCluster, UpGateSlices, m![L % 32 = 16, H]>,
) -> DmTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![L % 32 = 16, H]> {
    device.main.begin(packed.view())
        .fetch::<m![L % 32 = 16], m![H]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![L % 32 = 16, H / 8], m![H % 8]>()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

"""
    candidate = original[:end] + helpers + original[end:]
    start = candidate.index("    let up_weight_packed: HbmTensorView")
    end = candidate.index("    let mut gate:", start)
    candidate = candidate[:start] + """    let up_weight_packed: HbmTensorView<'_, f4e2m1, Chip, m![Dummy2, L / 32 % 240, L % 32, H]> =
        unsafe { up_weight_packed.view().reshape() };
    let up_weight_scale: HbmTensorView<'_, f8e4m3, Chip, m![Dummy2, L % 7680, H / 16]> =
        unsafe { up_weight_scale.view().reshape() };
    let up_weight_scale: DmTensor<f8e4m3, Chip, UpGateCluster, UpGateSlices, m![L % 32, H / 16]> =
        up_weight_scale.to_dm(&mut device.tdma);

    // These are independent immutable tensors. Allocation and overlap remain
    // compiler decisions; no fixed physical buffer count is assumed.
    let first_packed = load_up_sixteen_rows(device, &up_weight_packed, 0);
    let second_packed = load_up_sixteen_rows(device, &up_weight_packed, 16);
    let first = decode_up_sixteen_rows(device, &first_packed);
    project_eight_rows_from_sixteen(device, x_trf, &first, &up_weight_scale, &mut up, 0, 0);
    project_eight_rows_from_sixteen(device, x_trf, &first, &up_weight_scale, &mut up, 8, 8);
    let second = decode_up_sixteen_rows(device, &second_packed);
    project_eight_rows_from_sixteen(device, x_trf, &second, &up_weight_scale, &mut up, 16, 0);
    project_eight_rows_from_sixteen(device, x_trf, &second, &up_weight_scale, &mut up, 24, 8);

""" + candidate[end:]
    gate = "    let mut gate:"
    assert original[original.index(gate):] == candidate[candidate.index(gate):]
    CRATE.mkdir(parents=True, exist_ok=True)
    inputs = [*sorted((BASE / "src").rglob("*.rs")), *(BASE / n for n in ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml"])]
    manifest = {}
    for path in inputs:
        rel = path.relative_to(BASE)
        if rel.name != "Cargo.lock":
            tagged = subprocess.check_output(["git", "show", f"baseline-e55:furiosa-opt-gemma4-12B/{rel}"], cwd=ROOT)
            assert path.read_bytes() == tagged, str(rel)
        else:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == "aaff8dd41f9c643dd8c9faa358b9bc780e88a42c09172c1a5d7a0726d07efe09"
        dest = CRATE / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        if rel == SOURCE:
            dest.write_text(candidate)
        manifest[str(rel)] = {"bytes": dest.stat().st_size, "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}
    (HERE / "mlp.rs").write_text(candidate)
    (HERE / "candidate.patch").write_text("".join(difflib.unified_diff(original.splitlines(True), candidate.splitlines(True), fromfile="a/src/device/shared/mlp.rs", tofile="b/src/device/shared/mlp.rs")))
    (HERE / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(CRATE)


if __name__ == "__main__":
    main()
