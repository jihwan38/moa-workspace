#!/usr/bin/env python3
"""Freeze two M01 transport alternatives without modifying adopted runtime files."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = "8109a27c25b2c5fb4e34e6839be75d5387d98a13"
TARGET = "furiosa-opt-gemma4-12B/src/device/sliding/projection.rs"

BODY = '''pub(crate) fn project_output(
    device: &mut Device,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    weight_scale: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, OutputEpilogueSlices, m![H % 480]> {
    type IngestSlices = m![Qs / 256, H / 120 % 16];
    type OutputRows = m![H / 120 % 16, 1 # 16];

    // Both operands have the same column-major slice owner:
    // s=16*k+r, h=1920*c+120*r+t, j=256*k+u.
    let x: DmTensor<bf16, Chip, OutputCluster, IngestSlices, m![Qs % 256]> = x.to_dm(&mut device.tdma);
    let x_trf: TrfTensor<bf16, Chip, OutputCluster, IngestSlices, m![1], m![Qs % 256]> = device
        .sub.begin(x.view())
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>()
        .collect::<m![Qs / 16 % 16], m![Qs % 16]>()
        .to_trf();
    let weight: HbmTensorView<'_, f8e4m3, Chip, m![Dummy2, H % 1920, Qs]> = unsafe { weight.view().reshape() };
    let weight: DmTensor<f8e4m3, Chip, OutputCluster, IngestSlices, m![H % 120, Qs % 256]> = weight.to_dm(&mut device.tdma);

    // Compute in the ingress layout. No full-weight DM repack and no BF16
    // rounding of the partial: adjacent F32 rows form compact 8-byte commits.
    let partial: DmTensor<f32, Chip, OutputCluster, IngestSlices, m![H % 120]> = device
        .main.begin(weight.view())
        .fetch::<m![H % 120, Qs / 16 % 16], m![Qs % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![H % 120, Qs / 16 % 16], m![Qs % 16]>()
        .contract_outer::<m![H % 120, Qs / 32 % 8], m![Qs % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120]>()
        .contract_lane::<m![H % 120], m![1 # 8]>(LaneMode::Interleaved)
        .transpose::<m![H / 2 % 60], m![H % 2 # 8]>()
        .commit_trim::<m![H % 2]>()
        .commit();

__TRANSFER__
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<OutputRows, m![H / 8 % 15]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit();
    let result: DmTensor<f32, Chip, m![H / 1920], OutputRows, m![H % 120]> = unsafe { result.reshape() };
    let result: DmTensor<f32, Chip, Cluster, OutputEpilogueSlices, m![H % 480]> = result.to_dm(&mut device.tdma);
    apply_output_channel_scale_f32(device, &result, weight_scale)
}
'''

TRANSFERS = {
    "dma-v1": '''    // Move only F32 partials to put K innermost for inter-slice reduction.
    let partial: DmTensor<f32, Chip, OutputCluster, OutputColumns, m![H % 120]> = partial.to_dm(&mut device.tdma);
    let result: DmTensor<f32, Chip, OutputCluster, OutputRows, m![H % 120]> = device
        .main.begin(partial.view())
        .fetch::<m![H / 8 % 15], m![H % 8]>()''',
    "switch-v1": '''    // Fetch all compact partials, transpose slice(k,r) -> slice(r,k),
    // and reduce in the same consumer stream. Ring256 cost is not free.
    let result: DmTensor<f32, Chip, OutputCluster, OutputRows, m![H % 120]> = device
        .main.begin(partial.view())
        .fetch::<m![H / 8 % 15], m![H % 8]>()
        .switch::<OutputColumns, m![H / 8 % 15]>(SwitchConfig::Transpose { slice1: 16, slice0: 16 })''',
}


def main():
    commit = subprocess.check_output(["git", "rev-parse", BASE], cwd=ROOT, text=True).strip()
    original = subprocess.check_output(["git", "show", commit + ":" + TARGET], cwd=ROOT, text=True)
    start = original.index("pub(crate) fn project_output(")
    end = original.index("\npub(crate) type OutputEpilogueSlices", start)
    for variant, transfer in TRANSFERS.items():
        dest = HERE / variant
        dest.mkdir(exist_ok=False)
        source = original[:start] + BODY.replace("__TRANSFER__", transfer) + original[end:]
        (dest / "projection.rs").write_text(source)
        recipe = {
            "id": "M01", "variant": variant, "base_commit": commit,
            "comparison_recipe": "baseline/e104.json", "target_kernel": "sliding_attention_output",
            "changed_sources": {"projection.rs": "src/device/sliding/projection.rs"},
            "source_reference_build": "moa-dev/experiments/candidates/c07-native-scale-dpe/build-c07-e104-reviewed-v1-repro.json",
            "unchanged_kernels": ["sliding_project_qkv", "decoder_feedforward"],
            "scope": "whole O projection mapping and reduction-path replacement, E64-family retry",
            "source_composition": {
                "origin": "E64 column-major weight ingress; newly matched BF16 input and direct dot",
                "arithmetic_and_epilogue": "E104/E53 BF16 dot, F32 reduction and scale before first BF16",
                "retained_file_regions": "Every function outside project_output is byte-identical to the selected origin",
                "removed": "E64 full-weight DM repack; E104 inline dot/reduction coupling",
            },
            "graph": "HBM x/W -> column-major slice DM/TRF -> lookup+BF16 dot -> compact F32 partial DM -> "
                     + ("typed DMA slice transpose -> fetch" if variant.startswith("dma") else "fetch+ring256 slice transpose")
                     + " -> F32 16-column sum -> H480x8 gather -> F32 channel scale -> BF16 -> existing post-RMS/residual",
            "prior_related": ["E64 +149.61% measured O; full-weight repack dominated, ingress itself had no static gain",
                              "C12 added partial boundaries without useful overlap; this candidate has one120-row dot and compact partials"],
            "cost_removed": ["15MiB weight repack vs E64; E104 did not pay this cost"],
            "cost_added": ["new245760B F32 partial and its write/read", "typed DMA or ring256 transpose", "separate reducer launch"],
            "diagnostic": "Two predeclared full-path alternatives, not two architecture families. Every input conversion/transfer remains timed. No preconverted-weight benchmark.",
            "first_stop_condition": "Inspect supply and added partial path. Basic NPU screen with bracketing E104 if executable; close regression/<5% O gain without parameter sweep. Static small differences do not prove hardware loss. Repair identified source/axis faults rather than declaring SDK unsupported.",
            "adopted": False,
        }
        (dest / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
        print(variant, hashlib.sha256(source.encode()).hexdigest())


if __name__ == "__main__":
    main()
