#!/usr/bin/env python3
"""Freeze M02's complete K-owned O projection; leave the adopted crate untouched."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = "ecdeaca8187504e833a415731f626a3fa1ef3d5e"
TARGET = "furiosa-opt-gemma4-12B/src/device/sliding/projection.rs"

BODY = '''pub(crate) fn project_output(
    device: &mut Device,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    weight_scale: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, OutputEpilogueSlices, m![H % 480]> {
    type KCluster = m![Dummy2];
    type RowSlices = m![H % 256];

    // Rename only the contiguous high K bit: c owns k=2048*c+u.
    // Eight ring32 leaders per cluster load x; Switch materializes all replicas.
    let x: HbmTensorView<'_, bf16, Chip, m![Dummy2, Qs % 2048]> = unsafe { x.reshape() };
    let x: DmTensor<bf16, Chip, KCluster, m![Dummy256 / 32, 1 # 32], m![Qs % 2048]> = x.to_dm(&mut device.tdma);
    let x: DmTensor<bf16, Chip, KCluster, m![Dummy256], m![Qs % 2048]> = device.main
        .begin(x.view())
        .fetch::<m![1], m![Qs % 2048]>()
        .switch::<m![Dummy256], m![1]>(SwitchConfig::CustomBroadcast { ring_size: 32 })
        .collect::<m![Qs / 16 % 128], m![Qs % 16]>()
        .commit_trim::<m![Qs % 16]>()
        .commit();
    // Physical replicas already exist: rename slice coordinates to row owners.
    let x: DmTensor<bf16, Chip, KCluster, RowSlices, m![Qs % 2048]> = unsafe { x.reshape() };
    let x_trf: TrfTensor<bf16, Chip, KCluster, RowSlices, m![1], m![Qs % 2048]> = device.sub
        .begin(x.view())
        .fetch::<m![Qs / 16 % 128], m![Qs % 16]>()
        .collect::<m![Qs / 16 % 128], m![Qs % 16]>()
        .to_trf();

    // s owns h=256*t+s; local row segments are 2048 contiguous FP8 bytes.
    let weight: HbmTensorView<'_, f8e4m3, Chip, m![H, Dummy2, Qs % 2048]> = unsafe { weight.view().reshape() };
    let weight: DmTensor<f8e4m3, Chip, KCluster, RowSlices, m![H / 256, Qs % 2048]> = weight.to_dm(&mut device.tdma);
    let partial: DmTensor<f32, Chip, KCluster, RowSlices, m![H / 256, 1 # 2]> = device.main
        .begin(weight.view())
        .fetch::<m![H / 256, Qs / 16 % 128], m![Qs % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![H / 256, Qs / 16 % 128], m![Qs % 16]>()
        .contract_outer::<m![H / 256, Qs / 32 % 64], m![Qs % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H / 256]>()
        .contract_lane::<m![H / 256], m![1 # 8]>(LaneMode::Interleaved)
        // Fifteen rows cannot form integral two-F32 commits; keep 8-byte slots.
        .commit_trim::<m![1 # 2]>()
        .commit();

    // Typed DMA actually moves cross-cluster partials and preserves slot padding.
    // Only after gathering do adjacent H values form legal compact F32 pairs.
    let partial: DmTensor<f32, Chip, Cluster, OutputEpilogueSlices, m![Dummy2, H % 480, 1 # 2]> = partial.to_dm(&mut device.tdma);
    let partial: DmTensor<f32, Chip, Cluster, OutputEpilogueSlices, m![Dummy2, H % 480]> = device.main
        .begin(partial.view())
        .fetch::<m![Dummy2, H % 480], m![1 # 2]>()
        .collect::<m![Dummy2, H % 480], m![1 # 8]>()
        .transpose::<m![Dummy2, H / 2 % 240], m![H % 2 # 8]>()
        .commit_trim::<m![H % 2]>()
        .commit();

    // Tile removes a size-one logical cluster-partial axis, retaining its pitch.
    // F32 AddF -> MulF avoids an extra reduced-H DM write/read and casts only once.
    let second: VrfTensor<f32, Chip, Cluster, OutputEpilogueSlices, m![H % 480]> = device.sub
        .begin(partial.view().tile::<m![Dummy2], 1, m![1 # 2, H % 480]>(1))
        .fetch::<m![H / 8 % 60], m![H % 8]>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();
    let scale: DmTensor<bf16, Chip, Cluster, OutputEpilogueSlices, m![H % 480]> = weight_scale.to_dm(&mut device.tdma);
    let scale: VrfTensor<f32, Chip, Cluster, OutputEpilogueSlices, m![H % 480]> = device.sub
        .begin(scale.view())
        .fetch::<m![1], m![H % 480]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();
    device.main
        .begin(partial.view().tile::<m![Dummy2], 1, m![1 # 2, H % 480]>(0))
        .fetch::<m![H / 8 % 60], m![H % 8]>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, &second)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}
'''


def main():
    original = subprocess.check_output(["git", "show", BASE + ":" + TARGET], cwd=ROOT, text=True)
    start = original.index("pub(crate) fn project_output(")
    end = original.index("\npub(crate) type OutputEpilogueSlices", start)
    source = (original[:start] + BODY + original[end:]).replace(
        "{Ds, Dummy2, Gs, H, Ns, Ps, Qs}", "{Ds, Dummy2, Dummy256, Gs, H, Ns, Ps, Qs}")
    dest = HERE / "khalf-v1"
    dest.mkdir(exist_ok=False)
    (dest / "projection.rs").write_text(source)
    recipe = {
        "id": "M02", "variant": "khalf-v1", "base_commit": BASE,
        "comparison_recipe": "baseline/e104.json", "target_kernel": "sliding_attention_output",
        "changed_sources": {"projection.rs": "src/device/sliding/projection.rs"},
        "source_reference_build": "moa-dev/experiments/candidates/c07-native-scale-dpe/build-c07-e104-reviewed-v1-repro.json",
        "unchanged_kernels": ["sliding_project_qkv", "decoder_feedforward"],
        "scope": "whole O projection cluster/slice ownership and memory-dataflow redesign",
        "source_composition": {
            "origin": "New K-half cluster owner and H-interleaved slices; outside project_output E104 plus Dummy256 import",
            "arithmetic": "E104/E53 BF16 dot and F32 scale-before-first-BF16; reduction grouping changes 16x256 to 2x2048",
            "reused_methods": ["output-column-sweep 2cols: 15x2048 tile and padded scalar gather/compaction", "C14: cross-cluster F32 partial gathering", "existing custom ring broadcast and qkv-fp8-split AddF+MulF epilogue API pattern"],
            "removed": ["cluster H-half ownership", "16-column intra-cluster reduction", "120x256 weight tile", "full replicated input HBM ingress", "separate complete-dot scale helper"]
        },
        "graph": "HBM x -> ring32 leader DM -> Switch broadcast -> BF16 TRF; HBM W -> 15x2048 interleaved-row FP8 DM -> lookup+BF16 dot -> padded F32 partial DM -> cross-cluster H480 gather preserving8B slots -> TU compaction DM -> tile0 stream + tile1 VRF -> F32 AddF+channel MulF -> first BF16 -> existing post-RMS/residual/HBM",
        "prior_related": ["Old2cols already used15x2048 but H-owned clusters, BF16-before-scale and no K-half cluster sum; compile-only, weight DMA no gain", "M01 kept120x256 and H-owned clusters; extra partial paths regressed on NPU"],
        "cost_removed": ["16-column inter-slice reduction", "complete-dot DM write/read between F32 sum and scale"],
        "cost_added": ["2MiB activation replicas vs256KiB", "ring32 leader broadcast/setup", "61440B padded partial storage plus gathered copy", "30720B compact partial storage", "cross-cluster gather and F32 second-partial VRF loading"],
        "hypothesis": "Larger contiguous HBM row segments plus H interleave and removal of inter-slice reduction may offset input broadcast and partial movement; segment count is not measured DMA transaction count.",
        "first_stop_condition": "Whole-O compile and basic NPU with adjacent E104 controls. Repair concrete source mapping/API failures with frozen evidence. Close regression or <5% O gain without parameter sweeps; >=5% requires independent repeats and expanded correctness before adoption.",
        "adopted": False,
    }
    (dest / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    print("khalf-v1", hashlib.sha256(source.encode()).hexdigest())


if __name__ == "__main__":
    main()
