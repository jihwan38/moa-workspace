#!/usr/bin/env python3
"""Couple native FP8 contraction, streamed FP4 unpack and a second scale DPE.

These are experimental approximations, not an exact BF16 decomposition claim.
Starting sources are our E41 native quantizer/row layout and E79 epilogue/down.
"""
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = "2735b5899f41195dedeab1e076c82317e24efcb2"
SOURCE = "furiosa-opt-gemma4-12B/src/device/shared/mlp.rs"


def generate():
    e79 = subprocess.check_output(["git", "show", f"{BASE}:{SOURCE}"], cwd=ROOT, text=True)
    e41 = (HERE.parent / "ffn-native-fp8-row30/mlp.rs").read_text()
    prefix = e41.split("pub(crate) fn feedforward(")[0]
    suffix = "pub(crate) fn feedforward(" + e79.split("pub(crate) fn feedforward(", 1)[1]
    start = suffix.index("    // Broadcast the normalized")
    end = suffix.index("    let x = geglu", start)
    new = e41[e41.index("    // Quantize the actual"):e41.index("    let x = geglu")]
    suffix = suffix[:start] + new + suffix[end:]
    quant = e41[e41.index("type FfnQuantSlices"):]

    # FP4 is unpacked directly into the native FP8 DPE stream; no FP8 weight DM.
    prefix = prefix.replace("weights: &DmTensor<f8e4m3,", "weights: &DmTensor<f4e2m1,")
    prefix = prefix.replace(".fetch::<m![FfnRow, H / 32], m![H % 32]>()\n        .collect", ".fetch::<m![FfnRow], m![H]>()\n        .fetch_table_lookup::<f8e4m3>()\n        .collect", 1)
    for which in ("up", "gate"):
        start = prefix.index("    // Decode all30")
        end = prefix.index(f"    let {which}_weight_scale:", start)
        prefix = prefix[:start] + prefix[end:]
    # FP8 scales widen exactly through F32 on Sub; no Main LUT setup.
    old = """        .main.begin(scales.view())
        .fetch::<m![FfnRow, H / 16 / 16], m![H / 16 % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![FfnRow, H / 16 / 16], m![H / 16 % 16]>()
        .commit_trim::<m![H / 16 % 16]>()"""
    new = """        .sub.begin(scales.view())
        .fetch::<m![FfnRow, H / 128], m![H / 16 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![FfnRow, H / 128], m![H / 16 % 8]>()
        .cast::<bf16, m![H / 16 % 8 # 16]>()
        .commit_trim::<m![H / 16 % 8]>()"""
    assert old in prefix
    prefix = prefix.replace(old, new)
    variants = {"up-stream-pair-dpe": prefix + suffix + "\n" + quant}
    # A separate timing architecture puts terms on the DPE time axis. Both
    # terms share a power-of-two scale, so the DPE can sum them without VE zip.
    # The unamplified FP8 residual has a larger underflow region than pair16.
    time = variants["up-stream-pair-dpe"]
    time = time.replace("m![FfnTerm], m![H]>", "m![1], m![FfnTerm, H]>")
    time = time.replace(".fetch::<m![1], m![FfnTerm, H]>()", ".fetch::<m![FfnTerm], m![H]>()")
    start = time.index("        .contract_outer::<m![FfnRow, H / 64]")
    end = time.index("        .cast::<bf16, m![H / 16 % 4 # 16]>()", start)
    time = time[:start] + """        .contract_outer::<m![FfnRow, H / 64, FfnTerm], m![H % 64], _, _, _>(x_trf)
        .contract_packet::<m![H / 16 % 4]>()
        .contract_time::<m![FfnRow, H / 64]>()
        .contract_lane::<m![FfnRow, H / 64], m![H / 16 % 4 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![H / 16 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), x_scale_vrf)
        .vector_widen_pad::<m![H / 16 % 4 # 8]>()
        .vector_final()
""" + time[end:]
    time = time.replace("        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 16.0f32)\n", "")
    variants["up-stream-time-dpe"] = time
    for name, source in list(variants.items()):
        contiguous = source.replace("Dummy2, FfnSlice / 2, FfnRow, FfnSlice % 2,", "Dummy2, FfnSlice, FfnRow,")
        contiguous = contiguous.replace("DmTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![FfnRow, 1 # 4]>", "DmTensor<f32, Chip, UpGateCluster, UpGateSlices, m![FfnRow]>")
        contiguous = contiguous.replace("        .cast::<bf16, m![1 # 16]>()\n        .commit_trim::<m![1 # 4]>()", "        .transpose::<m![FfnRow / 2], m![FfnRow % 2 # 8]>()\n        .commit_trim::<m![FfnRow % 2]>()", 1)
        a = contiguous.index("    // L = 7680*cluster")
        b = contiguous.index("pub(crate) fn feedforward(", a)
        contiguous = contiguous[:a] + """    // F32 rows allow 8-byte commits for 30-row slices. Cast after the gather.
    let up: DmTensor<f32, Chip, m![L / 7680], m![L / 30 % 256], m![L % 30]> = unsafe { up.reshape() };
    let gate: DmTensor<f32, Chip, m![L / 7680], m![L / 30 % 256], m![L % 30]> = unsafe { gate.reshape() };
    let up: DmTensor<f32, Chip, Cluster, UpGateRows, m![L % 60]> = up.to_dm(&mut device.tdma);
    let gate: DmTensor<f32, Chip, Cluster, UpGateRows, m![L % 60]> = gate.to_dm(&mut device.tdma);
    (cast_gathered_rows(device, &up), cast_gathered_rows(device, &gate))
}

fn cast_gathered_rows(device: &mut Device, x: &DmTensor<f32, Chip, Cluster, UpGateRows, m![L % 60]>)
    -> DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]> {
    device.main.begin(x.view()).fetch::<m![L / 4 % 15], m![L % 4 # 8]>()
        .collect::<m![L / 4 % 15], m![L % 4 # 8]>()
        .cast::<bf16, m![L % 4 # 16]>().commit_trim::<m![L % 4]>().commit()
}

""" + contiguous[b:]
        variants[name + "-contiguous"] = contiguous
    # A coherent wider-packet design: reduce both terms in the DPE, retain
    # eight block sums, and restore the common power-of-two scale after DPE 2.
    for width in (64, 128, 256):
        wide = variants["up-stream-time-dpe-contiguous"]
        a = wide.index("        .vector_init()")
        b = wide.index("        .cast::<bf16", a)
        wide = wide[:a] + wide[b:]
        a = wide.index("    let scales: DmTensor<bf16")
        first = wide[:a].replace("H / 64", f"H / {width}").replace("H % 64", f"H % {width}")
        first = first.replace("H / 16 % 4", f"H / 16 % {width // 16}")
        wide = first + wide[a:]
        old = "        .contract_lane::<m![FfnRow], m![1 # 8]>(LaneMode::Interleaved)\n"
        new = old + """        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), x_scale_vrf)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
"""
        assert wide.count(old) == 1
        variants[f"up-time-packet{width}-late-scale"] = wide.replace(old, new)
    from down import native_down
    for name, source in list(variants.items()):
        variants[name.replace("up-", "full-")] = native_down(source, variants["up-stream-pair-dpe"])
    from extensions import time_down, two_columns
    base = variants["full-time-packet64-late-scale"]
    variants["full-time4-late-scale"] = time_down(base)
    variants["full-time2-late-scale"] = two_columns(time_down(base))
    variants["full-pair2-late-scale"] = two_columns(base)
    from precision import f32_blocks, e5_three_terms, compensated_scale_dpe, reload_projection_input, materialize_fp8
    variants["up-time-f32-ve"] = f32_blocks(variants["up-stream-time-dpe-contiguous"])
    variants["full-time4-f32-ve"] = f32_blocks(f32_blocks(time_down(variants["full-stream-time-dpe-contiguous"])), down=True)
    variants["up-e5-three-dpe"] = e5_three_terms(variants["up-stream-time-dpe-contiguous"])
    variants["up-e5-three-f32-ve"] = e5_three_terms(variants["up-time-f32-ve"])
    variants["up-e5-compensated-dpe"] = compensated_scale_dpe(variants["up-e5-three-dpe"])
    variants["up-e5-f32-reload"] = reload_projection_input(variants["up-e5-three-f32-ve"])
    variants["up-e5-f32-materialized"] = materialize_fp8(variants["up-e5-three-f32-ve"])
    variants["up-e5-f32-reload-materialized"] = materialize_fp8(variants["up-e5-f32-reload"])
    for original in ("up-e5-three-f32-ve", "up-e5-compensated-dpe"):
        four = variants[original].replace("FfnTerm = 3", "FfnTerm = 4").replace("FfnTerm = 1 #{!} 3", "FfnTerm = 1 #{!} 4")
        marker = "    hi.view().to_dm_view(&mut device.tdma,\n        packed.view_mut()"
        assert four.count(marker) == 1
        zero = """    let mut zero: DmTensor<f32, Chip, m![2], FfnQuantSlices, m![H % 480]> = DmTensor::new();
    zero.view_mut().memset(0.0f32, &mut device.sub);
    device.sub.begin(zero.view()).fetch::<m![H / 8 % 60], m![H % 8]>()
        .collect::<m![H / 8 % 60], m![H % 8]>().cast::<f8e5m2, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>()
        .commit_view(packed.view_mut().tile::<m![FfnTerm], 1, m![FfnTerm = 1 #{!} 4, H % 480]>(3));
"""
        four = four.replace(marker, zero + marker)
        variants[original + "-term4"] = four
    # Keep the first native DPE's F32 stream free of a vector adapter and
    # restore its common scale in the F32 scale/reduction pipeline.
    for original in ("up-e5-three-f32-ve", "up-e5-three-f32-ve-term4"):
        late = variants[original]
        a = late.index("        .vector_init()")
        b = late.index("        .commit_trim::<m![H / 16 % 4]>()", a)
        late = late[:a] + late[b:]
        marker = "            .vector_intra_slice_reduce::<H, m![FfnRow = 6], m![1 # 4]>(IntraSliceReduceOpF32::Add)"
        assert late.count(marker) == 1
        late = late.replace(marker, "            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), x_scale_vrf)\n" + marker)
        variants[original + "-late"] = late
    # Isolate quantized-input delivery: direct typed DMA replication instead
    # of leader DMA followed by a Main ring broadcast. This is a diagnostic
    # architecture, not an assertion that broadcast caused prior corruption.
    direct = variants["up-e5-f32-reload"]
    a = direct.index("    let x: DmTensor<f8e5m2", direct.index("fn broadcast_ffn_quantized("))
    b = direct.index("pub(crate) fn prepare_ffn_fp8_split(", a)
    direct = direct[:a] + "    x.to_dm(&mut device.tdma)\n}\n\n" + direct[b:]
    variants["up-e5-f32-direct-dma"] = direct
    manifest = {"base": BASE, "sources": {"E41": hashlib.sha256(e41.encode()).hexdigest(), "E79": hashlib.sha256(e79.encode()).hexdigest()}, "variants": {}}
    for name, source in variants.items():
        path = HERE / name / "mlp.rs"
        path.parent.mkdir(exist_ok=True)
        path.write_text(source)
        manifest["variants"][name] = {"sha256": hashlib.sha256(source.encode()).hexdigest(), "bytes": len(source.encode())}
    (HERE / "generated.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    generate()
