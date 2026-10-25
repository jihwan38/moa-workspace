#!/usr/bin/env python3
"""Read-only public-source/evidence capture. No compilation or NPU calls."""
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REV = "5ec376347e108689096928bc4be2c0e4176fd7c9"
BASE = f"https://raw.githubusercontent.com/furiosa-ai/furiosa-opt/{REV}/"
REG = "/home/moa/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f"
PATHS = [
    "docs/src/computing-tensors/contraction-engine/outer.md",
    "docs/src/computing-tensors/contraction-engine/packet-reducer.md",
    "docs/src/computing-tensors/register-files.md",
    "furiosa-opt-std/src/cast.rs",
    "furiosa-opt-std/src/scalar.rs",
    "furiosa-opt-std/src/engine/contraction/outer/mod.rs",
    "furiosa-opt-std/src/engine/contraction/outer/trf_sequencer.rs",
    "furiosa-opt-std/src/engine/contraction/lane.rs",
    "furiosa-opt-lower/src/verify/mod.rs",
    "furiosa-opt-lower/src/verify/engine/contraction/outer/stream_adapter.rs",
    "furiosa-opt-lower/src/verify/engine/contraction/packet.rs",
]
ARCHIVES = [
    "batch-kernels/src/ops_native_i4_p512_bound.rs",
    "batch-kernels/experiments/batch-kpart/int4-bound-result.md",
    "batch-kernels/experiments/batch-kpart/int4-bound-manifest.json",
    "batch-kernels/experiments/batch-kpart/int4-bound-actual-trace.json",
    "batch-kernels/experiments/batch-kpart/int4-up-packet128-rejected.rs",
    "batch-kernels/experiments/batch-kpart/logs/int4-up-compile4.log",
    "runs/r726-native-int4-bound-submit/arena.log",
    "int4-raw-binding/src/ops_i4_raw_binding.rs",
    "int4-raw-binding/experiments/int4-raw-binding/result.md",
]

def digest(b):
    return hashlib.sha256(b).hexdigest()

def fetch(path):
    raw = urllib.request.urlopen(BASE + path, timeout=60).read()
    item = {"path": path, "url": BASE + path, "sha256": digest(raw), "bytes": len(raw)}
    if not path.startswith("docs/"):
        package, relative = path.split("/", 1)
        installed = f"{REG}/{package}-0.9.0/{relative}"
        local = subprocess.check_output(["docker", "exec", "moa2026-round2", "cat", installed])
        assert raw == local, f"public pinned source differs from installed crate: {path}"
        item["installed_path"] = installed
        item["installed_byte_identical"] = True
    return path, raw.decode(), item

def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        fetched = list(pool.map(fetch, PATHS))
    text = {p: s for p, s, _ in fetched}
    assert "PackSize = 1" in text[PATHS[0]] and "effective throughput halves" in text[PATHS[0]]
    assert "not wired, so an `i4` weight stays `i4` today" in text[PATHS[2]]
    assert "matches!(size, 1 | 2 | 4 | 8)" in text[PATHS[8]]
    casts = text[PATHS[3]]
    assert "impl ContractionWeight<i4> for i4 {}" in casts
    assert "impl ContractionWeight<i8> for i4" not in casts
    assert "impl ContractionWeight<i4> for i8" not in casts
    assert "TODO: validate the TRF-side expansion" in text[PATHS[6]]
    archives = []
    for rel in ARCHIVES:
        b = (ROOT / rel).read_bytes()
        archives.append({"relative_to_round2": rel, "sha256": digest(b), "bytes": len(b)})
    rejected = (ROOT / ARCHIVES[5]).read_text()
    for token in ("element_type: RawInt5", "collect_flits: Flit2", "invalid collect_flits"):
        assert token in rejected
    trace = json.loads((ROOT / ARCHIVES[3]).read_text())
    fp8_tu = [v["duration"] for v in trace["0"] if v["name"] == "Renegade::TuExec"]
    i4_tu = [v["duration"] for v in trace["1"] if v["name"] == "Renegade::TuExec"]
    assert fp8_tu == [41187, 59825, 8027, 42062, 58078, 8027, 42116, 58078, 8027, 41817, 58075, 8027]
    assert i4_tu == [19903, 116499, 15707, 20127, 116523, 15707]
    summary = {
        "captured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sdk": "0.9.0", "public_revision": REV,
        "public_sources": [v for _, _, v in fetched], "archived_evidence": archives,
        "r726_actual_raw_profiler_units": {
            "task": [577902, 395731], "weight_dma": [190907, 96709],
            "switch_trf": [sum(fp8_tu[::3]), sum(i4_tu[::3])],
            "contraction_reduce_scale": [sum(fp8_tu[1::3]), sum(i4_tu[1::3])],
            "transpose": [sum(fp8_tu[2::3]), sum(i4_tu[2::3])],
            "clock_conversion_assumed": False,
            "overlapping_spans_are_not_additive": True,
        },
        "conclusion": "HOLD full-rate INT4 assumption: hardware/doc support exists, but no proven public SDK0.9 compiled full-rate form was found. Legal K64 movement gain is real; failed K128 is preserved and was not retried.",
        "actions": {"builds": 0, "npu_dispatches": 0, "sdk_edits": 0, "production_edits": 0},
    }
    (HERE / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"public_sources": len(fetched), "installed_byte_equal": 8,
                      "archives": len(archives), "trace": summary["r726_actual_raw_profiler_units"]}, indent=2))

if __name__ == "__main__":
    main()
