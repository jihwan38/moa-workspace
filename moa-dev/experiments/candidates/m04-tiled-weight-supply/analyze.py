#!/usr/bin/env python3
"""Check row coverage and summarize actual lowered producer/consumer intervals."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def digest(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def intervals(nodes):
    return [{"index": n["index"], **n["lifetime"]} for n in nodes]


def peak_bytes(tensors):
    events = []
    for t in tensors:
        events += [(t["lifetime"]["begin"], t["size"]), (t["lifetime"]["end"], -t["size"])]
    current = peak = 0
    for _, delta in sorted(events):
        current += delta
        peak = max(current, peak)
    return peak


def analyze(path, tiled):
    s = json.loads(path.read_text())
    ins = s["instructions"]
    main = [n for n in ins if n["tpe"] == "Main"]
    dma = [n for n in ins if n["tpe"].startswith("Dma")]
    expand = [n for n in main if "fetch_table_lookup::<f8e" in n["description"]]
    dots = [n for n in main if "contract_outer" in n["description"]]
    weight = [n for n in dma if n["tpe"] == "DmaLoad" and
              ("weight\n.tile" in n["description"] if tiled else "weight_packed.to_dm" in n["description"])]
    norm = [n for n in main if "src/device/shared/rmsnorm.rs:95:" in n["description"]]
    amax = [n for n in main if "0x7fff_ffff" in n["description"]]
    assert len(norm) == len(amax) == 1
    assert len(weight) == len(expand) == len(dots) == (15 if tiled else 3)
    t = {str(t["index"]): t for t in s["tensors"]}
    expanded = [t[str(i)] for n in expand for i in n["output_tensors"]]
    assert len(expanded) == len(expand)
    upgate_dot = [n for n in dots if "FfnRow" in n["description"]]
    down_dot = [n for n in dots if "FfnRow" not in n["description"]]
    upgate_expand = [n for n in expand if "FfnRow" in n["description"]]
    down_expand = [n for n in expand if "FfnRow" not in n["description"]]
    return {
        "schedule": digest(path.read_bytes()),
        "instruction_count": len(ins),
        "makespan": max(n["lifetime"]["end"] for n in ins),
        "main_duration_sum": sum(n["lifetime"]["end"]-n["lifetime"]["begin"] for n in main),
        "dma_duration_sum": sum(n["lifetime"]["end"]-n["lifetime"]["begin"] for n in dma),
        "dma_count": len(dma),
        "pre_rms_last_main": intervals(norm)[0],
        "activation_amax": intervals(amax)[0],
        "pre_rms_to_amax_gap": amax[0]["lifetime"]["begin"]-norm[0]["lifetime"]["end"],
        "weight_loads": intervals(weight),
        "upgate_expand": intervals(upgate_expand),
        "upgate_dot": intervals(upgate_dot),
        "down_expand": intervals(down_expand),
        "down_dot": intervals(down_dot),
        "fp8_buffers": expanded,
        "fp8_buffer_peak_live_allocated_bytes": peak_bytes(expanded),
        "fp8_logical_write_bytes": 3*3840*15360,
        "limits": "Compiler intervals/allocation, not measured bus traffic or hardware stalls. Context-duration sums overlap; do not add as runtime savings. Peak is only selected FP8 buffers, not all SRAM.",
    }


def coordinates():
    # Enumerate HBM row segments, not floating point behavior. NPU host checks
    # the arithmetic. Every packed value and scale is covered exactly once.
    up, down, scales = [], [], []
    for c in range(2):
        for s in range(256):
            for i in range(5):
                for r in range(6):
                    l = 7680*c+30*s+6*i+r
                    up.append((l*3840//2, (l+1)*3840//2))
                for r in range(12):
                    h, col = 1920*c+60*(s//8)+12*i+r, s%8
                    down.append(((h*15360+1920*col)//2, (h*15360+1920*(col+1))//2))
                    scales.append((h*960+120*col, h*960+120*(col+1)))
    for seq, total in [(up,3840*15360//2), (down,3840*15360//2), (scales,3840*15360//16)]:
        seq.sort()
        assert seq[0][0] == 0 and seq[-1][1] == total
        assert all(a[1] == b[0] for a,b in zip(seq,seq[1:]))
    assert 6*4 % 8 == 12*2 % 8 == 0
    return {"upgate_unique_rows_per_projection": len(up), "down_unique_row_segments": len(down),
            "exact_weight_and_down_scale_coverage": True,
            "output_tile_bytes": 24, "packed_tile_bytes_per_slice": 11520,
            "fp8_tile_bytes_per_slice": 23040, "partial_tile_bytes_per_slice": 5760,
            "scope": "Coordinate/alignment accounting only; not arithmetic validation."}


if __name__ == "__main__":
    candidate = ROOT / "moa-dev/experiments/artifacts/m04-tiled-weight-supply/m04-row-tiles-v1/schedule.json"
    frozen = HERE / "evidence/m04-row-tiles-v1/schedule.json"
    if frozen.exists():
        assert frozen.read_bytes() == candidate.read_bytes()
    else:
        shutil.copyfile(candidate, frozen)
    control = HERE.parent / "e104-weight-path/evidence/profile-v1/schedule.json"
    result = {"control": analyze(control, False), "candidate": analyze(frozen, True),
              "coordinates": coordinates()}
    result["static_change_percent"] = 100*(result["candidate"]["makespan"]/result["control"]["makespan"]-1)
    (HERE / "static-analysis.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: {f:v[f] for f in ["makespan", "dma_count", "main_duration_sum", "dma_duration_sum", "pre_rms_to_amax_gap", "fp8_buffer_peak_live_allocated_bytes"]}
                      for k,v in result.items() if k in {"control", "candidate"}}, indent=2))
