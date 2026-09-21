#!/usr/bin/env python3
"""Recompute schedule occupancy and a physical-page lifetime union metric."""
import collections
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE.parents[1] / "artifacts"


def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not result or start > result[-1][1]:
            result.append([start, end])
        else:
            result[-1][1] = max(result[-1][1], end)
    return result


def size(intervals):
    return sum(end - start for start, end in merge(intervals))


def physical_lifetime_union(data):
    page_size = data["page_size"]
    records = []
    for tensor in data["tensors"]:
        life = tensor["lifetime"]
        if tensor["buffer_type"] != "Sram" or life["end"] <= life["begin"]:
            continue
        offset = tensor["address"] % page_size
        assert math.ceil((offset + tensor["size"]) / page_size) == len(tensor["physical_pages"])
        remaining = tensor["size"]
        intervals = []
        for page in tensor["physical_pages"]:
            length = min(remaining, page_size - offset)
            intervals.append((page * page_size + offset, page * page_size + offset + length))
            remaining -= length
            offset = 0
        assert remaining == 0
        records.append((tensor, intervals))
    samples = []
    for cycle in sorted({tensor["lifetime"]["begin"] for tensor, _ in records}):
        active = [(tensor, intervals) for tensor, intervals in records
                  if tensor["lifetime"]["begin"] <= cycle < tensor["lifetime"]["end"]]
        count = size([interval for _, intervals in active for interval in intervals])
        samples.append((count, cycle, [tensor["index"] for tensor, _ in active]))
    count, cycle, active = max(samples, key=lambda item: item[:2])
    return {"peak_bytes": count, "cycle": cycle, "active_tensor_ids": active,
            "sram_tensors_with_nonempty_lifetime": len(records),
            "scope": "Schedule Sram tensor extents mapped through physical_pages, unioned at each half-open lifetime event. Aliases counted once; includes recorded padding. Not runtime telemetry or a separate proof of worst per-slice occupancy."}


def profile(path):
    data = json.loads(path.read_text())
    contexts = collections.defaultdict(list)
    for instruction in data["instructions"]:
        life = instruction["lifetime"]
        for context in instruction["contexts"]:
            contexts[context].append((life["begin"], life["end"]))
    main = merge(contexts["MainContext"])
    dma = merge(contexts["DmaEngine"])
    overlap = sum(max(0, min(b, d) - max(a, c)) for a, b in main for c, d in dma)
    result = {
        "makespan_cycles": max(x["lifetime"]["end"] for x in data["instructions"]),
        "instruction_count": len(data["instructions"]),
        "context_interval_union_cycles": {k: size(v) for k, v in contexts.items() if k != "Noop"},
        "main_dma_overlap_cycles": overlap,
        "sram_physical_lifetime_union": physical_lifetime_union(data),
    }
    return data, result


def main():
    _, baseline = profile(ARTIFACTS / "ffn-down-scale-coalesced/baseline.json")
    data, candidate = profile(ARTIFACTS / "ffn-up-tiled-fused-decode-e47/candidate.json")
    selected = [t for t in data["tensors"] if t["buffer_type"] == "Sram"
                and t["shape"].endswith("Element: (L % 32 = 8, H)]")]
    packed = [t for t in selected if t["size"] == 7864320]
    bf16 = [t for t in selected if t["size"] == 31457280]
    vrf = sorted([t for t in data["tensors"] if t["buffer_type"] == "Vrf"
                  and "L % 32 = 8, H / 16" in t["shape"]],
                 key=lambda t: t["lifetime"]["begin"])
    assert len(packed) == len(bf16) == 4
    assert len(vrf) == 8  # Four original gate tiles and four staged up tiles.
    assert len({t["address"] for t in vrf}) == 1
    assert all(a["lifetime"]["end"] <= b["lifetime"]["begin"] for a, b in zip(vrf, vrf[1:]))
    def max_live(tensors):
        return max(sum(t["lifetime"]["begin"] <= cycle < t["lifetime"]["end"] for t in tensors)
                   for cycle in {t["lifetime"]["begin"] for t in tensors})
    candidate["up_weight_tensors"] = {
        "packed": packed, "bf16": bf16,
        "maximum_simultaneously_live_packed": max_live(packed),
        "maximum_simultaneously_live_bf16": max_live(bf16),
        "scale_vrf_tensors": vrf,
        "scale_vrf_lifetimes_nonoverlapping": True,
        "scale_vrf_bytes_per_live_slice": 7680,
    }
    print(json.dumps({"baseline": baseline, "candidate": candidate}, indent=2))


if __name__ == "__main__":
    main()
