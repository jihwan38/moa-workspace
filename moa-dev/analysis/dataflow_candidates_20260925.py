#!/usr/bin/env python3
"""Check proposed tensor ownership and logical byte ledgers, without compiling kernels.

Run from any directory. This is a coordinate model, not a DMA descriptor, allocator,
floating-point correctness, or latency model. Candidate formulas are documented in
docs/research/rngd-dataflow-candidates-20260925.md.
"""

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AXES = ROOT / "furiosa-opt-gemma4-12B/src/axes.rs"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_matrix(rows, columns, regions, bits):
    """Each region is (cluster, slice, row, column_start, column_count)."""
    coverage = defaultdict(list)
    owner_elements = defaultdict(int)
    starts_256 = lengths_256 = count = total_elements = 0
    for cluster, slice_id, row, start, size in regions:
        require(0 <= cluster < 2 and 0 <= slice_id < 256, "invalid physical owner")
        require(0 <= row < rows and 0 <= start < columns, "invalid matrix coordinate")
        require(size > 0 and start + size <= columns, "invalid row segment")
        address_bits, length_bits = (row * columns + start) * bits, size * bits
        require(address_bits % 8 == 0 and length_bits % 8 == 0, "partial byte")
        address, length = address_bits // 8, length_bits // 8
        require(address % 8 == 0 and length % 8 == 0, "8-byte source constraint")
        starts_256 += address % 256 == 0
        lengths_256 += length % 256 == 0
        coverage[row].append((start, start + size))
        owner_elements[cluster, slice_id] += size
        count += 1
        total_elements += size
    require(len(coverage) == rows, "missing row")
    for row, intervals in coverage.items():
        cursor = 0
        for start, end in sorted(intervals):
            require(start == cursor, f"overlap or hole in row {row}")
            cursor = end
        require(cursor == columns, "incomplete row")
    require(len(owner_elements) == 512, "not all slices assigned useful weights")
    require(total_elements == rows * columns, "weight coverage mismatch")
    return {
        "coverage": "every weight element exactly once",
        "all_512_slices_have_work": True,
        "row_segments_not_dma_descriptors": count,
        "source_8_byte_alignment": "pass, assuming aligned matrix base",
        "segments_with_256_byte_start": starts_256,
        "segments_with_256_byte_length": lengths_256,
        "unique_payload_bytes": total_elements * bits // 8,
        "max_weight_payload_bytes_per_slice": max(owner_elements.values()) * bits // 8,
    }


def main():
    axes = {k: int(v) for k, v in re.findall(r"\b(\w+)\s*=\s*(\d+)", AXES.read_text())}
    h, l, q, p = (axes[k] for k in ("H", "L", "Qs", "Ps"))
    require((h, l, q, p) == (3840, 15360, 4096, 2048), "redesign formulas for new shapes")

    m01_regions = [(c, 16 * k + r, 1920 * c + 120 * r + t, 256 * k, 256)
                   for c in range(2) for k in range(16) for r in range(16) for t in range(120)]
    m02_regions = [(c, s, 256 * t + s, 2048 * c, 2048)
                   for c in range(2) for s in range(256) for t in range(15)]
    m03_regions = [(c, 4 * r + k, 60 * r + t, 7680 * c + 1920 * k, 1920)
                   for c in range(2) for r in range(64) for k in range(4) for t in range(60)]
    m03_scale_regions = [(c, s, row, start // 16, size // 16)
                         for c, s, row, start, size in m03_regions]
    # Up/gate local 30-row producers and paired 60-row consumers must agree.
    for c in range(2):
        produced = [7680 * c + 30 * s + t for s in range(256) for t in range(30)]
        paired = [7680 * c + 60 * pair + t for pair in range(128) for t in range(60)]
        require(produced == paired, "local 30 -> 60 row pairing changes order")
        consumed = [7680 * c + 1920 * k + u for k in range(4) for u in range(1920)]
        require(produced == consumed, "GeGLU producer / down input ownership mismatch")

    # Each M01 transposed slice retains exactly its (row-group, K-segment) pair.
    for _, s, _, start, _ in m01_regions:
        k, r = divmod(s, 16)
        target = 16 * r + k
        require(target % 16 == start // 256, "partial transpose mixes K segments")

    m01 = check_matrix(h, q, m01_regions, 8)
    m01.update({
        "f32_partial_useful_bytes": 512 * 120 * 4,
        "partial_repack_only_read_write_useful_bytes": 2 * 512 * 120 * 4,
        "dma_path_all_extra_partial_edges_useful_bytes": 4 * 512 * 120 * 4,
        "fused_switch_path_extra_dm_edges_excluding_switch_traffic_bytes": 2 * 512 * 120 * 4,
        "e64_full_weight_repack_write_read_useful_bytes": 2 * h * q,
        "bf16_activation_replicas_useful_bytes": 512 * 256 * 2,
        "trf_input_useful_bytes_per_slice": 256 * 2,
    })
    m02 = check_matrix(h, q, m02_regions, 8)
    m02.update({
        "rows_per_slice": 15,
        "f32_partial_useful_bytes": 2 * h * 4,
        "f32_partial_8byte_slot_storage_bytes": 512 * 15 * 8,
        "bf16_activation_replicas_useful_bytes": 512 * 2048 * 2,
        "activation_replica_ratio_vs_e104": 8,
        "ring32_input_leader_payload_bytes": 2 * 8 * 2048 * 2,
        "trf_input_useful_bytes_per_slice": 2048 * 2,
    })
    m03 = {
        "down_weight": check_matrix(h, l, m03_regions, 4),
        "down_block_scale": check_matrix(h, l // 16, m03_scale_regions, 8),
        "producer_pairing_and_consumer_coverage": "pass, all 15360 L coordinates",
        "down_activation_replicas_useful_bytes": 512 * 1920 * 2,
        "e104_down_activation_replicas_useful_bytes": 512 * 1920 * 2,
        "ring32_leader_payload_bytes": 2 * 8 * 4 * 1920 * 2,
        "e104_ring32_leader_payload_bytes": 2 * 8 * 8 * 1920 * 2,
        "down_f32_cluster_partial_useful_bytes": 2 * h * 4,
        "down_trf_input_useful_bytes_per_slice": 1920 * 2,
        "down_f32_block_partial_bytes_per_slice": 60 * 120 * 4,
        "down_scale_vrf_useful_bytes_per_slice": 12 * 120 * 4,
        "upgate_scale_vrf_useful_bytes_per_slice": 6 * 240 * 4,
        "stage_count": 1,
        "matrix_decode_streams": 3,
        "early_down_overlap_claimed": False,
    }
    payloads = {
        "qkv": {"weight_bytes": (q + 2 * p) * h, "scale_bytes": (q + 2 * p) * 2},
        "output": {"weight_bytes": h * q, "scale_bytes": h * 2},
        "ffn": {"weight_bytes": 3 * h * l // 2, "block_scale_bytes": 3 * h * l // 16},
    }
    report = {
        "schema_version": 1,
        "scope": "logical coordinates and payload accounting only",
        "axes_sha256": hashlib.sha256(AXES.read_bytes()).hexdigest(),
        "dimensions": {k: axes[k] for k in ("H", "L", "Qs", "Ps")},
        "payloads_exclude_scalar_and_norm_parameters": payloads,
        "candidates": {"M01": m01, "M02": m02, "M03": m03},
        "limitations": [
            "No Rust candidate exists and no compilation or NPU execution is implied.",
            "Row segments are not compiled DMA descriptors or measured transactions.",
            "Payload bytes exclude allocator padding, decoder tables and bus amplification.",
            "Logical TRF/VRF footprints do not establish physical allocation feasibility.",
            "No floating-point accuracy, engine overlap or latency prediction.",
        ],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
