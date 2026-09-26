#!/usr/bin/env python3
"""Enumerate emitted source packets and prove bounds separately from runtime."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SLICE_STRIDE = 4194304


def digest(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def descriptors(description, side, region):
    rows = []
    for number, loops, address in re.findall(rf"^{side}(\d+): \[([^\]]+)\], {region}\(([^)]+)\)$", description, re.M):
        dimensions = [[name, int(count), int(stride)] for name, count, stride in re.findall(r"([^, ]+)=([0-9]+):([0-9]+)", loops)]
        assert len(dimensions) == len(loops.split(",")) and dimensions[-1][1] == 1
        fields = dict(re.findall(r"(\w+)=(0x[0-9A-Fa-f]+|[0-9]+)", address))
        fields = {key: int(value, 0) for key, value in fields.items()}
        rows.append({"number": int(number), "dimensions": dimensions, "address": fields,
                     "packet_bytes": dimensions[-1][2]})
    assert rows
    return rows


def packets(row):
    dimensions = row["dimensions"][:-1]
    for coordinate in itertools.product(*(range(count) for _, count, _ in dimensions)):
        offset = sum(value * dim[2] for value, dim in zip(coordinate, dimensions))
        yield offset, row["packet_bytes"]


def union_bytes(intervals):
    total, last = 0, -1
    for begin, end in sorted(intervals):
        total += max(0, end - max(begin, last))
        last = max(last, end)
    return total


def hbm_audit(graph):
    names = {1: "q_weight", 2: "k_weight", 3: "v_weight", 4: "q_scale", 5: "k_scale", 6: "v_scale"}
    result = {}
    for key, op in graph["operators"].items():
        if op["kind"] != "Dma.DtoS":
            continue
        tensor = graph["tensors"][str(op["inputs"][0])]
        binding = re.fullmatch(r"IO\((\d+)\)\+(\d+)", tensor["optionals"]["address"])
        if not binding or int(binding[1]) not in names:
            continue
        number, binding_offset = map(int, binding.groups())
        assert binding_offset == 0
        rows = descriptors(op["description"], "src", "Dram")
        intervals = []
        for row in rows:
            start = row["address"]["in_chip"]
            intervals.extend((start + offset, start + offset + size) for offset, size in packets(row))
        tensor_bytes = tensor["size"]
        volume = sum(end - begin for begin, end in intervals)
        result[names[number]] = {"graph_operator": key, "operator_name": op["name"],
            "source_tensor": op["inputs"][0], "source_binding": tensor["optionals"]["address"],
            "source_shape": tensor["shape"], "element_type": tensor["element_type"],
            "logical_tensor_bytes": tensor_bytes, "source_descriptors": rows,
            "packet_count": len(intervals), "descriptor_byte_volume": volume,
            "volume_over_logical_bytes": volume / tensor_bytes,
            "source_min_byte": min(begin for begin, _ in intervals),
            "source_end_exclusive": max(end for _, end in intervals),
            "unique_source_bytes": union_bytes(intervals),
            "out_of_bounds_packet_count": sum(begin < 0 or end > tensor_bytes for begin, end in intervals),
            "out_of_bounds_byte_volume": sum(max(0, end - max(begin, tensor_bytes)) for begin, end in intervals),
            "source_bounds_valid": all(0 <= begin < end <= tensor_bytes for begin, end in intervals),
            "full_description": op["description"]}
    assert set(result) == set(names.values())
    return result


def scalar_index(graph, tensor):
    index_access, = [op for op in graph["operators"].values() if tensor in op["outputs"] and op["kind"] == "IndexAccess"]
    scalar, = [op for op in graph["operators"].values() if index_access["inputs"][1] in op["outputs"]]
    value = int(re.search(r"__scalar_out\[\] = (\d+)i32", scalar["description"])[1])
    stride = int(re.search(r"axis: stride (\d+)", index_access["description"])[1])
    return value, stride, index_access, scalar


def sram_coordinate(row, offset):
    address = row["address"]
    local = address["in_slice"] + offset
    physical_slice = 256 * address["cluster"] + 64 * address["pe"] + 32 * address["dmn"] + address["slice"] + local // SLICE_STRIDE
    return physical_slice, local % SLICE_STRIDE


def output_gather_audit(graph):
    result = {}
    for key, op in graph["operators"].items():
        if op["kind"] != "Dma.StoS" or "QkvRow" not in op["description"]:
            continue
        index, stride, access, scalar = scalar_index(graph, op["inputs"][0])
        name = {0: "q", 2: "k", 3: "v"}[index]
        assert stride == 4
        src_rows = descriptors(op["description"], "src", "Sram")
        dst_rows = descriptors(op["description"], "dst", "Sram")
        assert len(src_rows) == len(dst_rows) == 8
        seen_source, seen_destination = set(), set()
        total = 0
        for src, dst in zip(src_rows, dst_rows):
            src_packets, dst_packets = list(packets(src)), list(packets(dst))
            assert len(src_packets) == len(dst_packets)
            for (src_offset, src_bytes), (dst_offset, dst_bytes) in zip(src_packets, dst_packets):
                assert src_bytes == dst_bytes == 8
                source_slice, source_local = sram_coordinate(src, src_offset)
                source_local += index * stride * 2  # BF16 IndexAccess uses element strides.
                dest_slice, dest_local = sram_coordinate(dst, dst_offset)
                assert 0 <= source_slice < 512 and 0 <= source_local < source_local + 8 <= 32
                assert 0 <= dest_slice < 8 and 0 <= dest_local < dest_local + 8 <= (1024 if name == "q" else 512)
                for byte in range(8):
                    matrix, local = divmod(source_local + byte, 8)
                    logical_source = (matrix if name == "q" else 0) * 4096 + source_slice * 8 + local
                    logical_destination = dest_slice * (1024 if name == "q" else 512) + dest_local + byte
                    assert logical_source == logical_destination
                    assert (source_slice, source_local + byte) not in seen_source
                    assert logical_destination not in seen_destination
                    seen_source.add((source_slice, source_local + byte))
                    seen_destination.add(logical_destination)
                total += 8
        expected = 8192 if name == "q" else 4096
        assert total == expected and seen_destination == set(range(expected))
        result[name] = {"graph_operator": key, "name": op["name"], "source_index": index,
            "source_index_element_stride": stride, "source_index_byte_offset": index * stride * 2,
            "logical_bytes": expected, "descriptor_byte_volume": total,
            "source_per_slice_byte_range": [min(x[1] for x in seen_source), max(x[1] for x in seen_source) + 1],
            "exact_source_to_destination_byte_mapping_verified": True,
            "source_within_joint_output_allocation": True, "all_destination_bytes_written_once": True,
            "index_access": access, "index_constant": scalar, "full_description": op["description"]}
    assert set(result) == {"q", "k", "v"}
    return result


def analyze(label):
    evidence = HERE / "evidence" / label
    build = json.loads((HERE / f"build-{label}.json").read_text())
    assert build["image_byte_identical_to_e105"] and build["exact_compile_exit_code"] == 0
    for name, expected in build["dumps"].items():
        assert digest(evidence / name) == expected
    graph = json.loads((evidence / "graph.json").read_text())
    ingress = hbm_audit(graph)
    for name, entry in ingress.items():
        assert entry["source_bounds_valid"] == name.startswith("q_")
        assert entry["volume_over_logical_bytes"] == (1 if name.startswith("q_") else 4)
    assert ingress["k_weight"]["source_end_exclusive"] == ingress["v_weight"]["source_end_exclusive"] == 7910400
    assert ingress["k_scale"]["source_end_exclusive"] == ingress["v_scale"]["source_end_exclusive"] == 4120
    return {"scope": "Exact-image E105 emitted descriptors; byte enumeration, not NPU transactions or timing",
            "build": digest(HERE / f"build-{label}.json"), "image": build["image"],
            "e105_image_identity_verified": True, "ingress": ingress, "output_gathers": output_gather_audit(graph),
            "conclusion": "K/V weights and scales read padded packet extents with the original source pitch, repeat bytes, and exceed the logical HBM source bounds. Q ingress and all three output gather descriptor mappings pass this audit. This is a concrete defect in E105's transfer contract; it does not prove there are no other arithmetic/layout defects or that a corrected implementation is faster.",
            "repair_design_cost_bytes": {
                "existing_weight_descriptor_volume": sum(ingress[n]["descriptor_byte_volume"] for n in ("q_weight", "k_weight", "v_weight")),
                "required_weight_hbm_bytes": 31457280,
                "existing_scale_descriptor_volume": sum(ingress[n]["descriptor_byte_volume"] for n in ("q_scale", "k_scale", "v_scale")),
                "required_scale_hbm_bytes": 16384,
                "compact_kv_weight_copy_dm_reads_plus_writes": 31457280,
                "compact_kv_scale_copy_dm_reads_plus_writes": 16384,
                "one_compact_kv_weight_temporary_bytes": 7864320,
                "scope": "Logical exact-copy lower bound; padding, allocator reservation, packet alignment, lifetime and engine costs require new compile. All-weight-before-dot dependency remains."}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--label", default="exact-v2")
    args = parser.parse_args()
    output = analyze(args.label)
    path = HERE / f"descriptor-analysis-{args.label}.json"
    if args.check:
        assert json.loads(path.read_text()) == output
    else:
        path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({name: {key: entry[key] for key in ("logical_tensor_bytes", "descriptor_byte_volume", "source_end_exclusive", "out_of_bounds_byte_volume", "source_bounds_valid")} for name, entry in output["ingress"].items()}, indent=2))
    print("Q/K/V output gather byte mappings: PASS")
