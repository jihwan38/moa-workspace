#!/usr/bin/env python3
"""Gate corrected ingress on complete descriptor and raw-copy byte coverage."""
import argparse
import itertools
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
from analyze import (HERE, digest, hbm_audit, output_gather_audit, scalar_index,
                     descriptors, packets, sram_coordinate, union_bytes)


def ingress_ownership(graph, ingress):
    result = {}
    for name, entry in ingress.items():
        op = graph["operators"][entry["graph_operator"]]
        source = descriptors(op["description"], "src", "Dram")
        destination = descriptors(op["description"], "dst", "Sram")
        per_slot = 15360 if name.endswith("weight") else 8
        q = name.startswith("q_")
        spans = {i: [] for i in range(512)}
        assert len(source) == len(destination) == 8
        for src, dst in zip(source, destination):
            left, right = list(packets(src)), list(packets(dst))
            assert len(left) == len(right)
            for (offset, length), (dest_offset, dest_length) in zip(left, right):
                physical_slice, local = sram_coordinate(dst, dest_offset)
                assert length == dest_length and 0 <= physical_slice < 512
                matrix, row_offset = divmod(local, per_slot)
                assert matrix < (2 if q else 1) and row_offset + length <= per_slot
                expected_source = matrix * 512 * per_slot + physical_slice * per_slot + row_offset
                assert src["address"]["in_chip"] + offset == expected_source
                spans[physical_slice].append((local, local + length))
        expected_bytes = (2 if q else 1) * per_slot
        assert all(union_bytes(intervals) == sum(b - a for a, b in intervals) == expected_bytes
                   and min(a for a, _ in intervals) == 0 and max(b for _, b in intervals) == expected_bytes
                   for intervals in spans.values())
        result[name] = {"each_hbm_packet_reaches_its_expected_physical_slice_and_local_row": True,
                        "per_slice_compact_or_q_joint_bytes": expected_bytes,
                        "all_destination_bytes_written_once": True}
    return result


def sequence(description, name):
    raw = re.search(rf"^ *{name}: 0\+\[([^\]]+)\]$", description, re.M)[1]
    dimensions = [(int(count), int(stride)) for count, stride in re.findall(r"=(\d+):(\d+)", raw)]
    return [sum(value * dimensions[i][1] for i, value in enumerate(values))
            for values in itertools.product(*(range(count) for count, _ in dimensions))]


def result(label):
    build = json.loads((HERE / f"build-{label}.json").read_text())
    assert build["exact_compile_exit_code"] == 0
    evidence = HERE / "evidence" / label
    for name, expected in build["dumps"].items():
        assert digest(evidence / name) == expected
    graph = json.loads((evidence / "graph.json").read_text())
    ingress = hbm_audit(graph)
    assert all(row["source_bounds_valid"] and row["unique_source_bytes"] == row["descriptor_byte_volume"] == row["logical_tensor_bytes"] for row in ingress.values())
    copies = {}
    for name in ("k_weight", "v_weight", "k_scale", "v_scale"):
        dma = graph["operators"][ingress[name]["graph_operator"]]
        op_key, op = next((key, op) for key, op in graph["operators"].items()
                          if op["kind"] == "TuExec" and dma["outputs"][0] in op["inputs"])
        assert op["context"] == "Sub" and len(op["inputs"]) == 2
        index, stride, access, scalar = scalar_index(graph, op["inputs"][1])
        assert index == (2 if name.startswith("k") else 3)
        element_bytes = 1 if name.endswith("weight") else 2
        elements = 15360 if element_bytes == 1 else 4
        assert stride == elements
        assert sequence(op["description"], "fetch_sequencer") == sequence(op["description"], "commit_projection") == list(range(elements))
        assert f"commit_in_size: {32 if element_bytes == 1 else 8}B" in op["description"]
        assert graph["tensors"][str(op["inputs"][0])]["size"] == elements * element_bytes * 512
        copies[name] = {"graph_operator": op_key, "name": op["name"],
                       "element_bytes": element_bytes, "valid_elements_per_slice": elements,
                       "source_per_slice_range": [0, elements * element_bytes],
                       "joint_destination_per_slice_range": [index * stride * element_bytes, (index + 1) * stride * element_bytes],
                       "all_valid_elements_copied_once_in_order": True,
                       "index_access": access, "index_constant": scalar, "full_description": op["description"]}
    variant = HERE / build["variant"]
    original = (HERE / "frozen-v2/projection.rs").read_text()
    corrected = (variant / "projection.rs").read_text()
    boundary = "    let x: DmTensorView<'_, bf16, Chip, FusedCluster, FusedSlices, m![H]>"
    assert original[original.index(boundary):] == corrected[corrected.index(boundary):]
    for name in ("ops.rs", "rmsnorm.rs", "rope.rs"):
        assert (variant / name).read_bytes() == (HERE / "frozen-v2" / name).read_bytes()
    schedule = json.loads((evidence / "schedule.json").read_text())
    important = [op for op in schedule["instructions"]
                 if (op["tpe"] == "DmaLoad" and "projection.rs:" in op["description"])
                 or (op["tpe"] in {"Main", "Sub"} and ("compact.view()" in op["description"] or "QkvMatrix" in op["description"] and "contract_outer" in op["description"]))]
    return {"scope": "Corrected whole QKV compile gate, not NPU correctness or performance",
            "label": label, "image": build["image"], "build": digest(HERE / f"build-{label}.json"),
            "all_hbm_inputs_in_bounds_covered_once": True, "ingress": ingress,
            "ingress_destination_ownership": ingress_ownership(graph, ingress),
            "copies": copies, "output_gathers": output_gather_audit(graph),
            "post_ingress_arithmetic_and_public_ops_byte_unchanged": True,
            "compiler_makespan": build["compiler_makespan"], "selected_schedule": important,
            "lookup_prepare_count": sum(op["kind"] == "StoTab" for op in graph["operators"].values()),
            "input_trf_count": sum(op["kind"] == "StoTrf" for op in graph["operators"].values())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    value = result(args.label)
    path = HERE / f"descriptor-analysis-{args.label}.json"
    if args.check:
        assert json.loads(path.read_text()) == value
    else:
        path.write_text(json.dumps(value, indent=2) + "\n")
    print("Ingress bounds/coverage, packet copies, output gathers: PASS; static", value["compiler_makespan"])
