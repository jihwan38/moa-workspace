"""Verify the frozen two-change merge, coordinates and scheduled initialization."""
import hashlib
import json
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    expected = {
        "mlp.ancestor.rs": "7c363a7f116ff9ebf4027f6a9957faa5cb3ce74aac85b3c5f7b6579ff628b267",
        "mlp.row2.rs": "bf5e8c153ec6b226136cf0a9887267fedefd562a67b74ef34d47429960dbf602",
        "mlp.tail-zero.rs": "07910fa5b11bd23c7da97c7c31c431db6629fe7db95cead2eaf5cdda482f1514",
    }
    for name, digest in expected.items():
        assert sha(HERE / name) == digest
    merge = subprocess.run(["diff3", "-m", str(HERE / "mlp.row2.rs"), str(HERE / "mlp.ancestor.rs"), str(HERE / "mlp.tail-zero.rs")], capture_output=True)
    assert merge.returncode == 0 and merge.stdout == (HERE / "mlp.rs").read_bytes()
    source = (HERE / "mlp.rs").read_text()
    ancestor = (HERE / "mlp.ancestor.rs").read_text()
    row2 = (HERE / "mlp.row2.rs").read_text()
    prefix = "pub(crate) fn project_up_and_gate("
    suffix = "pub(crate) fn feedforward("
    gather = "fn compact_full256_rows("
    assert source[:source.index(prefix)] == ancestor[:ancestor.index(prefix)]
    assert source[source.index(suffix):] == ancestor[ancestor.index(suffix):]
    assert source[source.index(gather):source.index(suffix)] == row2[row2.index(gather):row2.index(suffix)]
    assert source.count(".cast::<") == ancestor.count(".cast::<")
    assert source.count("_tail_zero.view_mut().memset(") == 4
    assert "_dense.view_mut().memset(" not in source

    rows = set()
    for cluster in range(2):
        for group in range(128):
            stream = []
            for row in range(30):
                for side in range(2):
                    logical = 7680 * cluster + 60 * group + 2 * row + side
                    assert logical not in rows
                    rows.add(logical)
                    assert (((cluster * 128 + group) * 30 + row) * 2 + side) == logical
                    stream.append(logical)
            packets = [[7680 * cluster + 60 * group + 2 * (2 * p + r) + s for r in range(2) for s in range(2)] for p in range(15)]
            assert [v for packet in packets for v in packet] == stream
            assert stream == list(range(60 * (128 * cluster + group), 60 * (128 * cluster + group + 1)))
    assert rows == set(range(15360))
    coverage = {}
    for label, cols in [("weight", 3840), ("scale", 240)]:
        dest = [None] * (32 * cols)
        for row in range(2):
            for col in range(cols):
                dest[(30 + row) * cols + col] = 0
        for row in range(30):
            for col in range(cols):
                index = row * cols + col
                assert dest[index] is None
                dest[index] = index + 1
        assert all(x is not None for x in dest)
        assert dest[30 * cols:] == [0] * (2 * cols)
        coverage[label] = {"real_range": [0, 30 * cols], "zero_range": [30 * cols, 32 * cols], "every_byte_written_once": True}

    schedule = json.loads((HERE / "candidate.json").read_text())
    nodes = schedule["instructions"]
    index = {n["index"]: n for n in nodes}
    producers = {}
    for n in nodes:
        for tensor in n["output_tensors"]:
            assert tensor not in producers
            producers[tensor] = n["index"]

    def ancestors(node_id, seen=None):
        seen = set() if seen is None else seen
        for tensor in index[node_id]["input_tensors"]:
            parent = producers.get(tensor)
            if parent is not None and parent not in seen:
                seen.add(parent)
                ancestors(parent, seen)
        return seen

    def one(kind, text):
        matches = [n for n in nodes if n["tpe"] == kind and text in n["description"]]
        assert len(matches) == 1, (kind, text, [n["index"] for n in matches])
        return matches[0]

    dependencies = []
    for projection in ("up", "gate"):
        for buffer in ("weight", "scale"):
            label = projection + "_" + buffer
            zero = one("Sub", label + "_tail_zero.view_mut().memset(")
            copy = one("Sub", "begin(" + label + "_tail_zero.view())")
            if buffer == "weight":
                write = one("Main", ".commit_view(" + label + "_dense.view_mut().tile::<m![FfnRow], 30")
            else:
                write = one("DmaLoad", projection + "_weight_scale.to_dm_view(")
            required = {n["index"] for n in (zero, copy, write)}
            candidates = [n for n in nodes if n["tpe"] == "Main" and "begin(" + ("weights" if buffer == "weight" else "scales") + ".view())" in n["description"] and required <= ancestors(n["index"])]
            assert candidates, label
            consumer = min(candidates, key=lambda n: n["lifetime"]["begin"])
            assert copy["lifetime"]["begin"] >= zero["lifetime"]["end"]
            assert consumer["lifetime"]["begin"] >= max(copy["lifetime"]["end"], write["lifetime"]["end"])
            dependencies.append({"buffer": label, **{key: {"node": n["index"], **n["lifetime"]} for key, n in [("zero_memset", zero), ("tail_copy", copy), ("real30_write", write), ("first_consumer", consumer)]}, "all_producers_in_consumer_ancestry": True})

    print(json.dumps({"status": "PASS", "diff3_clean_and_exact": True, "parent_sha256": expected,
        "candidate_sha256": sha(HERE / "mlp.rs"), "dpe_helper_and_suffix_unchanged": True,
        "row2_gather_unchanged": True, "additional_casts": 0,
        "bijective_rows": len(rows), "tail_coverage": coverage,
        "scheduled_initialization_dependencies": dependencies,
        "limitation": "No NPU correctness or performance run. Source/coordinate/schedule checks do not substitute for the official oracle."}, indent=2))


if __name__ == "__main__":
    main()
