#!/usr/bin/env python3
"""Verify C04 source, images, uploaded payloads, logs and generated reports."""
import argparse
import json
import re
import sys
import tarfile
from build import HERE, common
from analyze import analyze
from results import results

ROOT, ARTIFACTS, digest = common.ROOT, common.ARTIFACTS, common.digest
sys.path.insert(0, str(ROOT / "moa-dev"))
from measure import parse_log
from extra_seeds import extended_harness, parse_log as parse_extra


def check_builds():
    baseline = json.loads((ROOT / "moa-dev/experiments/candidates/c05-streamed-decode/compiled-source-manifest.json").read_text())
    original_host = ROOT / "moa-dev/experiments/artifacts/c05-streamed-decode/c05-partials-transpose2/crate/src/bin/test_kernels.rs"
    prefix = (ROOT / "moa-dev/experiments/candidates/c05-streamed-decode/partials/mlp.rs").read_bytes().split(b"pub(crate) type DownEpilogueSlices")[0]
    builds = {}
    for path in sorted(HERE.glob("build-*-repro.json")):
        build = json.loads(path.read_text())
        label, variant = build["label"], build["variant"]
        out = ARTIFACTS / label
        source = (HERE / variant / "mlp.rs").read_bytes()
        assert source.split(b"pub(crate) type DownEpilogueSlices")[0] == prefix
        expected = dict(baseline)
        expected["src/device/shared/mlp.rs"] = digest(source)
        if build["extra_inputs"]:
            expected["src/bin/test_kernels.rs"] = digest(extended_harness(original_host.read_bytes()))
        assert build["base_commit"] == common.BASE
        assert build["source_inputs_verified_in_container"] and expected == build["source_inputs"]
        for name, value in expected.items():
            assert digest((out / "crate" / name).read_bytes()) == value, (label, name)
        for name, value in build["logs"].items():
            assert digest((out / name).read_bytes()) == value, (label, name)
        assert build["exact_compile_exit_code"] == build["host_build_exit_code"] == 0
        host = (out / "package/test_runtime").read_bytes()
        assert {p.name: digest(p.read_bytes()) for p in (out / "package").iterdir()} == build["package_files"]
        for kernel, original_hash in common.IMAGES.items():
            image = (out / "kernels" / f"furiosa_opt_gemma4::ops::{kernel}.bin").read_bytes()
            assert dict(digest(image), host_occurrences=host.count(image)) == build["images"][kernel]
            assert host.count(image) == 1
            if kernel == "decoder_feedforward":
                assert image == (out / "exact-kernel.bin").read_bytes()
            else:
                assert digest(image)["sha256"] == original_hash
        builds[label] = build
    return builds


def check_upload(record):
    with tarfile.open(ARTIFACTS / f"job-{record['job']['id']}.tar") as tar:
        members = [m for m in tar if m.isfile()]
        assert len(members) == len({m.name for m in members})
        assert {m.name: digest(tar.extractfile(m).read()) for m in members} == record["package_files"]
    log = (ROOT / record["log"]).read_bytes()
    assert digest(log)["sha256"] == record["log_sha256"]
    return log.decode()


def check_batches(builds):
    count = 0
    for path in HERE.glob("c04-screen-*-results.json"):
        report = json.loads(path.read_text())
        assert report["valid"]
        log = check_upload(report)
        label = path.stem.split("-job-")[0].removesuffix("-results")
        batch = json.loads((HERE / (label + "-build.json")).read_text())
        assert {k: v for k, v in report["package_files"].items() if k != "job.yaml"} == batch["package_files"]
        sections = re.findall(r"^C04_BEGIN (\w+)\n(.*?)^C04_END \1$", log, re.M | re.S)
        assert [n for n, _ in sections] == [s["name"] for s in batch["stages"]]
        assert log.count("C04_COMPLETE") == 1
        for stage, (name, content) in zip(batch["stages"], sections):
            assert re.findall(r"^C04_EXIT " + name + r" (\d+)$", content, re.M) == ["0"]
            assert parse_log(content) == report["stages"][name]["kernels"]
            build = stage["build"]
            if build == "e55":
                files = json.loads((ROOT / "docs/experiments/e69-c02-c03-control.json").read_text())["runs"][0]["package_files"]
            elif build == "e74":
                files = json.loads((HERE.parent / "c05-streamed-decode/build-c05-partials-transpose2-repro.json").read_text())["package_files"]
            else:
                files = builds[build]["package_files"]
            for filename in ("test_runtime", "fixtures.safetensors", "remote_entrypoint.sh"):
                assert report["package_files"][name + "_" + filename] == files[filename]
        count += 1
    return count


def check_probe_build(builds):
    record = json.loads((HERE / "build-projection-probe-v1.json").read_text())
    out = ARTIFACTS / "projection-probe-v1"
    assert record["build_exit_code"] == record["oracle_exit_code"] == 0
    assert record["source_inputs_verified_in_container"]
    for path, value in record["source_inputs"].items():
        assert digest((out / "crate" / path).read_bytes()) == value
    assert record["source_inputs"]["src/device/mlp.rs"] == builds["c04-eight-ring32-v1"]["source_inputs"]["src/device/shared/mlp.rs"]
    for path, value in record["logs"].items():
        assert digest((out / path).read_bytes()) == value
    for source, target in {"src/lib.rs": "probe_lib.rs", "src/stream_probe.rs": "stream_probe.rs", "src/bin/probe_stream.rs": "probe_stream.rs"}.items():
        assert digest((HERE / target).read_bytes()) == record["source_inputs"][source]
    host = (out / "package/probe_stream").read_bytes()
    for image in (out / "kernels").rglob("*.bin"):
        data = image.read_bytes()
        assert host.count(data) == 1 and dict(digest(data), host_occurrences=1) == record["images"][image.name]
    assert {p.name: digest(p.read_bytes()) for p in (out / "package").iterdir()} == record["package_files"]


def check_diagnostics(builds):
    for kind in ("extra", "probe"):
        record = json.loads((ROOT / f"docs/experiments/e79-c04-{kind}-correctness.json").read_text())
        assert record["valid"]
        log = check_upload(record)
        build = json.loads((ROOT / record["build"]).read_text())
        assert {k: v for k, v in record["package_files"].items() if k != "job.yaml"} == build["package_files"]
        if kind == "extra":
            assert parse_extra(log) == record["results"]
            assert build["images"] == builds["c04-eight-ring32-v1"]["images"]
        else:
            expected = {(str(c), name, str(size), "0") for c in range(4) for name, size in (("up", 15360), ("gate", 15360), ("down", 3840))}
            actual = re.findall(r"^C04_PROBE case=(\d+) projection=(\w+) checked=(\d+) mismatches=(\d+)$", log, re.M)
            assert len(actual) == 12 and set(actual) == expected
            assert log.count("C04_PROBE_COMPLETE oracle_only=false cases=4 checked=138240 mismatches=0") == 1


def graph_proof():
    out = ARTIFACTS / "c04-eight-ring32-v1"
    graph = json.loads((out / "broadcast-graph.json").read_text())
    matches = [(index, op) for index, op in graph["operators"].items()
               if "CustomWithBitmap" in op["description"] and "Dummy256_32=8, Dummy=4, L_1920=8" in op["description"]]
    assert len(matches) == 1
    index, op = matches[0]
    bitmap = [int(v) for v in re.findall(r"SliceBitmap \[(\d+)\]", op["description"])]
    assert bitmap == [32 * (s // 32) + s % 8 for s in range(256)]
    assert (out / "graph-kernel.bin").read_bytes() == (out / "exact-kernel.bin").read_bytes()
    return dict(graph_operator=index, slices_verified=len(bitmap),
        source_slice="32 * (destination_slice // 32) + destination_slice % 8",
        global_activation_index="1920 * (destination_slice % 8) + local_element",
        files={name: digest((out / name).read_bytes()) for name in ("broadcast-graph.json", "broadcast-schedule.json", "graph-kernel.bin", "graph-compile.log")},
        image_matches_measured=True, scope="Compiler routing and image proof; NPU correctness recorded separately.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--builds-only", action="store_true")
    args = parser.parse_args()
    builds = check_builds()
    check_probe_build(builds)
    proof = graph_proof()
    if not args.builds_only:
        assert json.loads((HERE / "static-summary.json").read_text()) == analyze()
        batches = check_batches(builds)
        check_diagnostics(builds)
        results(check=True)
        print(f"Verified {len(builds)} builds, {batches} batch jobs, extra inputs and full-shape diagnostics.")
    else:
        print(f"Verified {len(builds)} builds and projection diagnostic build.")
    (HERE / "broadcast-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
