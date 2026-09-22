#!/usr/bin/env python3
"""Audit frozen source provenance, exact embedded images, uploads and parsed results."""
import io
import json
import re
import subprocess
import sys
import tarfile
from build import HERE, common
from freeze import freeze
from results import results
from analyze import analyze

ROOT, ARTIFACTS, digest = common.ROOT, common.ARTIFACTS, common.digest
sys.path.insert(0, str(ROOT / "moa-dev"))
from measure import parse_log
from extra_seeds import extended_harness, parse_log as parse_extra


def check_builds():
    archive = subprocess.check_output(["git", "archive", common.BASE, common.CRATE], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        original = {m.name.removeprefix(common.CRATE + "/"): tar.extractfile(m).read()
                    for m in tar if m.isfile()}
    original["Cargo.lock"] = (ROOT / "moa-dev/locks/gemma4-Cargo.lock").read_bytes()
    baseline = {n: digest(v) for n, v in original.items()
                if n.startswith("src/") or n in {"Cargo.toml", "Cargo.lock", "rust-toolchain.toml"}}
    builds = {}
    for path in sorted(HERE.glob("build-*-repro.json")):
        record = json.loads(path.read_text())
        label = record["label"]
        out = ARTIFACTS / label
        expected = dict(baseline)
        expected["src/device/shared/mlp.rs"] = digest((out / "crate/src/device/shared/mlp.rs").read_bytes())
        if record["extra_inputs"]:
            expected["src/bin/test_kernels.rs"] = digest(extended_harness(original["src/bin/test_kernels.rs"]))
        assert record["base_commit"] == common.BASE
        assert record["source_inputs_verified_in_container"] and expected == record["source_inputs"], label
        assert record["changed_sources"] == ["src/device/shared/mlp.rs"]
        for name, value in expected.items():
            assert digest((out / "crate" / name).read_bytes()) == value, (label, name)
        if "failed_exit_code" in record:
            assert record["failed_exit_code"] != 0
        else:
            assert record["exact_compile_exit_code"] == record["host_build_exit_code"] == 0
            assert {p.name: digest(p.read_bytes()) for p in (out / "package").iterdir()} == record["package_files"]
            host = (out / "package/test_runtime").read_bytes()
            for kernel, original_hash in common.IMAGES.items():
                image = (out / "kernels" / f"furiosa_opt_gemma4::ops::{kernel}.bin").read_bytes()
                assert dict(digest(image), host_occurrences=host.count(image)) == record["images"][kernel]
                assert host.count(image) == 1
                if kernel == "decoder_feedforward":
                    assert image == (out / "exact-kernel.bin").read_bytes()
                else:
                    assert digest(image)["sha256"] == original_hash
            schedule = json.loads((out / "schedule.json").read_text())
            assert record["static_cycles"] == max(i["lifetime"]["end"] for i in schedule["instructions"])
        builds[label] = record
    for path in sorted(HERE.glob("build-projection-probe-*.json")):
        record = json.loads(path.read_text())
        label = path.stem.removeprefix("build-")
        out = ARTIFACTS / label
        source = json.loads((HERE / record["candidate_build"]).read_text())
        assert record["source_inputs"]["src/device/mlp.rs"] == source["source_inputs"]["src/device/shared/mlp.rs"]
        if "failed_exit_code" not in record:
            assert record["build_exit_code"] == record["oracle_exit_code"] == 0
            host = (out / "package/probe_stream").read_bytes()
            for image_path in (out / "kernels").rglob("*.bin"):
                image = image_path.read_bytes()
                assert host.count(image) == 1
                assert dict(digest(image), host_occurrences=1) == record["images"][image_path.name]
            assert {p.name: digest(p.read_bytes()) for p in (out / "package").iterdir()} == record["package_files"]
    return builds


def upload(record):
    with tarfile.open(ARTIFACTS / f"job-{record['job']['id']}.tar") as tar:
        members = [m for m in tar if m.isfile()]
        assert len(members) == len({m.name for m in members})
        assert {m.name: digest(tar.extractfile(m).read()) for m in members} == record["package_files"]
    data = (ROOT / record["log"]).read_bytes()
    assert digest(data)["sha256"] == record["log_sha256"]
    return data.decode()


def check_jobs(builds):
    count = 0
    for path in sorted(HERE.glob("c07-screen-*-results.json")):
        report = json.loads(path.read_text())
        log = upload(report)
        label = path.stem.split("-job-")[0].removesuffix("-results")
        batch = json.loads((HERE / (label + "-build.json")).read_text())
        assert {k: v for k, v in report["package_files"].items() if k != "job.yaml"} == batch["package_files"]
        sections = re.findall(r"^C07_BEGIN (\w+)\n(.*?)^C07_END \1$", log, re.M | re.S)
        assert [n for n, _ in sections] == [s["name"] for s in batch["stages"]]
        assert log.count("C07_COMPLETE") == 1
        for stage, (name, content) in zip(batch["stages"], sections):
            actual = {}
            try:
                assert re.findall(r"^C07_EXIT " + name + r" (\d+)$", content, re.M) == ["0"]
                actual = parse_extra(content) if stage["extra"] else {"kernels": parse_log(content), "comparisons_passed": 15}
                actual["valid"] = True
            except (AssertionError, ValueError):
                actual = {"valid": False}
            recorded = report["stages"][name]
            assert actual["valid"] == recorded["valid"], (path, name)
            if actual["valid"]:
                assert actual == recorded
            build = stage["build"]
            if build == "e55":
                files = json.loads((ROOT / "docs/experiments/e69-c02-c03-control.json").read_text())["runs"][0]["package_files"]
            elif build == "e79":
                files = json.loads((HERE.parent / "c04-down-transfer/build-c04-eight-ring32-v1-repro.json").read_text())["package_files"]
            else:
                files = builds[build]["package_files"]
            for filename in ("test_runtime", "fixtures.safetensors", "remote_entrypoint.sh"):
                assert report["package_files"][name + "_" + filename] == files[filename]
        assert report["valid"] == (report["job"]["exit_code"] == 0 and report["job"]["status"] in {"SUCCEEDED", "COMPLETED"} and all(v["valid"] for v in report["stages"].values()))
        count += 1
    for path in sorted(HERE.glob("projection-probe-*-results.json")):
        record = json.loads(path.read_text())
        log = upload(record)
        details = re.findall(r"^C07_PROBE case=(\d+) projection=(\w+) checked=(\d+) mismatches=(\d+)$", log, re.M)
        assert [list(d) for d in details] == record["details"]
        expected = {(str(c), p, str(n)) for c in range(6) for p, n in (("up", 15360), ("gate", 15360), ("down", 3840))}
        complete = len(details) == 18 and {d[:3] for d in details} == expected
        assert complete == record["complete"]
        assert record["valid"] == (complete and all(d[3] == "0" for d in details) and record["job"]["exit_code"] == 0 and record["job"]["status"] in {"SUCCEEDED", "COMPLETED"})
        build = json.loads((HERE / record["build"]).read_text())
        assert {k: v for k, v in record["package_files"].items() if k != "job.yaml"} == build["package_files"]
        count += 1
    return count


if __name__ == "__main__":
    generated = json.loads((HERE / "generated.json").read_text())
    assert generated["base"] == common.BASE
    for variant, value in generated["variants"].items():
        assert digest((HERE / variant / "mlp.rs").read_bytes()) == value, variant
    freeze(check=True)
    builds = check_builds()
    jobs = check_jobs(builds)
    analyze(check=True)
    results(check=True)
    print(f"Verified {len(builds)} full build attempts and {jobs} collected Arena jobs, including failures.")
