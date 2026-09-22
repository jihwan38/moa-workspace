#!/usr/bin/env python3
"""Verify frozen sources, successful/failed compiler records and exact packages."""
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
from extra_seeds import extended_harness
from extra_seeds import parse_log as parse_extra


def check_builds():
    base = json.loads((ROOT / "baseline/e55.json").read_text())
    baseline = json.loads((ROOT / base["compile_inputs"]).read_text())
    builds = {}
    for path in sorted(HERE.glob("build-*-repro.json")):
        build = json.loads(path.read_text())
        label = build["label"]
        out = ARTIFACTS / label
        expected = dict(baseline)
        source = HERE / ("attempts/partials-transpose4.rs" if label == "c05-partials-first" else build["variant"] + "/mlp.rs")
        expected["src/device/shared/mlp.rs"] = digest(source.read_bytes())
        if build["extra_inputs"]:
            original = (ARTIFACTS / "c05-direct-first/crate/src/bin/test_kernels.rs").read_bytes()
            expected["src/bin/test_kernels.rs"] = digest(extended_harness(original))
        assert expected == build["source_inputs"] and build["source_inputs_verified_in_container"], label
        for name, value in expected.items():
            assert digest((out / "crate" / name).read_bytes()) == value, (label, name)
        for name, value in build["logs"].items():
            assert digest((out / name).read_bytes()) == value, (label, name)
        if "failed_exit_code" in build:
            assert build["failed_exit_code"] != 0 and "host_build_exit_code" not in build
            assert not (out / "package").exists()
            continue
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
    path = ARTIFACTS / f"job-{record['job']['id']}.tar"
    with tarfile.open(path) as tar:
        members = [m for m in tar if m.isfile()]
        assert len(members) == len({m.name for m in members})
        assert {m.name: digest(tar.extractfile(m).read()) for m in members} == record["package_files"]
    log = (ROOT / record["log"]).read_bytes()
    assert digest(log)["sha256"] == record["log_sha256"]
    return log.decode()


def check_diagnostics(builds):
    cases = {"build-probe.json": "projection-probe", "build-projection-probe-v2.json": "projection-probe-v2",
             "build-projection-probe-v3.json": "projection-probe-v3"}
    for name, label in cases.items():
        build = json.loads((HERE / name).read_text())
        out = ARTIFACTS / label
        assert build["source_inputs_verified_in_container"]
        for source, value in build["source_inputs"].items():
            assert digest((out / "crate" / source).read_bytes()) == value
        assert build["source_inputs"]["src/device/mlp.rs"] == builds["c05-partials-transpose2"]["source_inputs"]["src/device/shared/mlp.rs"]
        for log, value in build["logs"].items():
            assert digest((out / log).read_bytes()) == value
        if label != "projection-probe-v3":
            assert build["failed_exit_code"] != 0 and "build_exit_code" not in build
            continue
        assert build["build_exit_code"] == build["oracle_exit_code"] == 0
        for source, target in {"src/lib.rs": "probe_lib.rs", "src/stream_probe.rs": "stream_probe.rs", "src/bin/probe_stream.rs": "probe_stream.rs"}.items():
            assert digest((HERE / target).read_bytes()) == build["source_inputs"][source]
        host = (out / "package/probe_stream").read_bytes()
        for image in (out / "kernels").rglob("*.bin"):
            data = image.read_bytes()
            assert dict(digest(data), host_occurrences=host.count(data)) == build["images"][image.name]
            assert host.count(data) == 1
        assert {p.name: digest(p.read_bytes()) for p in (out / "package").iterdir()} == build["package_files"]
    for kind in ("extra", "probe"):
        record = json.loads((ROOT / f"docs/experiments/e74-c05-{kind}-correctness.json").read_text())
        assert record["valid"]
        log = check_upload(record)
        build = json.loads((ROOT / record["build"]).read_text())
        assert {k: v for k, v in record["package_files"].items() if k != "job.yaml"} == build["package_files"]
        if kind == "extra":
            assert parse_extra(log) == record["results"]
            assert build["images"] == builds["c05-partials-transpose2"]["images"]
        else:
            expected = {(str(c), n, str(size), "0") for c in range(4) for n, size in (("up", 15360), ("gate", 15360), ("down", 3840))}
            actual = re.findall(r"^C05_PROBE case=(\d+) projection=(\w+) checked=(\d+) mismatches=(\d+)$", log, re.M)
            assert len(actual) == 12 and set(actual) == expected
            assert log.count("C05_PROBE_COMPLETE oracle_only=false cases=4 checked=138240 mismatches=0") == 1
    for path in HERE.glob("c05-screen-first*-results.json"):
        record = json.loads(path.read_text())
        assert record["valid"]
        log = check_upload(record)
        batch = json.loads((HERE / "c05-screen-first-build.json").read_text())
        assert {k: v for k, v in record["package_files"].items() if k != "job.yaml"} == batch["package_files"]
        sections = dict(re.findall(r"^C05_BEGIN (\w+)\n(.*?)^C05_END \1$", log, re.M | re.S))
        assert set(sections) == {s["name"] for s in batch["stages"]}
        for stage in batch["stages"]:
            key = stage["name"]
            assert parse_log(sections[key]) == record["stages"][key]["kernels"]
            expected = (builds[stage["build"]]["package_files"] if stage["build"] else
                        json.loads((ROOT / "docs/experiments/e69-c02-c03-control.json").read_text())["runs"][0]["package_files"])
            for filename in ("test_runtime", "fixtures.safetensors", "remote_entrypoint.sh"):
                assert record["package_files"][key + "_" + filename] == expected[filename]


if __name__ == "__main__":
    builds = check_builds()
    assert json.loads((HERE / "static-summary.json").read_text()) == analyze()
    check_diagnostics(builds)
    results(check=True)
    for report in (ROOT / "docs/experiments").glob("e7*-c05-*.json"):
        value = json.loads(report.read_text())
        if "pending_job" in value:
            continue
        for run in value.get("runs", []):
            log = (ROOT / run["log"]).read_bytes()
            assert digest(log)["sha256"] == run["log_sha256"]
            if run.get("valid"):
                if "batch_stage" in run:
                    sections = dict(re.findall(r"^C05_BEGIN (\w+)\n(.*?)^C05_END \1$", log.decode(), re.M | re.S))
                    content = sections[run["batch_stage"]]
                else:
                    content = log.decode()
                assert parse_log(content) == run["kernels"]
    print(f"Verified {len(builds)} successful builds, failed-build archives and available measurements.")
