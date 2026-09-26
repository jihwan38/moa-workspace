#!/usr/bin/env python3
"""Read-only verification of preserved compile, descriptor, host and Arena evidence."""
import json
from pathlib import Path
import sys
import tarfile

sys.dont_write_bytecode = True
from build import HERE, ROOT, digest
from analyze import analyze
from analyze_corrected import result
from results import outputs


def main():
    assert analyze("exact-v2") == json.loads((HERE / "descriptor-analysis-exact-v2.json").read_text())
    assert result("corrected-v2") == json.loads((HERE / "descriptor-analysis-corrected-v2.json").read_text())
    artifacts = ROOT / "moa-dev/experiments/artifacts/c08-joint-ingress"
    for label in ("exact-v2", "corrected-v1", "corrected-v2"):
        build = json.loads((HERE / f"build-{label}.json").read_text())
        for name, expected in build["source_inputs"].items():
            assert digest((artifacts / label / "crate" / name).read_bytes()) == expected
        for name, value in build["source_plan"]["source_overrides"].items():
            assert digest((ROOT / value["path"]).read_bytes()) == build["source_inputs"][name]
        assert digest((HERE / "evidence" / label / "compile.log").read_bytes()) == build["compile_log"]
        if label == "corrected-v1":
            assert build["exact_compile_exit_code"] != 0
    package = json.loads((HERE / "build-corrected-v2-repro.json").read_text())
    assert digest((ROOT / package["package_builder_source"]).read_bytes()) == package["package_builder"]
    for name, expected in package["package_files"].items():
        assert digest((artifacts / "corrected-v2/package" / name).read_bytes()) == expected
    host = (artifacts / "corrected-v2/package/test_runtime").read_bytes()
    for kernel, expected in package["images"].items():
        raw = (artifacts / "corrected-v2/kernels" / f"furiosa_opt_gemma4::ops::{kernel}.bin").read_bytes()
        assert digest(raw) == {key: expected[key] for key in ("bytes", "sha256")}
        assert host.count(raw) == expected["host_occurrences"] == 1
    screen = json.loads((HERE / "e124-c08-corrected-v2-build.json").read_text())
    batch = json.loads((HERE / "e124-c08-corrected-v2-results.json").read_text())
    assert batch["valid"] and batch["job"]["exit_code"] == 0
    with tarfile.open(artifacts / "job-105093.tar") as archive:
        actual = {m.name: digest(archive.extractfile(m).read()) for m in archive if m.isfile()}
    assert actual == batch["package_files"]
    assert {name: value for name, value in actual.items() if name != "job.yaml"} == screen["package_files"]
    log = (ROOT / batch["log"]).read_bytes()
    assert digest(log)["sha256"] == batch["log_sha256"]
    assert log == (HERE / "evidence/arena-105093.log").read_bytes()
    for stage in batch["stages"].values():
        assert stage["valid"] and stage["comparisons_passed"] == 15
        assert sum(kernel["output_comparisons_passed"] for kernel in stage["kernels"].values()) == 15
    for path, value in outputs().items():
        assert json.loads(path.read_text()) == value
    print("PASS: frozen/repaired descriptors, source identities, host images, actual upload, full log, 45 comparisons and ledger reports")


if __name__ == "__main__":
    main()
