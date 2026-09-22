#!/usr/bin/env python3
"""One bounded Arena screening job: probe, unchanged benchmark processes, cache edges.

Every performance process uses the original three-input harness, fixture and runner.
The surrounding batch is new. Its stages are not independent Arena repetitions.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile

from build import ARTIFACTS, HERE, ROOT, digest
sys.path.insert(0, str(ROOT / "moa-dev"))
from measure import cli, parse_log
from extra_seeds import parse_log as parse_extra

PACKAGE = ARTIFACTS / "batch-package"
REMOTE = "/tmp/moa-c02-c03/batch-package"
LABELS = {"upgate": "c02-upgate-sequencer", "down": "c02-down-sequencer", "combined": "c02-combined-sequencer", "edges": "c03-edges"}
ORDER = ["probe", "baseline_before", "upgate", "down", "combined", "edges", "baseline_after"]


def prepare():
    PACKAGE.mkdir(exist_ok=True)
    control = json.loads((ROOT / "docs/experiments/e69-c02-c03-control.json").read_text())["runs"][0]
    archive = ROOT / f"furiosa-opt-gemma4-12B/target/arena-packages/job-{control['job']['id']}.tar"
    with tarfile.open(archive) as source:
        for name in ("test_runtime", "fixtures.safetensors", "remote_entrypoint.sh"):
            data = source.extractfile(name).read()
            assert digest(data) == control["package_files"][name]
            (PACKAGE / ("baseline_runtime" if name == "test_runtime" else name)).write_bytes(data)
    builds = {}
    for stage, label in LABELS.items():
        build = json.loads((HERE / f"build-{label}-repro.json").read_text())
        assert build["exact_compile_exit_code"] == build["host_build_exit_code"] == 0
        source = ARTIFACTS / label / "package"
        assert {p.name: digest(p.read_bytes()) for p in source.iterdir()} == build["package_files"]
        (PACKAGE / f"{stage}_runtime").write_bytes((source / "test_runtime").read_bytes())
        assert (source / "remote_entrypoint.sh").read_bytes() == (PACKAGE / "remote_entrypoint.sh").read_bytes()
        if stage == "edges":
            (PACKAGE / "edges.safetensors").write_bytes((source / "fixtures.safetensors").read_bytes())
        else:
            assert (source / "fixtures.safetensors").read_bytes() == (PACKAGE / "fixtures.safetensors").read_bytes()
        builds[stage] = label
    probe = (ARTIFACTS / "packet-probe-sequencer/probe_packet").read_bytes()
    assert digest(probe) == json.loads((HERE / "build-packet-probe.json").read_text())["host"]
    (PACKAGE / "probe_packet").write_bytes(probe)
    runner = '''#!/bin/sh
set -eu
cd "$(dirname "$0")"
any_failed=0
printf 'C02_C03_BEGIN probe\\n'
cp ./probe_packet ./probe.exec
chmod +x ./probe.exec
if ./probe.exec; then code=0; else code=$?; any_failed=1; fi
printf 'C02_C03_EXIT probe %s\\n' "$code"
printf 'C02_C03_END probe\\n'
run_case() {
    stage=$1
    executable=$2
    fixture=$3
    mkdir "$stage"
    cp "$executable" "$stage/test_runtime"
    cp "$fixture" "$stage/fixtures.safetensors"
    cp remote_entrypoint.sh "$stage/remote_entrypoint.sh"
    printf 'C02_C03_BEGIN %s\\n' "$stage"
    if sh "$stage/remote_entrypoint.sh"; then code=0; else code=$?; any_failed=1; fi
    printf 'C02_C03_EXIT %s %s\\n' "$stage" "$code"
    printf 'C02_C03_END %s\\n' "$stage"
}
run_case baseline_before baseline_runtime fixtures.safetensors
run_case upgate upgate_runtime fixtures.safetensors
run_case down down_runtime fixtures.safetensors
run_case combined combined_runtime fixtures.safetensors
run_case edges edges_runtime edges.safetensors
run_case baseline_after baseline_runtime fixtures.safetensors
printf 'C02_C03_BATCH_COMPLETE\\n'
exit "$any_failed"
'''
    (PACKAGE / "batch.sh").write_text(runner)
    for executable in PACKAGE.iterdir():
        if executable.suffix == ".sh" or executable.name.endswith("_runtime") or executable.name == "probe_packet":
            executable.chmod(0o755)
    record = {"stages": ORDER, "candidate_builds": builds, "control_job": control["job"]["id"],
              "method": "One Arena allocation; seven serial diagnostic/benchmark processes. Each basic benchmark retains the frozen official three-input host/fixture/runner. No independent repetitions are claimed.",
              "package_files": {p.name: digest(p.read_bytes()) for p in sorted(PACKAGE.iterdir())}, "timeout_seconds": 70}
    (HERE / "batch-build.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Prepared {sum(v['bytes'] for v in record['package_files'].values()):,} bytes, seven stages, timeout 70 seconds")


def submit():
    record = json.loads((HERE / "batch-build.json").read_text())
    assert {p.name: digest(p.read_bytes()) for p in PACKAGE.iterdir()} == record["package_files"]
    path = HERE / "batch-submission.json"
    with path.open("x") as f:
        json.dump({"attempted_at_utc": datetime.now(timezone.utc).isoformat()}, f)
    subprocess.run(["docker", "exec", "moa2026-dev", "mkdir", "-p", REMOTE], check=True)
    subprocess.run(["docker", "cp", str(PACKAGE) + "/.", "moa2026-dev:" + REMOTE], check=True)
    for name, value in record["package_files"].items():
        actual = subprocess.check_output(["docker", "exec", "moa2026-dev", "sha256sum", REMOTE + "/" + name], text=True).split()[0]
        assert actual == value["sha256"]
    response = cli("submit", *(REMOTE + "/" + name for name in record["package_files"]), "--entrypoint", "batch.sh", "--timeout", 70, "--name", "beef-c02-c03-screen-and-edges")
    data = json.loads(path.read_text()); data["response"] = response
    matches = re.findall(r"submitted job (\d+)", response)
    if len(matches) == 1:
        data["job_id"] = int(matches[0])
    path.write_text(json.dumps(data, indent=2) + "\n")
    assert len(matches) == 1, response
    print(response, end="")


def collect():
    build = json.loads((HERE / "batch-build.json").read_text())
    job = json.loads((HERE / "batch-submission.json").read_text())["job_id"]
    status = json.loads(cli("status", job))
    assert status["status"] in {"SUCCEEDED", "FAILED", "CANCELLED", "COMPLETED"}, status
    log = cli("logs", job)
    log_path = ROOT / "moa-dev/logs" / f"c02-c03-batch-job-{job}.log"
    log_path.write_text(log)
    cli("download", job, "--out", REMOTE + f"/job-{job}.tar")
    local = ARTIFACTS / f"job-{job}.tar"
    subprocess.run(["docker", "cp", "moa2026-dev:" + REMOTE + f"/job-{job}.tar", str(local)], check=True)
    with tarfile.open(local) as archive:
        members = [m for m in archive if m.isfile()]
        assert len({m.name for m in members}) == len(members)
        files = {m.name: digest(archive.extractfile(m).read()) for m in members}
    assert "job.yaml" in files
    assert {k: v for k, v in files.items() if k != "job.yaml"} == build["package_files"]
    sections = re.findall(r"^C02_C03_BEGIN (\w+)\n(.*?)^C02_C03_END \1$", log, re.M | re.S)
    result = {"job": status, "log": str(log_path.relative_to(ROOT)), "log_sha256": digest(log.encode())["sha256"],
              "package_files": files, "method": build["method"], "stages": {}, "valid": False}
    try:
        assert [name for name, _ in sections] == ORDER and log.count("C02_C03_BATCH_COMPLETE") == 1
        for name, text in sections:
            exits = re.findall(r"^C02_C03_EXIT " + name + r" (\d+)$", text, re.M)
            if exits != ["0"]:
                result["stages"][name] = {"valid": False, "exit_codes": exits, "error": "Stage failed; inspect the preserved full log"}
                continue
            if name == "probe":
                matches = re.findall(r"C02_RESULT case=(\d+) scale_checked=(\d+) scale_errors=(\d+) output_checked=(\d+) output_errors=(\d+) actual=(\[[^\n]*\])", text)
                assert len(matches) == 4 and {m[0] for m in matches} == {"0", "1", "2", "3"}
                assert all(m[1:5] == ("960", "0", "4", "0") for m in matches)
                assert text.count("C02_COMPLETE oracle_only=false errors=0") == 1
                result["stages"][name] = {"cases": 4, "scales_checked": 3840, "outputs_checked": 16, "mismatches": 0, "details": matches}
            elif name == "edges":
                from edges import POSITIONS
                extra = parse_extra(text)
                matches = re.findall(r"C03_CACHE run=(\d+) cache=([kv]) pos=(\d+) slot=(\d+) checked=(\d+) failures=0", text)
                expected = {(str(run), key, str(pos), str(pos % 1024), "2095104") for run, pos in zip(range(3, 9), POSITIONS) for key in ("k", "v")}
                assert len(matches) == 12 and set(matches) == expected
                extra.update(cache_checks=matches, untouched_cache_elements_checked=12 * 2095104)
                result["stages"][name] = extra
            else:
                result["stages"][name] = {"kernels": parse_log(text), "comparisons_passed": 15}
        result["valid"] = (status["exit_code"] == 0 and status["status"] in {"SUCCEEDED", "COMPLETED"}
                           and all(s.get("valid", True) for s in result["stages"].values()))
    except (AssertionError, ValueError) as error:
        result["error"] = str(error) or "Required status, stage or diagnostic check did not pass"
    (HERE / "batch-results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"job": job, "valid": result["valid"], "stages": {k: v.get("kernels", v.get("correctness", v.get("mismatches"))) for k, v in result["stages"].items()}, "error": result.get("error")}, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "submit", "collect"])
    args = parser.parse_args()
    {"prepare": prepare, "submit": submit, "collect": collect}[args.action]()
