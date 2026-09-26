#!/usr/bin/env python3
"""Bounded screening with unchanged benchmark hosts and adjacent E104 controls.

Each stage is a separate process in one allocation, not an independent Arena job.
The uploaded archive is checked against the locally verified payload on collection.
"""
import argparse
from datetime import datetime, timezone
import json
import re
import subprocess
import sys
sys.dont_write_bytecode = True
import tarfile
from package import HERE, common

ROOT, ARTIFACTS, digest = common.ROOT, common.ARTIFACTS, common.digest
sys.path.insert(0, str(ROOT / "moa-dev"))
from measure import cli, parse_log
from extra_seeds import parse_log as parse_extra


def record(path, data):
    with path.open("x") as out:
        json.dump(data, out, indent=2)
        out.write("\n")


def prepare(label, candidates):
    out = ARTIFACTS / label
    out.mkdir(parents=True, exist_ok=False)
    e104 = ROOT / "moa-dev/experiments/artifacts/c07-native-scale-dpe/c07-e104-reviewed-v1/package"
    stages = []
    for stage, path, extra, build in [
        ("control_before", e104, False, "e104"),
        *[(f"candidate_{i}", ARTIFACTS / c / "package", json.loads((HERE / f"build-{c}-repro.json").read_text())["extra_inputs"], c) for i, c in enumerate(candidates)],
        ("control_after", e104, False, "e104"),
    ]:
        files = {name: (path / name).read_bytes()
                 for name in ("test_runtime", "fixtures.safetensors", "remote_entrypoint.sh")}
        if build == "e104":
            expected = json.loads((HERE.parent / "c07-native-scale-dpe/build-c07-e104-reviewed-v1-repro.json").read_text())
            assert {k: digest(v) for k, v in files.items()} == expected["package_files"]
        else:
            expected = json.loads((HERE / f"build-{build}-repro.json").read_text())
            assert expected["exact_compile_exit_code"] == expected["host_build_exit_code"] == 0
            assert {k: digest(v) for k, v in files.items()} == expected["package_files"]
        for name, data in files.items():
            (out / (stage + "_" + name)).write_bytes(data)
        stages.append({"name": stage, "build": build, "extra": extra})
    runner = ['#!/bin/sh', 'set -eu', 'cd "$(dirname "$0")"', 'any_failed=0']
    for stage in stages:
        name = stage["name"]
        runner += [f"mkdir {name}", *[f"cp {name}_{f} {name}/{f}" for f in ("test_runtime", "fixtures.safetensors", "remote_entrypoint.sh")],
                   f"chmod +x {name}/test_runtime", f"printf 'C08_BEGIN {name}\\n'",
                   f"if sh {name}/remote_entrypoint.sh; then code=0; else code=$?; any_failed=1; fi",
                   f"printf 'C08_EXIT {name} %s\\n' \"$code\"", f"printf 'C08_END {name}\\n'"]
    runner += ["printf 'C08_COMPLETE\\n'", 'exit "$any_failed"']
    (out / "screen.sh").write_text("\n".join(runner) + "\n")
    (out / "screen.sh").chmod(0o755)
    data = {"label": label, "stages": stages, "timeout_seconds": 70,
            "method": "Serial unchanged benchmark processes, with E104 before and after; one Arena allocation, not independent repetitions.",
            "package_files": {p.name: digest(p.read_bytes()) for p in sorted(out.iterdir())}}
    record(HERE / f"{label}-build.json", data)
    print(f"Prepared {label}: {len(stages)} stages")


def submit(label):
    out = ARTIFACTS / label
    data = json.loads((HERE / f"{label}-build.json").read_text())
    assert {p.name: digest(p.read_bytes()) for p in out.iterdir()} == data["package_files"]
    remote = "/tmp/moa-c08/" + label
    subprocess.run(["docker", "exec", "moa2026-dev", "mkdir", "-p", remote], check=True)
    subprocess.run(["docker", "cp", str(out) + "/.", "moa2026-dev:" + remote], check=True)
    for name, value in data["package_files"].items():
        actual = subprocess.check_output(["docker", "exec", "moa2026-dev", "sha256sum", remote + "/" + name], text=True).split()[0]
        assert actual == value["sha256"]
    path = HERE / f"{label}-submission.json"
    record(path, {"attempted_at_utc": datetime.now(timezone.utc).isoformat()})
    response = cli("submit", *(remote + "/" + n for n in data["package_files"]), "--entrypoint", "screen.sh", "--timeout", 70, "--name", "beef-" + label)
    saved = json.loads(path.read_text())
    saved["response"] = response
    matches = re.findall(r"submitted job (\d+)", response)
    if len(matches) == 1:
        saved["job_id"] = int(matches[0])
    path.write_text(json.dumps(saved, indent=2) + "\n")
    assert len(matches) == 1, response
    print(response, end="")


def collect(label, job=None):
    data = json.loads((HERE / f"{label}-build.json").read_text())
    original_job = json.loads((HERE / f"{label}-submission.json").read_text())["job_id"]
    job = original_job if job is None else job
    status = json.loads(cli("status", job))
    assert status["status"] in {"SUCCEEDED", "FAILED", "CANCELLED", "COMPLETED"}, status
    log = cli("logs", job)
    log_path = ROOT / "moa-dev/logs" / f"{label}-{job}.log"
    log_path.write_text(log)
    remote = f"/tmp/moa-c08/{label}/job-{job}.tar"
    cli("download", job, "--out", remote)
    local = ARTIFACTS / f"job-{job}.tar"
    subprocess.run(["docker", "cp", "moa2026-dev:" + remote, str(local)], check=True)
    with tarfile.open(local) as archive:
        members = [m for m in archive if m.isfile()]
        assert len({m.name for m in members}) == len(members)
        files = {m.name: digest(archive.extractfile(m).read()) for m in members}
    assert {k: v for k, v in files.items() if k != "job.yaml"} == data["package_files"]
    result = {"job": status, "log": str(log_path.relative_to(ROOT)), "log_sha256": digest(log.encode())["sha256"],
              "package_files": files, "method": data["method"], "stages": {}, "valid": False}
    try:
        sections = re.findall(r"^C08_BEGIN (\w+)\n(.*?)^C08_END \1$", log, re.M | re.S)
        assert [n for n, _ in sections] == [s["name"] for s in data["stages"]]
        assert log.count("C08_COMPLETE") == 1
        for stage, (name, content) in zip(data["stages"], sections):
            try:
                assert re.findall(r"^C08_EXIT " + name + r" (\d+)$", content, re.M) == ["0"]
                value = parse_extra(content) if stage["extra"] else {"kernels": parse_log(content), "comparisons_passed": 15}
                result["stages"][name] = dict(value, valid=True)
            except (AssertionError, ValueError) as error:
                result["stages"][name] = {"valid": False, "error": str(error) or "Stage exited nonzero"}
        result["valid"] = (status["exit_code"] == 0 and status["status"] in {"SUCCEEDED", "COMPLETED"}
                           and all(s["valid"] for s in result["stages"].values()))
    except (AssertionError, ValueError) as error:
        result["error"] = str(error) or "Incomplete screening job"
    suffix = "" if job == original_job else f"-job-{job}"
    record(HERE / f"{label}{suffix}-results.json", result)
    print(json.dumps(result["stages"], indent=2))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "submit", "collect"])
    parser.add_argument("label")
    parser.add_argument("candidates", nargs="*")
    parser.add_argument("--job", type=int, help="Collect an immutable rerun of this screening package")
    args = parser.parse_args()
    assert re.fullmatch(r"[a-z0-9-]{1,70}", args.label)
    if args.action == "prepare":
        assert 1 <= len(args.candidates) <= 5
        prepare(args.label, args.candidates)
    else:
        if args.action == "collect":
            collect(args.label, args.job)
        else:
            assert args.job is None
            submit(args.label)
