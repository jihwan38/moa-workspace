#!/usr/bin/env python3
"""Collect detailed profiles between immutable E104 controls in one allocation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
from image_profile import digest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARTIFACTS = ROOT / "moa-dev/experiments/artifacts/e104-weight-path"
sys.path.insert(0, str(ROOT / "moa-dev"))
from measure import cli, parse_log


def write(path, data):
    with path.open("x") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def prepare(label, build):
    out = ARTIFACTS / label
    out.mkdir(parents=True, exist_ok=False)
    control_dir = ROOT / "moa-dev/experiments/artifacts/c07-native-scale-dpe/c07-e104-reviewed-v1/package"
    control_build = json.loads((HERE.parent / "c07-native-scale-dpe/build-c07-e104-reviewed-v1-repro.json").read_text())
    comparison = json.loads((ROOT / "baseline/e104.json").read_text())
    assert control_build["source_inputs"] == json.loads((ROOT / comparison["compile_inputs"]).read_text())
    for name in ["test_runtime", "fixtures.safetensors"]:
        raw = (control_dir / name).read_bytes()
        assert digest(raw) == control_build["package_files"][name]
        (out / name).write_bytes(raw)
    diagnostic_build = json.loads((HERE / f"build-{build}.json").read_text())
    probe = (ARTIFACTS / build / "package/probe").read_bytes()
    assert digest(probe) == diagnostic_build["labelled_host"]
    (out / "probe").write_bytes(probe)
    script = ['#!/bin/sh', 'set -u', 'cd "$(dirname "$0")"', 'chmod +x test_runtime probe', 'any_failed=0']
    stages = [("control_before", "info", "test_runtime"), ("probe_info", "info", "probe"),
              ("probe_debug", "debug", "probe"), ("probe_trace", "trace", "probe"), ("control_after", "info", "test_runtime")]
    for name, level, binary in stages:
        script += [f"printf 'DIAG_STAGE_BEGIN {name}\\n'",
                   f"if FURIOSA_OPT_PROFILE={level} DIAG_KERNEL=decoder_feedforward ./{binary}; then code=0; else code=$?; any_failed=1; fi",
                   f"printf 'DIAG_STAGE_EXIT {name} %s\\n' \"$code\"", f"printf 'DIAG_STAGE_END {name}\\n'"]
    script += ["printf 'DIAG_SCREEN_COMPLETE\\n'", 'exit "$any_failed"']
    (out / "run.sh").write_text("\n".join(script)+"\n")
    write(HERE / f"{label}-package.json", {"label": label, "diagnostic_build": build, "stages": stages,
        "control_recipe": "baseline/e104.json", "control_source_manifest_matches": True,
        "scope": "One allocation; three distinct inputs per stage, not independent repetitions; compare profiling overhead",
        "package_files": {p.name: digest(p.read_bytes()) for p in sorted(out.iterdir())}})
    print("Prepared", label)


def submit(label):
    package = json.loads((HERE / f"{label}-package.json").read_text())
    out = ARTIFACTS / label
    assert {p.name: digest(p.read_bytes()) for p in out.iterdir()} == package["package_files"]
    remote = "/tmp/moa-e104-weight/" + label
    subprocess.run(["docker", "exec", "moa2026-dev", "mkdir", "-p", remote], check=True)
    subprocess.run(["docker", "cp", str(out)+"/.", "moa2026-dev:"+remote], check=True)
    for name, value in package["package_files"].items():
        actual = subprocess.check_output(["docker", "exec", "moa2026-dev", "sha256sum", remote+"/"+name], text=True).split()[0]
        assert actual == value["sha256"]
    path = HERE / f"{label}-submission.json"
    write(path, {"attempted_at_utc": datetime.now(timezone.utc).isoformat()})
    response = cli("submit", *(remote+"/"+name for name in package["package_files"]),
                   "--entrypoint", "run.sh", "--timeout", 70, "--name", "beef-"+label)
    data = json.loads(path.read_text())
    data["response"] = response
    match = re.findall(r"submitted job (\d+)", response)
    if len(match) == 1:
        data["job_id"] = int(match[0])
    path.write_text(json.dumps(data, indent=2)+"\n")
    assert len(match) == 1, response
    print(response, end="")


def collect(label, job=None):
    package = json.loads((HERE / f"{label}-package.json").read_text())
    original = json.loads((HERE / f"{label}-submission.json").read_text())["job_id"]
    job = original if job is None else job
    status = json.loads(cli("status", job))
    assert status["status"] in {"SUCCEEDED", "COMPLETED", "FAILED", "CANCELLED"}, status
    log = cli("logs", job)
    evidence = HERE / "evidence"
    log_path = evidence / f"arena-{job}.log"
    log_path.write_text(log)
    remote = f"/tmp/moa-e104-weight/{label}/job-{job}.tar"
    cli("download", job, "--out", remote)
    local = ARTIFACTS / f"job-{job}.tar"
    subprocess.run(["docker", "cp", "moa2026-dev:"+remote, str(local)], check=True)
    with tarfile.open(local) as tar:
        members = [m for m in tar if m.isfile()]
        assert len({m.name for m in members}) == len(members)
        files = {m.name: digest(tar.extractfile(m).read()) for m in members}
    assert {k:v for k,v in files.items() if k != "job.yaml"} == package["package_files"]
    result = {"job": status, "actual_upload_matches": True, "log": str(log_path.relative_to(ROOT)),
              "log_digest": digest(log.encode()), "stages": {}, "screen_complete": "DIAG_SCREEN_COMPLETE" in log}
    for name, content in re.findall(r"^DIAG_STAGE_BEGIN (\w+)\n(.*?)^DIAG_STAGE_END \1$", log, re.M|re.S):
        exits = re.findall(r"^DIAG_STAGE_EXIT "+name+r" (\d+)$", content, re.M)
        value = {"exit_code": int(exits[0]) if len(exits)==1 else None}
        if name.startswith("control") and value["exit_code"] == 0:
            value["kernels"] = parse_log(content)
        else:
            value["spans"] = [json.loads(line) for line in re.findall(r"^DIAG_SPAN (.+)$", content, re.M)]
            value["correctness_passes"] = len(re.findall(r"within tol=.*-> PASS", content))
            value["complete"] = "DIAG_CHECK_COMPLETE selected=1 runs=3 failures=0" in content
            value["cycles"] = [{"median": int(m), "values": json.loads(v)} for m,v in
                re.findall(r"median cycles=(\d+) \(of 3 runs: (\[[^]]+\])\)", content)]
        result["stages"][name] = value
    write(HERE / f"{label}-job-{job}-results.json", result)
    print(json.dumps({k:{key:v[key] for key in v if key!='spans'}|{"span_count":len(v.get('spans',[]))} for k,v in result['stages'].items()},indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare","submit","collect"])
    p.add_argument("label")
    p.add_argument("--build", default="profile-v1")
    p.add_argument("--job", type=int)
    a=p.parse_args()
    if a.action=="prepare": prepare(a.label,a.build)
    elif a.action=="submit": submit(a.label)
    else: collect(a.label,a.job)
