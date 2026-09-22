#!/usr/bin/env python3
"""Immutable diagnostic submissions, reruns and upload-verified collection."""
import argparse
from datetime import datetime, timezone
import json
import re
import subprocess
import tarfile
from build import HERE, common
from screen import cli, record

TERMINAL = {"SUCCEEDED", "COMPLETED", "FAILED", "CANCELLED", "CANCELED"}


def saved_submit(label, command):
    path = HERE / f"{label}-submission.json"
    record(path, {"attempted_at_utc": datetime.now(timezone.utc).isoformat(), "command": command})
    response = cli(*command)
    data = json.loads(path.read_text())
    data["response"] = response
    match = re.findall(r"submitted job (\d+)", response)
    if len(match) == 1:
        data["job_id"] = int(match[0])
    path.write_text(json.dumps(data, indent=2) + "\n")
    assert len(match) == 1, response
    print(response, end="")


def submit_probe(label):
    build = json.loads((HERE / f"build-{label}.json").read_text())
    assert build["build_exit_code"] == build["oracle_exit_code"] == 0
    local = common.ARTIFACTS / label / "package"
    assert {p.name: common.digest(p.read_bytes()) for p in local.iterdir()} == build["package_files"]
    remote = "/tmp/moa-c07/" + label + "/package"
    for name, value in build["package_files"].items():
        actual = subprocess.check_output(["docker", "exec", "moa2026-dev", "sha256sum", remote + "/" + name], text=True).split()[0]
        assert actual == value["sha256"]
    saved_submit(label, ["submit", *(remote + "/" + n for n in build["package_files"]),
        "--entrypoint", "run.sh", "--timeout", "70", "--name", "beef-c07-" + label])


def collect_probe(label):
    job = json.loads((HERE / f"{label}-submission.json").read_text())["job_id"]
    status = json.loads(cli("status", job))
    assert status["status"] in TERMINAL, status
    log = cli("logs", job)
    path = common.ROOT / f"moa-dev/logs/c07-{label}-{job}.log"
    path.write_text(log)
    remote = f"/tmp/moa-c07/job-{job}.tar"
    cli("download", job, "--out", remote)
    archive = common.ARTIFACTS / f"job-{job}.tar"
    subprocess.run(["docker", "cp", "moa2026-dev:" + remote, str(archive)], check=True)
    with tarfile.open(archive) as tar:
        members = [m for m in tar if m.isfile()]
        assert len({m.name for m in members}) == len(members)
        files = {m.name: common.digest(tar.extractfile(m).read()) for m in members}
    build = json.loads((HERE / f"build-{label}.json").read_text())
    assert {k: v for k, v in files.items() if k != "job.yaml"} == build["package_files"]
    details = re.findall(r"^C07_PROBE case=(\d+) projection=(\w+) checked=(\d+) mismatches=(\d+)$", log, re.M)
    expected = {(str(c), name, str(count)) for c in range(6) for name, count in (("up", 15360), ("gate", 15360), ("down", 3840))}
    complete = len(details) == 18 and {tuple(d[:3]) for d in details} == expected
    valid = complete and all(d[3] == "0" for d in details) and status["status"] in {"SUCCEEDED", "COMPLETED"} and status["exit_code"] == 0
    data = {"job": status, "build": f"build-{label}.json", "log": str(path.relative_to(common.ROOT)),
            "log_sha256": common.digest(log.encode())["sha256"], "uploaded_package_verified": True,
            "package_files": files, "details": details, "complete": complete, "valid": valid,
            "scope": "Separate diagnostic images; exact independent projection oracle, not official tolerance/performance."}
    record(HERE / f"{label}-results.json", data)
    print(json.dumps({k:data[k] for k in ("job", "details", "complete", "valid")}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["submit-probe", "collect-probe", "rerun"])
    p.add_argument("label")
    p.add_argument("job", nargs="?")
    args = p.parse_args()
    if args.action == "submit-probe":
        submit_probe(args.label)
    elif args.action == "collect-probe":
        collect_probe(args.label)
    else:
        assert args.job and args.job.isdigit()
        saved_submit(args.label, ["rerun", args.job])
