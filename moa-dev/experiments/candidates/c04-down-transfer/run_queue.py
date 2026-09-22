#!/usr/bin/env python3
"""Finish the bounded C04 validation queue; at most two of our jobs are active.

Submissions are recorded before the command. An interrupted/uncertain submission
is never retried automatically. This script does not change server scheduling.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from build import HERE, common
from screen import cli, record, submit as submit_screen

TERMINAL = {"SUCCEEDED", "COMPLETED", "FAILED", "CANCELLED", "CANCELED"}


def saved_submission(name, command):
    path = HERE / (name + "-submission.json")
    if path.exists():
        data = json.loads(path.read_text())
        assert "job_id" in data, f"Inspect incomplete submission manually: {path}"
        return data["job_id"]
    record(path, {"attempted_at_utc": datetime.now(timezone.utc).isoformat(), "command": command})
    response = cli(*command)
    data = json.loads(path.read_text())
    data["response"] = response
    matches = re.findall(r"submitted job (\d+)", response)
    if len(matches) == 1:
        data["job_id"] = int(matches[0])
    path.write_text(json.dumps(data, indent=2) + "\n")
    assert len(matches) == 1, response
    print(response.strip(), flush=True)
    return data["job_id"]


def diagnostic(kind):
    label = "c04-eight-ring32-extra" if kind == "extra" else "projection-probe-v1"
    build = json.loads((HERE / (f"build-{label}-repro.json" if kind == "extra" else f"build-{label}.json")).read_text())
    assert build.get("host_build_exit_code", build.get("build_exit_code")) == 0
    local = common.ARTIFACTS / label / "package"
    assert {p.name: common.digest(p.read_bytes()) for p in local.iterdir()} == build["package_files"]
    remote = build["container_package"] if kind == "extra" else "/tmp/moa-c04/projection-probe-v1/package"
    for name, value in build["package_files"].items():
        actual = subprocess.check_output(["docker", "exec", "moa2026-dev", "sha256sum", remote + "/" + name], text=True).split()[0]
        assert actual == value["sha256"]
    return saved_submission("c04-" + kind, ["submit", *(remote + "/" + n for n in build["package_files"]),
        "--entrypoint", "remote_entrypoint.sh" if kind == "extra" else "run.sh", "--timeout", "70", "--name", "beef-c04-" + kind])


def collect(job, label, kind=None):
    if kind:
        result = common.ROOT / f"docs/experiments/e79-c04-{kind}-correctness.json"
        command = [sys.executable, str(HERE / "diagnostics.py"), kind, str(job)]
    else:
        original = json.loads((HERE / f"{label}-submission.json").read_text())["job_id"]
        suffix = "" if original == job else f"-job-{job}"
        result = HERE / f"{label}{suffix}-results.json"
        command = [sys.executable, str(HERE / "screen.py"), "collect", label, "--job", str(job)]
    if not result.exists():
        process = subprocess.run(command, capture_output=True, text=True)
        if process.returncode != 0:
            print(process.stdout + process.stderr, flush=True)
            raise RuntimeError(f"Collection failed: {job}; inspect preserved result")
    data = json.loads(result.read_text())
    assert data["valid"], data
    if kind:
        print(f"job {job}: {kind} PASS", flush=True)
    else:
        print(f"job {job}: " + str({s: v["kernels"]["decoder_feedforward"]["median_cycles"] for s, v in data["stages"].items()}), flush=True)


if __name__ == "__main__":
    active = [(74152, "c04-screen-third", None), (74154, "c04-screen-second", None)]
    pending = ["repeat", "extra", "probe", "fourth"]
    previous = {}
    deadline = time.monotonic() + 1800
    while active or pending:
        assert time.monotonic() < deadline, "Queue deadline; Arena jobs remain untouched"
        for item in list(active):
            job, label, kind = item
            status = json.loads(cli("status", job))
            value = (status["status"], status.get("queue_ahead"))
            if previous.get(job) != value:
                print(f"job {job}: {value}", flush=True)
                previous[job] = value
            if status["status"] in TERMINAL:
                collect(job, label, kind)
                active.remove(item)
        while len(active) < 2 and pending:
            item = pending.pop(0)
            if item == "repeat":
                job = saved_submission("repeat-second-2", ["rerun", "74142"])
                active.append((job, "c04-screen-second", None))
            elif item == "fourth":
                path = HERE / "c04-screen-fourth-submission.json"
                if not path.exists():
                    submit_screen("c04-screen-fourth")
                job = json.loads(path.read_text())["job_id"]
                active.append((job, "c04-screen-fourth", None))
            else:
                active.append((diagnostic(item), None, item))
        if active:
            time.sleep(20)
    print("C04_QUEUE_COMPLETE", flush=True)
