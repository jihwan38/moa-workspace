#!/usr/bin/env python3
"""Submit the validated E104 source once, or read its recorded official status."""
import argparse
from datetime import datetime, timezone
import io
import json
import re
import subprocess
import sys
import tarfile
from build import HERE, common

ROOT = common.ROOT
PATH = ROOT / "docs/experiments/e104-official-submit-20260922.json"
CLI = [str(ROOT / "moa-dev/moa"), "moa-submitter", "--server", "https://micro2026-api.duckdns.org:7777"]


def submit():
    assert not PATH.exists(), "A submission was already attempted; inspect the record instead of submitting twice"
    report = json.loads((ROOT / "docs/experiments/e104-c07-e5-reload-materialized.json").read_text())
    assert report["valid"] and report["independent_arena_jobs"] == 3
    for kind in ("extra", "probe"):
        assert json.loads((ROOT / f"docs/experiments/e104-c07-{kind}-correctness.json").read_text())["valid"]
    build = json.loads((HERE / "build-c07-e104-reviewed-v1-repro.json").read_text())
    preflight = json.loads((HERE / "official-payload-preflight.json").read_text())
    sources = preflight["source_files"]
    assert len(sources) == 23 and all(build["source_inputs"][k] == v for k, v in sources.items())
    remote = "/tmp/moa-c02-c03/c07-e104-reviewed-v1"
    archive = subprocess.check_output(["docker", "exec", "--workdir", remote, "moa2026-dev", "tar", "-cf", "-", *sources])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        assert {m.name: common.digest(tar.extractfile(m).read()) for m in tar if m.isfile()} == sources
    commit = subprocess.check_output(["git", "rev-parse", "baseline-e104^{commit}"], cwd=ROOT, text=True).strip()
    for name, value in sources.items():
        data = subprocess.check_output(["git", "show", f"{commit}:furiosa-opt-gemma4-12B/{name}"], cwd=ROOT)
        assert common.digest(data) == value
    command = CLI + ["submit", "--source", remote]
    data = dict(started_at_utc=datetime.now(timezone.utc).isoformat(), source_checkpoint="E104",
        source_commit=commit, source_build=str((HERE / "build-c07-e104-reviewed-v1-repro.json").relative_to(ROOT)),
        source_files=sources, source_count=len(sources), command=command, accepted=False)
    with PATH.open("x") as out:
        json.dump(data, out, indent=2)
        out.write("\n")
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
    data.update(exit_code=result.returncode, response=result.stdout,
                finished_at_utc=datetime.now(timezone.utc).isoformat())
    matches = re.findall(r"\b[0-9a-f]{8}\b", result.stdout)
    ids = set(matches)
    if result.returncode == 0 and len(ids) == 1:
        data.update(accepted=True, submission_id=ids.pop())
    PATH.write_text(json.dumps(data, indent=2) + "\n")
    print(result.stdout)
    assert data["accepted"], "Submission response needs inspection; no automatic retry"


def status():
    data = json.loads(PATH.read_text())
    assert data["accepted"] and re.fullmatch(r"[0-9a-f]{8}", data["submission_id"])
    observation = {"checked_at_utc": datetime.now(timezone.utc).isoformat()}
    for action in ("status", "log"):
        command = CLI + [action, data["submission_id"]]
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=45)
        observation[action] = dict(command=command, exit_code=result.returncode, output=result.stdout)
        assert result.returncode == 0, result.stdout
    status, log = observation["status"]["output"], observation["log"]["output"]
    match = re.search(r"^Status:\s+(\w+)", status, re.M)
    assert match, status
    data.update(current_status=match[1], server_contract_ok="contract OK" in log)
    if data["current_status"] == "completed":
        scores = re.findall(r"^Score:\s+([\d.]+)\s*$", status, re.M)
        cycles = re.findall(r"^\s+(ops::\w+)\s+(\d+)\s*$", status, re.M)
        assert len(scores) == 1 and len(cycles) == 3
        sys.path.insert(0, str(ROOT / "moa-dev"))
        from measure import parse_log
        checked = parse_log(log)
        assert all(checked[k.removeprefix("ops::")]["median_cycles"] == int(v) for k, v in cycles)
        data.update(official_score=float(scores[0]), official_cycles={k: int(v) for k, v in cycles},
                    official_rank=None, official_basic_comparisons_passed=15)
    path = ROOT / f"moa-dev/logs/e104-official-{data['submission_id']}-observation-{len(data.get('observations', [])) + 1}.log"
    path.write_text(log)
    observation["log"].update(local_log=str(path.relative_to(ROOT)), sha256=common.digest(log.encode())["sha256"])
    data.setdefault("observations", []).append(observation)
    PATH.write_text(json.dumps(data, indent=2) + "\n")
    print(status)
    if data["current_status"] in {"failed", "rejected"}:
        print(log[-5000:])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["submit", "status"])
    args = parser.parse_args()
    submit() if args.action == "submit" else status()
