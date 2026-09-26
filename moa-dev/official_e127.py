#!/usr/bin/env python3
"""Submit the validated E127 source once, or read its recorded official status."""
import argparse
from datetime import datetime, timezone
import io
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]
def digest(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

PATH = ROOT / "docs/experiments/e127-official-submit-20260927.json"
CLI = [str(ROOT / "moa-dev/moa"), "moa-submitter", "--server", "https://micro2026-api.duckdns.org:7777"]


def submit():
    assert not PATH.exists(), "A submission was already attempted; inspect the record instead of submitting twice"
    subprocess.run([sys.executable, str(ROOT / "moa-dev/experiment_index.py"),
                    "--check", "--check-live"], cwd=ROOT, check=True)
    recipe = json.loads((ROOT / "baseline/e127.json").read_text())
    assert json.loads((ROOT / "baseline/current.json").read_text())["recipe"] == "baseline/e127.json"
    manifest = json.loads((ROOT / recipe["compile_inputs"]).read_text())
    report = json.loads((ROOT / recipe["performance_report"]).read_text())
    assert report["valid"] and report["independent_arena_jobs"] == 3
    for kind in ("extra", "probe"):
        assert json.loads((ROOT / f"docs/experiments/e127-c09-{kind}-correctness.json").read_text())["valid"]
    preflight = json.loads((ROOT / "docs/experiments/e127-official-payload-preflight-20260927.json").read_text())
    assert preflight["status"] == "LOCAL_CLI_PAYLOAD_CHECK_PASS"
    sources = preflight["source_files"]
    expected = {k: v for k, v in manifest.items() if k == "src/ops.rs" or k.startswith("src/device/")}
    assert sources == expected and len(sources) == 23
    remote = "/tmp/moa-official-e127-20260927"
    assert preflight["source"] == remote
    archive = subprocess.check_output(["docker", "exec", "--workdir", remote, "moa2026-dev", "tar", "-cf", "-", *sources])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        assert {m.name: digest(tar.extractfile(m).read()) for m in tar if m.isfile()} == sources
    commit = subprocess.check_output(["git", "rev-parse", "baseline-e127^{commit}"], cwd=ROOT, text=True).strip()
    for name, value in sources.items():
        data = subprocess.check_output(["git", "show", f"{commit}:furiosa-opt-gemma4-12B/{name}"], cwd=ROOT)
        assert digest(data) == value
    command = CLI + ["submit", "--source", remote]
    data = dict(started_at_utc=datetime.now(timezone.utc).isoformat(), source_checkpoint="E127",
        source_commit=commit, source_build_audit=recipe["source_build_audit"],
        payload_preflight="docs/experiments/e127-official-payload-preflight-20260927.json",
        source_files=sources, source_count=len(sources), command=command, accepted=False)
    with PATH.open("x") as out:
        json.dump(data, out, indent=2)
        out.write("\n")
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
    data.update(exit_code=result.returncode, response=result.stdout,
                finished_at_utc=datetime.now(timezone.utc).isoformat())
    match = re.search(r"^Submission:\s+([0-9a-f]{8})\s*$", result.stdout, re.M)
    if result.returncode == 0 and match:
        data.update(accepted=True, submission_id=match[1])
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
    path = ROOT / f"moa-dev/logs/e127-official-{data['submission_id']}-observation-{len(data.get('observations', [])) + 1}.log"
    path.write_text(log)
    observation["log"].update(local_log=str(path.relative_to(ROOT)), sha256=digest(log.encode())["sha256"])
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
