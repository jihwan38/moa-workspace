#!/usr/bin/env python3
"""Read and archive the already accepted E74 submission; never submit or retry."""
from datetime import datetime, timezone
import json
import re
import subprocess
import sys
from build import common

ROOT = common.ROOT
path = ROOT / "docs/experiments/e74-official-submit-20260922.json"
data = json.loads(path.read_text())
assert data["accepted"] and data["submission_id"] == "01ba2ec2"
observation = {"checked_at_utc": datetime.now(timezone.utc).isoformat()}
for action in ("status", "log"):
    command = [str(ROOT / "moa-dev/moa"), "moa-submitter", "--server", "https://micro2026-api.duckdns.org:7777", action, data["submission_id"]]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=45)
    observation[action] = {"command": command, "exit_code": result.returncode, "output": result.stdout}
    assert result.returncode == 0, result.stdout
data.setdefault("observations", []).append(observation)
status = observation["status"]["output"]
log = observation["log"]["output"]
match = re.search(r"^Status:\s+(\w+)", status, re.M)
assert match, status
data["current_status"] = match[1]
data["server_contract_ok"] = "contract OK" in log
if data["current_status"] == "completed":
    score = re.findall(r"^Score:\s+([\d.]+)\s*$", status, re.M)
    cycles = re.findall(r"^\s+(ops::\w+)\s+(\d+)\s*$", status, re.M)
    assert len(score) == 1 and len(cycles) == 3
    sys.path.insert(0, str(ROOT / "moa-dev"))
    from measure import parse_log
    checked = parse_log(log)
    assert all(checked[name.removeprefix("ops::")]["median_cycles"] == int(value) for name, value in cycles)
    data.update(official_score=float(score[0]), official_cycles={k: int(v) for k, v in cycles},
                official_rank=None, official_basic_comparisons_passed=15)
log_path = ROOT / "moa-dev/logs/e74-official-01ba2ec2.log"
log_path.write_text(log)
observation["log"].update(local_log=str(log_path.relative_to(ROOT)), sha256=common.digest(log.encode())["sha256"])
path.write_text(json.dumps(data, indent=2) + "\n")
print(status)
if data["current_status"] in {"failed", "rejected"}:
    print(log[-5000:])
