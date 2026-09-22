#!/usr/bin/env python3
"""Collect additional-input and full-shape diagnostic evidence, including upload hashes."""
import argparse
import json
import re
import subprocess
import sys
import tarfile
from build import HERE, common

ROOT, ARTIFACTS, digest = common.ROOT, common.ARTIFACTS, common.digest
sys.path.insert(0, str(ROOT / "moa-dev"))
from measure import cli
from extra_seeds import parse_log


def collect(kind, job):
    status = json.loads(cli("status", job))
    assert status["status"] in {"SUCCEEDED", "COMPLETED", "FAILED", "CANCELLED"}, status
    log = cli("logs", job)
    path = ROOT / f"moa-dev/logs/c04-{kind}-{job}.log"
    path.write_text(log)
    remote = f"/tmp/moa-c04/job-{job}.tar"
    cli("download", job, "--out", remote)
    archive = ARTIFACTS / f"job-{job}.tar"
    subprocess.run(["docker", "cp", "moa2026-dev:" + remote, str(archive)], check=True)
    with tarfile.open(archive) as tar:
        members = [m for m in tar if m.isfile()]
        assert len({m.name for m in members}) == len(members)
        files = {m.name: digest(tar.extractfile(m).read()) for m in members}
    build_path = HERE / ("build-c04-eight-ring32-extra-repro.json" if kind == "extra" else "build-projection-probe-v1.json")
    build = json.loads(build_path.read_text())
    assert {k: v for k, v in files.items() if k != "job.yaml"} == build["package_files"]
    result = {"job": status, "log": str(path.relative_to(ROOT)), "log_sha256": digest(log.encode())["sha256"],
              "package_files": files, "build": str(build_path.relative_to(ROOT)), "valid": False,
              "scope": "Additional correctness inputs; not a performance repeat or official evaluation."}
    try:
        assert status["status"] in {"SUCCEEDED", "COMPLETED"} and status["exit_code"] == 0
        if kind == "extra":
            result["results"] = parse_log(log)
            original = json.loads((HERE / "build-c04-eight-ring32-v1-repro.json").read_text())
            assert build["images"] == original["images"]
        else:
            matches = re.findall(r"^C04_PROBE case=(\d+) projection=(\w+) checked=(\d+) mismatches=(\d+)$", log, re.M)
            expected = {(str(c), name, str(count), "0") for c in range(4) for name, count in (("up", 15360), ("gate", 15360), ("down", 3840))}
            assert len(matches) == 12 and set(matches) == expected
            assert log.count("C04_PROBE_COMPLETE oracle_only=false cases=4 checked=138240 mismatches=0") == 1
            result["results"] = {"cases": 4, "projection_runs": 12, "elements_checked": 138240, "mismatches": 0, "details": matches}
        result["valid"] = True
    except (AssertionError, ValueError) as error:
        result["error"] = str(error) or "Required diagnostic check failed; inspect raw log"
    with (ROOT / f"docs/experiments/e79-c04-{kind}-correctness.json").open("x") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    print(json.dumps(result.get("results", result), indent=2))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["extra", "probe"])
    parser.add_argument("job", type=int)
    args = parser.parse_args()
    collect(args.kind, args.job)
