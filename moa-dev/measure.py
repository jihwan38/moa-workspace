#!/usr/bin/env python3
"""Collect immutable Arena jobs and their reruns into a reproducible experiment record."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
MOA = ROOT / "moa-dev/moa"
KERNELS = ("sliding_project_qkv", "sliding_attention_output", "decoder_feedforward")
OUTPUTS = {KERNELS[0]: ("q", "k", "v"), KERNELS[1]: ("",), KERNELS[2]: ("",)}


def parse_log(log):
    """Require the complete three-seed correctness sweep and complete cycle samples."""
    if "all 3 tests passed" not in log or re.search(r"\bFAIL\b", log):
        raise ValueError("The complete correctness sweep did not pass")
    results = {}
    sections = re.findall(r"^==> (\w+)\s*\n(.*?)(?=^==>|\Z)", log, re.M | re.S)
    for name, section in sections:
        if name not in KERNELS:
            continue
        if name in results:
            raise ValueError(f"Duplicate kernel section: {name}")
        expected_labels = {f"{name}{' ' + output if output else ''} run{run}"
                           for output in OUTPUTS[name] for run in range(3)}
        comparisons = re.findall(r"^\[([^]]+)\].*within tol=\s*([\d.]+)%\s*-> (PASS|FAIL)$",
                                 section, re.M)
        labels = [label.strip() for label, _, _ in comparisons]
        if set(labels) != expected_labels or len(labels) != len(expected_labels):
            raise ValueError(f"Incomplete or duplicate output comparisons: {name}")
        if any(status != "PASS" or float(tolerance) != 100.0
               for _, tolerance, status in comparisons):
            raise ValueError(f"Output comparison failed: {name}")
        matches = re.findall(r"median cycles=(\d+) \(of (\d+) runs: \[([\d, ]+)\]\)", section)
        if len(matches) != 1:
            raise ValueError(f"Missing or duplicate cycle measurements: {name}")
        reported, count, samples = matches[0]
        cycles = [int(value.strip()) for value in samples.split(",")]
        if int(count) != 3 or len(cycles) != 3 or min(cycles) <= 0:
            raise ValueError(f"Invalid cycle sample count or value: {name}")
        if statistics.median(cycles) != int(reported):
            raise ValueError(f"Reported median disagrees with samples: {name}")
        results[name] = {"correctness": "passed", "median_cycles": int(reported),
                         "cycles_by_run": cycles, "output_comparisons_passed": len(labels)}
    if set(results) != set(KERNELS):
        raise ValueError("Missing kernel result")
    return results


def cli(*args):
    return subprocess.check_output([str(MOA), "furiosa-arena", *map(str, args)],
                                   text=True, stderr=subprocess.STDOUT)


def package_fingerprint(job_id):
    """Hash the actual uploaded payload, not today's potentially changed workspace."""
    package = ROOT / "furiosa-opt-gemma4-12B/target/arena-packages" / f"job-{job_id}.tar"
    package.parent.mkdir(parents=True, exist_ok=True)
    container_path = f"/workspace/target/arena-packages/job-{job_id}.tar"
    cli("download", job_id, "--out", container_path, "--create-dirs")
    files = {}
    with tarfile.open(package) as archive:
        for member in archive.getmembers():
            if member.isfile():
                if member.name in files:
                    raise ValueError(f"Duplicate archive member: {member.name}")
                content = archive.extractfile(member).read()
                files[member.name] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    required = {"test_runtime", "fixtures.safetensors", "remote_entrypoint.sh", "job.yaml"}
    if set(files) != required:
        raise ValueError(f"Unexpected package contents for job {job_id}: {sorted(files)}")
    return files


def verify_artifacts(runs, provenance=None):
    """Never pool different binary, fixture, runner or manifest contents."""
    first = runs[0]["package_files"]
    for run in runs[1:]:
        if run["package_files"] != first:
            raise ValueError("Jobs do not contain the same immutable benchmark package")
    if provenance and "artifacts" in provenance:
        aliases = {"target/release/test_kernels": "test_runtime", "ref/fixtures.safetensors": "fixtures.safetensors"}
        for source, uploaded in aliases.items():
            expected = provenance["artifacts"].get(source)
            if expected and expected["sha256"] != first[uploaded]["sha256"]:
                raise ValueError(f"Provenance does not match uploaded {uploaded}")


def collect(job_id, label, timeout):
    deadline = time.monotonic() + timeout
    previous = None
    while True:
        status = json.loads(cli("status", job_id))
        state = status["status"].upper()
        display = (state, status.get("queue_ahead"))
        if display != previous:
            print(f"job {job_id}: {state}, queue_ahead={display[1]}", flush=True)
            previous = display
        if state in {"SUCCEEDED", "FAILED", "CANCELLED", "CANCELED", "COMPLETED"} or status.get("exit_code") is not None:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"job {job_id} is still {state}; it remains on Arena")
        time.sleep(5)
    log = cli("logs", job_id)
    log_path = ROOT / "moa-dev/logs" / f"{label}-job-{job_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(log)
    result = {"job": status, "log": str(log_path.relative_to(ROOT)),
              "log_sha256": hashlib.sha256(log.encode()).hexdigest(),
              "package_files": package_fingerprint(job_id)}
    try:
        if status.get("exit_code") != 0 or state not in {"SUCCEEDED", "COMPLETED"}:
            raise ValueError(f"Arena job did not succeed: {state}, exit={status.get('exit_code')}")
        result["kernels"] = parse_log(log)
        result["valid"] = True
    except ValueError as exc:
        result.update(valid=False, error=str(exc))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, nargs="+", required=True)
    parser.add_argument("--reruns", type=int, default=0, help="Additional reruns of the first job; queued serially")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, help="Previously recorded source/artifact metadata; not inferred from today's tree")
    parser.add_argument("--leaderboard-snapshot", type=Path)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", args.label) or args.reruns < 0:
        parser.error("Use a simple alphanumeric label and non-negative reruns")
    if len(set(args.jobs)) != len(args.jobs):
        parser.error("Duplicate job IDs are not independent repetitions")
    if args.output.exists():
        parser.error(f"Refusing to overwrite experiment: {args.output}")
    record = {"schema_version": 1, "label": args.label,
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "runs": [],
              "method": "Each job uses 3 official input configurations; aggregate medians across separate jobs."}
    if args.provenance:
        record["provenance"] = json.loads(args.provenance.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(args.output)

    save()
    try:
        for index in range(len(args.jobs) + args.reruns):
            if index < len(args.jobs):
                job_id = args.jobs[index]
            else:
                response = cli("rerun", args.jobs[0])
                match = re.search(r"submitted job (\d+)", response)
                if not match:
                    raise RuntimeError(f"Could not identify submitted rerun; inspect Arena before retrying: {response}")
                job_id = int(match[1])
                record.setdefault("rerun_parent", {})[str(job_id)] = args.jobs[0]
            record["pending_job"] = job_id
            save()
            result = collect(job_id, args.label, args.timeout)
            record["runs"].append(result)
            del record["pending_job"]
            save()
            if not result["valid"]:
                raise ValueError(result["error"])
        verify_artifacts(record["runs"], record.get("provenance"))
        record["aggregate"] = {}
        for kernel in KERNELS:
            medians = [run["kernels"][kernel]["median_cycles"] for run in record["runs"]]
            median = statistics.median(medians)
            record["aggregate"][kernel] = {"median_cycles": median, "job_medians": medians,
                                             "min_cycles": min(medians), "max_cycles": max(medians),
                                             "relative_range": (max(medians) - min(medians)) / median}
        if args.leaderboard_snapshot:
            board = json.loads(args.leaderboard_snapshot.read_text())
            reference = board["data"]["baselineCycles"]
            score = math.prod(reference[f"ops::{k}"] / record["aggregate"][k]["median_cycles"]
                              for k in KERNELS) ** (1 / 3)
            record["indicative_score"] = {"value": score, "reference": str(args.leaderboard_snapshot),
                                           "note": "Diagnostic estimate only; not an official score or rank."}
        record["valid"] = True
    except (ValueError, RuntimeError, TimeoutError, OSError, tarfile.TarError, subprocess.CalledProcessError) as exc:
        record.update(valid=False, error=str(exc))
    record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    print(json.dumps({"output": str(args.output), "valid": record["valid"],
                      "aggregate": record.get("aggregate"), "error": record.get("error")}, indent=2), flush=True)
    return 0 if record["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
