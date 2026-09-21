#!/usr/bin/env python3
"""Independent development correctness sweep for run3..8; never an official score.

Host Python/stdlib only. The existing Docker container supplies reference/build tools.
Commands are explicit: prepare generates CPU references; stage selects a capture.py
snapshot; build compiles the isolated scratch crate; submit sends one 70-second Arena
job; report checks its uploaded package and all 30 output comparisons. Nothing edits
the live crate, starts Docker, or uses its target directory or Cargo/cache directories.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
CRATE = ROOT / "furiosa-opt-gemma4-12B"
ARTIFACTS = ROOT / "moa-dev/experiments/artifacts"
RUN_IDS = list(range(3, 9))
OUTPUTS = {"sliding_project_qkv": ("q", "k", "v"),
           "sliding_attention_output": ("",), "decoder_feedforward": ("",)}
REFERENCE_FILES = ("src/bin/test_kernels.rs", "scripts/generate_references.py",
                   "scripts/fixture_prng.py", "scripts/reference/gemma4.py",
                   "ref/fixtures.safetensors", "scripts/rngd/remote_entrypoint.sh")


def fingerprint(content):
    return {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular file: {path}")
    return path.read_bytes()


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)
    path.chmod(0o444)


def write_json(path, value):
    write(path, (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode())


def run(*args):
    return subprocess.check_output([str(a) for a in args], text=True,
                                   stderr=subprocess.STDOUT, timeout=60).strip()


def inside(container, *args):
    return run("docker", "exec", container, *args)


def environment(container, expected=None):
    info = json.loads(run("docker", "inspect", container))[0]
    if not info["State"]["Running"]:
        raise ValueError("The existing container must be running; this helper never starts it")
    if expected and (info["Id"] != expected["container_id"] or info["Image"] != expected["image_id"]):
        raise ValueError("The container/image changed since scratch preparation")
    result = {"container": container, "container_id": info["Id"], "image_id": info["Image"],
              "python": inside(container, "python", "--version"),
              "rustc": inside(container, "rustc", "-Vv"),
              "compiler_sha256": inside(container, "sha256sum", "/home/moa/.cargo/bin/cargo-furiosa-opt",
                                         "/home/moa/.cargo/bin/furiosa-opt-driver")}
    if expected and result != expected:
        raise ValueError("Compiler or runtime environment changed after staging")
    return result


def stream_run(command, log):
    with log.open("x") as output:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            output.write(line)
            output.flush()
            print(line, end="", flush=True)
        status = process.wait()
    if status:
        raise ValueError(f"Command exited {status}; see {log}")


def patch_once(content, replacements):
    for before, after in replacements:
        if content.count(before) != 1:
            raise ValueError(f"Upstream layout changed; expected exactly one {before!r}")
        content = content.replace(before, after)
    return content


def extended_harness(content):
    return patch_once(content, [(b"const RUNS: usize = 3;", b"const RUNS: usize = 6;\nconst RUN_OFFSET: usize = 3;"),
                                (b"for run in 0..RUNS {", b"for run in RUN_OFFSET..RUN_OFFSET + RUNS {")])


def extended_generator(content):
    return patch_once(content, [(b"\nRUNS = 3\n", b"\nRUNS = 6\nRUN_OFFSET = 3\n"),
                                (b"for run in range(RUNS):", b"for run in range(RUN_OFFSET, RUN_OFFSET + RUNS):")])


def tensor_header(content):
    size = struct.unpack("<Q", content[:8])[0]
    header = json.loads(content[8:8 + size])
    return {name: {key: value[key] for key in ("dtype", "shape")}
            for name, value in header.items() if name != "__metadata__"}


def verify_fixture(original, extra):
    baseline = tensor_header(original)
    run0 = {name.removeprefix("run0."): value for name, value in baseline.items() if name.startswith("run0.")}
    if not run0:
        raise ValueError("Original fixture has no run0 schema")
    expected = {f"run{run}.{name}": value for run in RUN_IDS for name, value in run0.items()}
    actual = tensor_header(extra)
    if actual != expected:
        raise ValueError("Generated fixture names/dtypes/shapes are not exactly run3..8 of the original schema")
    return {"tensors": len(actual), "run_ids": RUN_IDS, "schema_matches_original": True}


def new_directory(label):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", label):
        raise ValueError("Use 1–80 alphanumeric/hyphen/underscore characters for the label")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    result = ARTIFACTS / label
    result.mkdir()  # Exclusive: never overwrite an old experiment, including a failed one.
    return result


def load(directory, filename, kind):
    manifest = json.loads(read(directory / filename))
    if manifest["kind"] != kind or manifest["run_ids"] != RUN_IDS:
        raise ValueError(f"Unexpected manifest: {directory / filename}")
    for name, expected in manifest["files"].items():
        if fingerprint(read(directory / name)) != expected:
            raise ValueError(f"Snapshot file changed: {directory / name}")
    return manifest


def prepare(args):
    env = environment(args.container)
    original = {name: read(CRATE / name) for name in REFERENCE_FILES}
    files = {"upstream/" + name: data for name, data in original.items()}
    for name, data in original.items():
        if name != "ref/fixtures.safetensors":
            files["crate/" + name] = data
    files["crate/src/bin/test_kernels.rs"] = extended_harness(original["src/bin/test_kernels.rs"])
    files["crate/scripts/generate_references.py"] = extended_generator(original["scripts/generate_references.py"])
    directory = new_directory(args.label)
    for name, content in files.items():
        write(directory / name, content)
    remote = "/tmp/moa-extra-seeds-" + args.label
    inside(args.container, "mkdir", remote)
    run("docker", "cp", "-a", str(directory / "crate"), f"{args.container}:{remote}/crate")
    # docker cp preserves ownership/modes; allow only the scratch owner to create ref/.
    inside(args.container, "chmod", "-R", "u+rwX", remote + "/crate")
    command = ["docker", "exec", "--workdir", remote + "/crate", args.container,
               "env", "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=2",
               "PYTHONUNBUFFERED=1", "python", "scripts/generate_references.py"]
    stream_run(command, directory / "reference-generation.log")
    (directory / "crate/ref").mkdir(parents=True, exist_ok=True)
    run("docker", "cp", f"{args.container}:{remote}/crate/ref/fixtures.safetensors",
        str(directory / "crate/ref/fixtures.safetensors"))
    (directory / "crate/ref/fixtures.safetensors").chmod(0o444)
    fixture = read(directory / "crate/ref/fixtures.safetensors")
    validation = verify_fixture(original["ref/fixtures.safetensors"], fixture)
    files["crate/ref/fixtures.safetensors"] = fixture
    for name, data in original.items():
        if read(CRATE / name) != data:
            raise ValueError(f"Official reference file changed during generation: {name}")
    manifest = {"schema_version": 1, "kind": "extra_seed_references", "label": args.label,
                "created_at_utc": now(), "run_ids": RUN_IDS, "official_score": False,
                "environment": env, "container_directory": remote,
                "files": {name: fingerprint(data) for name, data in files.items()},
                "fixture_validation": validation,
                "changes": "Only RUNS and the run loop offset changed; all PRNG/distributions/tolerances remain byte-identical."}
    write_json(directory / "references.json", manifest)
    print(f"CPU references ready: {directory / 'references.json'}", flush=True)


def stage(args):
    refs = load(args.references, "references.json", "extra_seed_references")
    source = json.loads(read(args.snapshot / "provenance.json"))
    originals = source["snapshot"]["files"]
    selected = [name for name in originals if name.startswith("src/")]
    selected += ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml"]
    files = {}
    for name in selected:
        data = read(args.snapshot / name)
        if fingerprint(data)["sha256"] != originals[name]["sha256"]:
            raise ValueError(f"Candidate source differs from capture provenance: {name}")
        files["crate/" + name] = data
    if files["crate/src/bin/test_kernels.rs"] != read(args.references / "upstream/src/bin/test_kernels.rs"):
        raise ValueError("Candidate's official harness differs from the reference harness")
    for name in refs["files"]:
        if name.startswith("crate/"):
            files[name] = read(args.references / name)
    env = environment(args.container)
    if env["image_id"] != refs["environment"]["image_id"]:
        raise ValueError("Use the reference generation image for candidate validation")
    directory = new_directory(args.label)
    for name, content in files.items():
        write(directory / name, content)
    remote = "/tmp/moa-extra-seeds-" + args.label
    inside(args.container, "mkdir", remote)
    run("docker", "cp", "-a", str(directory / "crate"), f"{args.container}:{remote}/crate")
    inside(args.container, "chmod", "-R", "u+rwX", remote + "/crate")
    manifest = {"schema_version": 1, "kind": "extra_seed_candidate", "label": args.label,
                "created_at_utc": now(), "run_ids": RUN_IDS, "official_score": False,
                "environment": env, "container_directory": remote,
                "references": {"path": str(args.references), **fingerprint(read(args.references / "references.json"))},
                "candidate": {"path": str(args.snapshot), "label": source["label"],
                              "provenance_sha256": fingerprint(read(args.snapshot / "provenance.json"))["sha256"],
                              "official_binary_sha256": source["artifacts"]["target/release/test_kernels"]["sha256"]},
                "files": {name: fingerprint(data) for name, data in files.items()},
                "note": "This creates a new six-run binary. Candidate's old three-run binary is not executed."}
    write_json(directory / "candidate.json", manifest)
    print(f"Candidate staged without compiling: {directory}", flush=True)


def verified_candidate(directory):
    manifest = load(directory, "candidate.json", "extra_seed_candidate")
    env = manifest["environment"]
    environment(env["container"], env)
    # Prevent silent edits in the container scratch copy before/after compilation.
    names = list(manifest["files"])
    remote = manifest["container_directory"]
    hashes = inside(env["container"], "sha256sum", *(remote + "/" + name for name in names))
    actual = {line.split(maxsplit=1)[1]: line.split()[0] for line in hashes.splitlines()}
    if actual != {remote + "/" + name: manifest["files"][name]["sha256"] for name in names}:
        raise ValueError("Container scratch sources/fixture changed after staging")
    return manifest


def build(args):
    manifest = verified_candidate(args.directory)
    if (args.directory / "build.json").exists():
        raise ValueError("Already built; create a new label for another build")
    remote, container = manifest["container_directory"], manifest["environment"]["container"]
    command = ["docker", "exec", "--workdir", remote + "/crate", container, "env",
               "CARGO_HOME=" + remote + "/cargo-home", "CARGO_TARGET_DIR=" + remote + "/target",
               "XDG_CACHE_HOME=" + remote + "/cache", "RUSTUP_HOME=/home/moa/.rustup",
               "cargo", "furiosa-opt", "build", "--release", "--locked", "--bin", "test_kernels"]
    stream_run(command, args.directory / "build.log")
    verified_candidate(args.directory)
    payload = args.directory / "payload"
    payload.mkdir()
    run("docker", "cp", f"{container}:{remote}/target/release/test_kernels", str(payload / "test_runtime"))
    for target, relative in {"fixtures.safetensors": "ref/fixtures.safetensors",
                             "remote_entrypoint.sh": "scripts/rngd/remote_entrypoint.sh"}.items():
        write(payload / target, read(args.directory / "crate" / relative))
    for name in ("test_runtime", "remote_entrypoint.sh"):
        (payload / name).chmod(0o555)
    record = {"built_at_utc": now(), "command": command,
              "candidate_manifest_sha256": fingerprint(read(args.directory / "candidate.json"))["sha256"],
              "payload": {path.name: fingerprint(read(path)) for path in payload.iterdir()},
              "official_score": False}
    write_json(args.directory / "build.json", record)
    print(f"Independent extra-seed binary ready: {payload / 'test_runtime'}", flush=True)


def built_candidate(directory):
    manifest = load(directory, "candidate.json", "extra_seed_candidate")
    record = json.loads(read(directory / "build.json"))
    if record["candidate_manifest_sha256"] != fingerprint(read(directory / "candidate.json"))["sha256"]:
        raise ValueError("Candidate manifest changed after compilation")
    if set(record["payload"]) != {"test_runtime", "fixtures.safetensors", "remote_entrypoint.sh"}:
        raise ValueError("Unexpected build payload")
    for name, expected in record["payload"].items():
        if fingerprint(read(directory / "payload" / name)) != expected:
            raise ValueError(f"Compiled payload changed: {name}")
    environment(manifest["environment"]["container"], manifest["environment"])
    return manifest, record


def submit(args):
    manifest, record = built_candidate(args.directory)
    reservation = args.directory / "submit-attempt.json"
    write_json(reservation, {"started_at_utc": now(), "timeout_seconds": 70,
                             "note": "One submission attempt per label, including uncertain network failures."})
    container, remote = manifest["environment"]["container"], manifest["container_directory"]
    run("docker", "cp", "-a", str(args.directory / "payload"), f"{container}:{remote}/payload")
    for name, expected in record["payload"].items():
        actual = inside(container, "sha256sum", remote + "/payload/" + name).split()[0]
        if actual != expected["sha256"]:
            raise ValueError(f"Staged submission payload differs: {name}")
    text = inside(container, "furiosa-arena", "submit",
                  *(remote + "/payload/" + name for name in ("remote_entrypoint.sh", "test_runtime", "fixtures.safetensors")),
                  "--name", "extra_seeds_" + manifest["label"], "--entrypoint", "remote_entrypoint.sh", "--timeout", "70")
    write(args.directory / "submission.log", (text + "\n").encode())
    matches = re.findall(r"submitted job (\d+)", text)
    if len(matches) != 1:
        raise ValueError("Cannot identify the submitted job; inspect submission.log before any retry")
    write_json(args.directory / "submission.json", {"job_id": int(matches[0]), "submitted_at_utc": now(),
                                                  "timeout_seconds": 70, "official_score": False})
    print(text, flush=True)


def parse_log(log):
    """Require all 30 comparisons exactly once; cycle counts are diagnostics only."""
    if re.search(r"\bFAIL\b", log) or log.count("\nall 3 tests passed\n") != 1:
        raise ValueError("The complete six-run correctness sweep did not pass")
    if "3 cases x 6 independently-seeded runs" not in log:
        raise ValueError("Not the six-run development harness")
    expected = {f"{name}{' ' + output if output else ''} run{run}"
                for name, outputs in OUTPUTS.items() for output in outputs for run in RUN_IDS}
    comparisons = re.findall(r"^\[([^]]+)\].*within tol=\s*([\d.]+)%\s*-> (PASS|FAIL)$", log, re.M)
    labels = [label.strip() for label, _, _ in comparisons]
    if len(labels) != len(expected) or set(labels) != expected:
        raise ValueError("Missing, extra or duplicate output comparisons for run3..8")
    if any(state != "PASS" or float(percent) != 100.0 for _, percent, state in comparisons):
        raise ValueError("Not all output elements passed the unchanged tolerance")
    cycles = {}
    sections = re.findall(r"^==> (\w+)\s*\n(.*?)(?=^==>|\Z)", log, re.M | re.S)
    for kernel, section in sections:
        if kernel not in OUTPUTS or kernel in cycles:
            raise ValueError("Unexpected or duplicate kernel section")
        matches = re.findall(r"median cycles=(\d+) \(of (\d+) runs: \[([\d, ]+)\]\)", section)
        if len(matches) != 1:
            raise ValueError(f"Missing or duplicate six-run cycle summary: {kernel}")
        reported, count, raw = matches[0]
        samples = [int(value.strip()) for value in raw.split(",")]
        if int(count) != 6 or len(samples) != 6 or min(samples) <= 0 or sorted(samples)[3] != int(reported):
            raise ValueError(f"Invalid six-run upper median: {kernel}")
        cycles[kernel] = {"cycles_by_run": samples, "upper_median_cycles_diagnostic": int(reported)}
    if set(cycles) != set(OUTPUTS):
        raise ValueError("Missing kernel sections")
    return {"correctness": "passed", "run_ids": RUN_IDS, "output_comparisons_passed": len(labels),
            "official_score": False, "cycles_diagnostic_only": cycles,
            "note": "Unmodified harness uses sorted[len/2], the upper median for six samples; no official score is computed."}


def report(args):
    if args.output.exists():
        raise ValueError("Report already exists; no overwrite")
    manifest, built = built_candidate(args.directory)
    submission = json.loads(read(args.directory / "submission.json"))
    container, remote = manifest["environment"]["container"], manifest["container_directory"]
    job = str(submission["job_id"])
    status = json.loads(inside(container, "furiosa-arena", "status", job))
    state = str(status.get("status", "")).lower()
    if state not in {"succeeded", "failed", "completed", "cancelled", "canceled"}:
        raise ValueError(f"Job {job} is {status.get('status')}; rerun report after it finishes")
    log = inside(container, "furiosa-arena", "logs", job) + "\n"
    package = args.directory / f"job-{job}.tar"
    if not package.exists():
        inside(container, "furiosa-arena", "download", job, "--out", remote + "/job-" + job + ".tar")
        run("docker", "cp", f"{container}:{remote}/job-{job}.tar", str(package))
    with tarfile.open(package) as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        if len(members) != 4 or {member.name for member in members} != {*built["payload"], "job.yaml"}:
            raise ValueError("Unexpected uploaded package contents")
        hashes = {member.name: fingerprint(archive.extractfile(member).read()) for member in members}
    for name, expected in built["payload"].items():
        if hashes[name] != expected:
            raise ValueError(f"Uploaded job contains a different {name}")
    log_path = args.directory / f"job-{job}.log"
    if log_path.exists():
        if read(log_path) != log.encode():
            raise ValueError("Terminal job log changed since its previous download")
    else:
        write(log_path, log.encode())
    result = {"recorded_at_utc": now(), "job_id": int(job), "status": status, "official_score": False,
              "run_ids": RUN_IDS, "candidate": manifest["candidate"], "package_files": hashes,
              "build_manifest_sha256": fingerprint(read(args.directory / "build.json"))["sha256"],
              "log": str(log_path), "log_sha256": fingerprint(log.encode())["sha256"]}
    try:
        if state not in {"succeeded", "completed"} or status.get("exit_code") != 0:
            raise ValueError("Arena job did not exit successfully")
        result.update(parse_log(log))
    except ValueError as error:
        result.update(correctness="failed_or_incomplete", reason=str(error))
    write_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return int(result["correctness"] != "passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in (("prepare", prepare), ("stage", stage), ("build", build), ("submit", submit), ("report", report)):
        command = commands.add_parser(name, help=function.__doc__)
        command.set_defaults(function=function)
        if name in {"prepare", "stage"}:
            command.add_argument("--label", required=True)
            command.add_argument("--container", default="moa2026-dev")
        else:
            command.add_argument("--directory", type=Path, required=True)
        if name == "stage":
            command.add_argument("--references", type=Path, required=True)
            command.add_argument("--snapshot", type=Path, required=True)
        if name == "report":
            command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        return args.function(args) or 0
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        print(f"extra_seeds: {error}", file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError) and error.output:
            print(error.output, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
