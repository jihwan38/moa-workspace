#!/usr/bin/env python3
"""Capture source and existing artifacts without building, submitting, or starting Docker.

Example, after the caller has completed a build:
    python3 moa-dev/capture.py --label candidate-01 \
        --output docs/experiments/candidate-01-provenance.json

The resulting JSON is accepted by measure.py --provenance. A snapshot records
bytes observed now; it does not prove that the binary was built from this source.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CRATE = ROOT / "furiosa-opt-gemma4-12B"
ARTIFACTS = ROOT / "moa-dev/experiments/artifacts"


def run(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.PIPE, timeout=30)


def git(*args):
    # Even status must not refresh/write the live index.
    return run("git", "--no-optional-locks", "-C", str(ROOT), *args)


def fingerprint(content):
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def read_stable(path):
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"Expected a regular file, not a symlink or special file: {path}")
    content = path.read_bytes()
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f"File changed while being captured: {path}")
    return content, before


def container_environment(container):
    info = json.loads(run("docker", "inspect", container))[0]
    if not info["State"]["Running"]:
        raise ValueError(f"Container {container} is not running; capture does not start containers")
    mounts = [m for m in info["Mounts"] if m["Destination"] == "/workspace"]
    if len(mounts) != 1 or Path(mounts[0]["Source"]).resolve() != CRATE.resolve():
        raise ValueError("Container /workspace is not the expected competition checkout")

    def inside(*args):
        return run("docker", "exec", "--workdir", "/workspace", container, *args).strip()

    compiler_metadata = json.loads(inside("cat", "/home/moa/.cargo/.crates2.json"))["installs"]
    compiler = next(({"package": key.split()[0], "version": key.split()[1]}
                     for key in compiler_metadata if key.startswith("cargo-furiosa-opt ")), None)
    if compiler is None:
        raise ValueError("No installed cargo-furiosa-opt version found")
    hashes = inside("sha256sum", "/home/moa/.cargo/bin/cargo-furiosa-opt",
                    "/home/moa/.cargo/bin/furiosa-opt-driver")
    compiler["binary_sha256"] = {line.split(maxsplit=1)[1]: line.split()[0]
                                 for line in hashes.splitlines()}
    return {"container": container, "container_id": info["Id"], "image_id": info["Image"],
            "source_mount": {"host": mounts[0]["Source"], "container": "/workspace"},
            "rustc": inside("rustc", "-Vv"), "cargo": inside("cargo", "-V"),
            "compiler": compiler}


def capture(label, output, binary, container):
    destination = ARTIFACTS / label
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"Refusing to overwrite snapshot label: {label}")
    if output.exists() or output.is_symlink():
        raise ValueError(f"Refusing to overwrite provenance: {output}")
    if destination.resolve() in output.resolve().parents or output.resolve() == destination.resolve():
        raise ValueError("Choose a provenance output outside the artifact snapshot directory")
    if not binary.is_file():
        raise ValueError(f"Build artifact is missing: {binary}; capture never builds it")
    if Path(git("rev-parse", "--show-toplevel").strip()).resolve() != ROOT.resolve():
        raise ValueError("Expected the workspace itself to be the Git repository root")

    started = datetime.now(timezone.utc).isoformat()
    head = git("rev-parse", "HEAD").strip()
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    patches = {
        "git/working-tree.patch": git("diff", "--binary", "--no-ext-diff", "--no-textconv", "HEAD", "--").encode(),
        "git/index.patch": git("diff", "--cached", "--binary", "--no-ext-diff", "--no-textconv", "--").encode(),
        "git/status.txt": status.encode(),
    }
    environment = container_environment(container)
    sources = sorted(path for path in (CRATE / "src").rglob("*") if path.is_file() or path.is_symlink())
    if not (CRATE / "src/ops.rs").is_file() or not (CRATE / "src/device").is_dir():
        raise ValueError("Required src/ops.rs or src/device tree is missing")
    paths = {path.relative_to(CRATE).as_posix(): path for path in sources}
    for relative in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "ref/fixtures.safetensors",
                     "scripts/rngd/remote_entrypoint.sh", "scripts/rngd_test.sh"):
        paths[relative] = CRATE / relative
    paths["target/release/test_kernels"] = binary

    payload, files = {}, {}
    for relative, path in paths.items():
        content, details = read_stable(path)
        payload[relative] = content
        files[relative] = {**fingerprint(content), "captured_from": str(path),
                           "observed_mtime_ns": details.st_mtime_ns,
                           "snapshot_path": str((destination / relative).relative_to(ROOT))}

    # Reject source/build races rather than silently combining versions. This does
    # not establish which source was used by a completed earlier build.
    if set(sources) != {p for p in (CRATE / "src").rglob("*") if p.is_file() or p.is_symlink()}:
        raise ValueError("Source file set changed during capture; retry after edits/builds finish")
    for relative, path in paths.items():
        content, _ = read_stable(path)
        if fingerprint(content) != fingerprint(payload[relative]):
            raise ValueError(f"Source or artifact changed during capture: {relative}")
    if git("rev-parse", "HEAD").strip() != head:
        raise ValueError("Git HEAD changed during capture")

    record = {
        "schema_version": 1, "label": label, "recorded_at_utc": started,
        "source_commit": head,
        "source_binary_relationship": {
            "verified": False,
            "note": "The caller must finish the build before capture. Existing binary and current source were copied independently; timestamps do not prove they match.",
        },
        "git": {"root": str(ROOT), "head": head, "dirty_at_start": bool(status),
                "status_porcelain_at_start": status,
                "patches": {name: {**fingerprint(content),
                                    "snapshot_path": str((destination / name).relative_to(ROOT))}
                            for name, content in patches.items()},
                "note": "Patches/status describe tracked Git state at capture start. Untracked src files are included in the source snapshot; other untracked file contents are not copied."},
        "environment": environment,
        "cargo_lock": files["Cargo.lock"],
        "allowed_source_files": {name: info for name, info in files.items()
                                 if name == "src/ops.rs" or name.startswith("src/device/")},
        "source_files": {name: info for name, info in files.items() if name.startswith("src/")},
        # These exact keys are measure.py's provenance contract.
        "artifacts": {name: files[name] for name in ("target/release/test_kernels", "ref/fixtures.safetensors")},
        "snapshot": {"directory": str(destination.relative_to(ROOT)), "files": files,
                     "protection": "No overwrite through this helper; copied files/directories are made read-only. This is filesystem permission protection, not an immutable storage service."},
        "capture_tool": fingerprint(Path(__file__).read_bytes()),
    }
    record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    encoded = (json.dumps(record, indent=2, ensure_ascii=False) + "\n").encode()

    # Exclusive creation protects labels/outputs even if two callers race.
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    created_output = False
    try:
        for relative, content in {**payload, **patches, "provenance.json": encoded}.items():
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(content)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            created_output = True
            stream.write(encoded)
        for path in destination.rglob("*"):
            if path.is_file():
                path.chmod(0o555 if path.name in {"test_kernels", "remote_entrypoint.sh", "rngd_test.sh"} else 0o444)
        for path in sorted((p for p in destination.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            path.chmod(0o555)
        destination.chmod(0o555)
    except Exception:
        # Only remove the exclusive directory/files this call created.
        for path in [destination, *(p for p in destination.rglob("*") if p.is_dir())]:
            path.chmod(0o755)
        shutil.rmtree(destination)
        if created_output:
            output.unlink()
        raise
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=CRATE / "target/release/test_kernels",
                        help="Existing binary to preserve; default is the live build output")
    parser.add_argument("--container", default="moa2026-dev")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", args.label):
        parser.error("Use an alphanumeric label with optional hyphens/underscores (1–80 characters)")
    try:
        record = capture(args.label, args.output.absolute(), args.binary.absolute(), args.container)
    except (ValueError, OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"capture: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(args.output), "snapshot": record["snapshot"]["directory"],
                      "binary_sha256": record["artifacts"]["target/release/test_kernels"]["sha256"],
                      "source_binary_match_verified": False}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
