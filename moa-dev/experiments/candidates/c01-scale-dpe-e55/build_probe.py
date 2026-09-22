#!/usr/bin/env python3
"""Rebuild a D01 diagnostic in a new scratch directory; never submits a job."""
import argparse
import json
import re
import shutil
import subprocess

from build import ARTIFACTS, BASE, CRATE, HERE, ROOT, digest, run


def build_probe(variant, label):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", label):
        raise SystemExit("Use a simple alphanumeric/hyphen/underscore label")
    out = ARTIFACTS / label
    out.mkdir()  # Never reuse an earlier executable after a failed build.
    crate = out / "crate"
    (crate / "src/bin").mkdir(parents=True)
    for name in ("Cargo.toml", "rust-toolchain.toml"):
        (crate / name).write_bytes(subprocess.check_output(
            ["git", "show", f"{BASE}:{CRATE}/{name}"], cwd=ROOT))
    shutil.copyfile(ROOT / "moa-dev/locks/gemma4-Cargo.lock", crate / "Cargo.lock")
    suffix = {"fixed": "", "original": ".original", "clear16": ".clear16"}[variant]
    for source, target in (("probe_lib.rs", "src/lib.rs"),
                           (f"scale_probe{suffix}.rs", "src/scale_probe.rs"),
                           (f"probe_scale{suffix}.rs", "src/bin/probe_scale.rs")):
        shutil.copyfile(HERE / source, crate / target)
    inputs = {str(p.relative_to(crate)): digest(p.read_bytes())
              for p in sorted(crate.rglob("*")) if p.is_file()}
    container = f"/tmp/moa-c01-probe-repro/{label}"
    run(["docker", "exec", "moa2026-dev", "mkdir", "-p", container])
    run(["docker", "cp", str(crate) + "/.", f"moa2026-dev:{container}"])
    env = ["docker", "exec", "--workdir", container,
           "-e", "CARGO_HOME=/home/moa/.cargo", "-e", "CARGO_NET_OFFLINE=true",
           "-e", "CARGO_TARGET_DIR=/tmp/moa-c01-probe/target",
           "-e", f"FURIOSA_OPT_OUT_DIR={container}/kernels", "moa2026-dev"]
    # Sequential only: this target is shared by D01 rebuilds.
    run(env + ["cargo", "clean", "--release", "-p", "furiosa-opt-gemma4"])
    run(env + ["cargo", "furiosa-opt", "build", "--release", "--locked", "--bin", "probe_scale"], out / "build.log")
    package = out / "package"
    package.mkdir()
    run(["docker", "cp", "moa2026-dev:/tmp/moa-c01-probe/target/release/probe_scale", str(package / "test_runtime")])
    run(["docker", "cp", f"moa2026-dev:{container}/kernels/furiosa-opt-gemma4/furiosa_opt_gemma4::scale_probe::scale_probe.bin",
         str(out / "exact.bin")])
    image = (out / "exact.bin").read_bytes()
    host = (package / "test_runtime").read_bytes()
    assert host.count(image) == 1
    shutil.copyfile(HERE / "probe-entrypoint.sh", package / "remote_entrypoint.sh")
    for p in package.iterdir():
        p.chmod(0o755)
    record = {"variant": variant, "source_inputs": inputs, "image": dict(digest(image), host_occurrences=1),
              "package_files": {p.name: digest(p.read_bytes()) for p in sorted(package.iterdir())}}
    (out / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    if variant == "fixed":
        previous = json.loads((HERE / "build-probe-clear8.json").read_text())
        assert record["source_inputs"] == previous["source_inputs"]
        assert record["image"] == previous["image"]
        assert record["package_files"] == previous["package_files"]
    run(["docker", "cp", str(package), f"moa2026-dev:{container}/package"])
    print(f"Ready: {container}/package; sources/image/package verified", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=["fixed", "original", "clear16"])
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    build_probe(args.variant, args.label)
