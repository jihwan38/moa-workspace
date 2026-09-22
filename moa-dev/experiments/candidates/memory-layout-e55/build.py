#!/usr/bin/env python3
"""Compile isolated layout variants and fingerprint their actual host packages."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from prepare import ARTIFACTS, HERE, ROOT, VARIANTS, kernel_name


def fingerprint(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def run(args, log=None):
    if log:
        with log.open("w") as output:
            return subprocess.run(args, stdout=output, stderr=subprocess.STDOUT).returncode
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL)


def build(variant):
    out = ARTIFACTS / variant
    result_dir = HERE / variant
    container = f"/tmp/moa-memory-layout-e55/{variant}"
    work = container + "/crate"
    env = ["docker", "exec", "--workdir", work,
           "-e", "CARGO_HOME=/home/moa/.cargo", "-e", "CARGO_NET_OFFLINE=true",
           "-e", "CARGO_TARGET_DIR=/tmp/moa-memory-layout-e55/target",
           "-e", f"FURIOSA_OPT_OUT_DIR={container}/kernels", "moa2026-dev"]
    run(["docker", "exec", "moa2026-dev", "mkdir", "-p", work])
    run(["docker", "cp", str(out / "crate") + "/.", f"moa2026-dev:{work}"])
    actual = {}
    archive = subprocess.check_output(["docker", "exec", "--workdir", work, "moa2026-dev", "tar", "-cf", "-", "src", "Cargo.toml", "Cargo.lock", "rust-toolchain.toml"])
    with tarfile.open(fileobj=io.BytesIO(archive)) as source_archive:
        for member in source_archive.getmembers():
            if member.isfile():
                data = source_archive.extractfile(member).read()
                actual[member.name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    assert actual == json.loads((result_dir / "source-manifest.json").read_text())
    record = {"variant": variant, "base_tag": "baseline-e55", "compiler_inputs_match": True,
              "compiler_input_count": len(actual), "adopted": False, "hardware_correctness": None}
    print(f"{variant}: exact compile", flush=True)
    target_kernel = kernel_name(variant)
    record["target_kernel"] = target_kernel
    record["exact_compile_exit_code"] = run(env + ["cargo", "furiosa-opt", "compile", "ops::" + target_kernel, "--exact", "--dump-schedule", "candidate.json", "--dump-graph", "candidate-graph.json"], out / "compile.log")
    record["compile_log"] = fingerprint(out / "compile.log")
    if record["exact_compile_exit_code"]:
        (result_dir / "build-evidence.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"{variant}: compile failed; see compile.log", flush=True)
        return
    for name in ("candidate.json", "candidate-graph.json"):
        run(["docker", "cp", f"moa2026-dev:{work}/{name}", str(out / name)])
        record[name] = fingerprint(out / name)
    image_dir = f"{container}/kernels/furiosa-opt-gemma4"
    exact_name = "exact-ffn.bin" if target_kernel == "decoder_feedforward" else "exact-kernel.bin"
    run(["docker", "cp", f"moa2026-dev:{image_dir}/furiosa_opt_gemma4::ops::{target_kernel}.bin", str(out / exact_name)])
    data = json.loads((out / "candidate.json").read_text())
    record["static_cycles"] = max(op["lifetime"]["end"] for op in data["instructions"])
    print(f"{variant}: static {record['static_cycles']}; host build", flush=True)
    # Cargo's shared target can consider an identical package from another
    # scratch path fresh. Remove this package's release units (not SDK deps)
    # so both the host and its embedded images are built from this source.
    run(env + ["cargo", "clean", "--release", "-p", "furiosa-opt-gemma4"])
    record["host_build_exit_code"] = run(env + ["cargo", "furiosa-opt", "build", "--release", "--bin", "test_kernels"], out / "build.log")
    record["build_log"] = fingerprint(out / "build.log")
    if record["host_build_exit_code"]:
        (result_dir / "build-evidence.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"{variant}: host build failed", flush=True)
        return
    package = out / "package"
    package.mkdir(exist_ok=True)
    run(["docker", "cp", "moa2026-dev:/tmp/moa-memory-layout-e55/target/release/test_kernels", str(package / "test_runtime")])
    (out / "compiled-kernels").mkdir(exist_ok=True)
    run(["docker", "cp", f"moa2026-dev:{image_dir}/.", str(out / "compiled-kernels")])
    expected = {"sliding_project_qkv": "5bdab7eafec5ac0aca352101deee334180b550c87db23e33c70d008892edba9e",
                "sliding_attention_output": "359473814839effc16ade242bad3c598a93610601bc650f550ff44228898d88b",
                "decoder_feedforward": "61ad0c50a03902de9208447951086df930669ded080c94e4c637bffac3c414b6"}
    host = (package / "test_runtime").read_bytes()
    record["images"] = {}
    for name in expected:
        path = out / "compiled-kernels" / f"furiosa_opt_gemma4::ops::{name}.bin"
        image = fingerprint(path)
        image["host_occurrences"] = host.count(path.read_bytes())
        assert image["host_occurrences"] == 1
        if name != target_kernel:
            assert image["sha256"] == expected[name]
        else:
            assert path.read_bytes() == (out / exact_name).read_bytes()
        record["images"][name] = image
    frozen = ROOT / "moa-dev/experiments/artifacts/e55-fused-qkv"
    for src, name, sha in [
        (frozen / "ref/fixtures.safetensors", "fixtures.safetensors", "338c0a8c06458a528b29d58f0f65a5ea13af79a14a9929a06c571b8adb002359"),
        (frozen / "scripts/rngd/remote_entrypoint.sh", "remote_entrypoint.sh", "6ea0fa3b0b224f85c33b4a8e893adad77421c3f228646520d83cf9b85bd26b9d"),
    ]:
        assert fingerprint(src)["sha256"] == sha
        shutil.copyfile(src, package / name)
    for name in ("test_runtime", "remote_entrypoint.sh"):
        (package / name).chmod(0o755)
    record["package_files"] = {p.name: fingerprint(p) for p in sorted(package.iterdir())}
    (result_dir / "build-evidence.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"{variant}: ready", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("variants", choices=VARIANTS, nargs="+")
    args = parser.parse_args()
    for variant in args.variants:
        build(variant)
