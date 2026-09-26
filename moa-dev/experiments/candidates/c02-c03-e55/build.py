#!/usr/bin/env python3
"""Build one frozen C02/C03 kernel package without modifying the working runtime."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARTIFACTS = ROOT / "moa-dev/experiments/artifacts/c02-c03-e55"
BASE = "873f64ed94229c88cd18d2731bdcca7eea0d6365"
CRATE = "furiosa-opt-gemma4-12B"
IMAGES = {
    "sliding_project_qkv": "5bdab7eafec5ac0aca352101deee334180b550c87db23e33c70d008892edba9e",
    "sliding_attention_output": "359473814839effc16ade242bad3c598a93610601bc650f550ff44228898d88b",
    "decoder_feedforward": "61ad0c50a03902de9208447951086df930669ded080c94e4c637bffac3c414b6",
}


def digest(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def run(args, log=None):
    if log:
        with log.open("w") as out:
            subprocess.run(args, stdout=out, stderr=subprocess.STDOUT, check=True)
    else:
        subprocess.run(args, check=True, stdout=subprocess.DEVNULL)


def build(variant, extra=False, label=None, edges=False, *, target_kernel=None,
          changed_sources=None, compile_only=False, expected_images=None,
          expected_source_inputs=None, dump_graph=False):
    label = label or variant + ("-edges" if edges else "-extra" if extra else "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", label):
        raise SystemExit("Use a simple alphanumeric/hyphen/underscore label")
    out = ARTIFACTS / label
    record_path = HERE / f"build-{label}-repro.json"
    if record_path.exists():
        raise SystemExit(f"Refusing to overwrite build record: {record_path}")
    out.mkdir(parents=True, exist_ok=False)
    crate = out / "crate"
    crate.mkdir(exist_ok=True)
    archive = subprocess.check_output(["git", "archive", BASE, CRATE], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        for member in source.getmembers():
            if member.isfile():
                path = crate / Path(member.name).relative_to(CRATE)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(source.extractfile(member).read())
    frozen = ROOT / "moa-dev/experiments/artifacts/e55-fused-qkv"
    lock = ROOT / "moa-dev/locks/gemma4-Cargo.lock"
    recorded_lock = json.loads((ROOT / "baseline/e55.json").read_text())
    recorded_lock = json.loads((ROOT / recorded_lock["compile_inputs"]).read_text())["Cargo.lock"]
    assert digest(lock.read_bytes()) == recorded_lock
    shutil.copyfile(lock, crate / "Cargo.lock")
    changed = changed_sources if changed_sources is not None else ({"projection.rs": "src/device/sliding/projection.rs", "rmsnorm.rs": "src/device/sliding/rmsnorm.rs", "rope.rs": "src/device/sliding/rope.rs", "ops.rs": "src/ops.rs"} if variant.startswith("c03") else {"mlp.rs": "src/device/shared/mlp.rs"})
    for name, target in changed.items():
        (crate / target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE / variant / name, crate / target)
    target_kernel = target_kernel or ("sliding_project_qkv" if variant.startswith("c03") else "decoder_feedforward")
    assert target_kernel in IMAGES
    # Historical callers retain their recorded image checks. New recipe-driven
    # callers explicitly choose checks for their source origin, independently
    # of whichever implementation is used as the performance comparison.
    image_checks = IMAGES if expected_images is None else expected_images
    if extra:
        sys.path.insert(0, str(ROOT / "moa-dev"))
        from extra_seeds import extended_harness
        host = crate / "src/bin/test_kernels.rs"
        host.write_bytes(extended_harness(host.read_bytes()))
    if edges:
        from edges import edge_harness
        host = crate / "src/bin/test_kernels.rs"
        host.write_bytes(edge_harness(host.read_bytes()))
    inputs = {str(p.relative_to(crate)): digest(p.read_bytes())
              for p in sorted(crate.rglob("*")) if p.is_file()
              and (p.is_relative_to(crate / "src") or p.name in
                   {"Cargo.toml", "Cargo.lock", "rust-toolchain.toml"})}
    if expected_source_inputs is not None:
        assert inputs == expected_source_inputs, "Assembled source differs from the selected recipe; stop before compilation"
    container = f"/tmp/moa-c02-c03/{label}"
    run(["docker", "exec", "moa2026-dev", "mkdir", "-p", container])
    run(["docker", "cp", str(crate) + "/.", f"moa2026-dev:{container}"])
    actual = {}
    archive = subprocess.check_output(["docker", "exec", "--workdir", container,
        "moa2026-dev", "tar", "-cf", "-", "src", "Cargo.toml", "Cargo.lock", "rust-toolchain.toml"])
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        for member in source.getmembers():
            if member.isfile():
                actual[member.name] = digest(source.extractfile(member).read())
    assert actual == inputs
    env = ["docker", "exec", "--workdir", container,
           "-e", "CARGO_HOME=/home/moa/.cargo", "-e", "CARGO_NET_OFFLINE=true",
           "-e", "CARGO_TARGET_DIR=/tmp/moa-c01-upgate/target",
           "-e", f"FURIOSA_OPT_OUT_DIR={container}/kernels", "moa2026-dev"]
    record = {"base_commit": BASE, "variant": variant, "label": label, "extra_inputs": extra, "edge_inputs": edges, "source_inputs": inputs,
              "source_inputs_verified_in_container": True, "adopted": False, "target_kernel": target_kernel, "changed_sources": list(changed.values())}
    try:
        print(f"{label}: compiling exact {target_kernel}", flush=True)
        compile_command = env + ["cargo", "furiosa-opt", "compile", "ops::" + target_kernel,
                                 "--exact", "--dump-schedule", "schedule.json"]
        if dump_graph:
            compile_command += ["--dump-graph", "graph.json"]
        record["compile_command"] = compile_command
        run(compile_command, out / "compile.log")
        record["exact_compile_exit_code"] = 0
        image_dir = container + "/kernels/furiosa-opt-gemma4"
        run(["docker", "cp", f"moa2026-dev:{container}/schedule.json", str(out / "schedule.json")])
        run(["docker", "cp", f"moa2026-dev:{image_dir}/furiosa_opt_gemma4::ops::{target_kernel}.bin",
             str(out / "exact-kernel.bin")])
        schedule = json.loads((out / "schedule.json").read_text())
        record["static_cycles"] = max(op["lifetime"]["end"] for op in schedule["instructions"])
        record["schedule"] = digest((out / "schedule.json").read_bytes())
        record["exact_image"] = digest((out / "exact-kernel.bin").read_bytes())
        if dump_graph:
            run(["docker", "cp", f"moa2026-dev:{container}/graph.json", str(out / "graph.json")])
            record["graph"] = digest((out / "graph.json").read_bytes())
        if compile_only:
            record["compile_only"] = True
            print(f"{label}: static {record['static_cycles']}; compile-only gate complete", flush=True)
            return
        print(f"{label}: static {record['static_cycles']}; building host", flush=True)
        run(env + ["cargo", "clean", "--release", "-p", "furiosa-opt-gemma4"])
        run(env + ["cargo", "furiosa-opt", "build", "--release", "--locked", "--bin", "test_kernels"],
            out / "build.log")
        record["host_build_exit_code"] = 0
        package = out / "package"
        package.mkdir(exist_ok=True)
        run(["docker", "cp", "moa2026-dev:/tmp/moa-c01-upgate/target/release/test_kernels",
             str(package / "test_runtime")])
        (out / "kernels").mkdir(exist_ok=True)
        run(["docker", "cp", f"moa2026-dev:{image_dir}/.", str(out / "kernels")])
        host = (package / "test_runtime").read_bytes()
        record["images"] = {}
        record["unchanged_image_checks"] = {}
        for name in IMAGES:
            data = (out / "kernels" / f"furiosa_opt_gemma4::ops::{name}.bin").read_bytes()
            record["images"][name] = dict(digest(data), host_occurrences=host.count(data))
            assert host.count(data) == 1
            if name == target_kernel:
                assert data == (out / "exact-kernel.bin").read_bytes()
            elif name in image_checks:
                assert digest(data)["sha256"] == image_checks[name]
                record["unchanged_image_checks"][name] = image_checks[name]
        fixture = (ROOT / "moa-dev/experiments/artifacts/extra-seeds-run3-8/crate/ref/fixtures.safetensors"
                   if extra else frozen / "ref/fixtures.safetensors")
        fixture_hash = ("f85f6e03d8eb359cc8589d375f89ba7b501924b095059800e322378e017cb4df" if extra else
                        "338c0a8c06458a528b29d58f0f65a5ea13af79a14a9929a06c571b8adb002359")
        if edges:
            fixture = ARTIFACTS / "edges/fixtures.safetensors"
            record["edge_reference"] = json.loads((ARTIFACTS / "edges/fixtures.json").read_text())
            fixture_hash = record["edge_reference"]["fixture"]["sha256"]
        assert digest(fixture.read_bytes())["sha256"] == fixture_hash
        shutil.copyfile(fixture, package / "fixtures.safetensors")
        runner = crate / "scripts/rngd/remote_entrypoint.sh"
        assert digest(runner.read_bytes())["sha256"] == "6ea0fa3b0b224f85c33b4a8e893adad77421c3f228646520d83cf9b85bd26b9d"
        shutil.copyfile(runner, package / runner.name)
        for name in ("test_runtime", "remote_entrypoint.sh"):
            (package / name).chmod(0o755)
        record["package_files"] = {p.name: digest(p.read_bytes()) for p in sorted(package.iterdir())}
        run(["docker", "cp", str(package), f"moa2026-dev:{container}/package"])
        record["container_package"] = container + "/package"
        print(f"{label}: ready {record['container_package']}", flush=True)
    except subprocess.CalledProcessError as exc:
        record["failed_command"] = exc.cmd
        record["failed_exit_code"] = exc.returncode
        raise
    finally:
        record["logs"] = {p.name: digest(p.read_bytes()) for p in (out / "compile.log", out / "build.log") if p.exists()}
        record_path.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=["c02-upgate", "c02-down", "c02-combined", "c03-cluster-heads"])
    diagnostic = parser.add_mutually_exclusive_group()
    diagnostic.add_argument("--extra", action="store_true", help="Use six extra inputs, runs 3..8")
    diagnostic.add_argument("--edges", action="store_true", help="Six inputs with RoPE/cache boundary positions and nonzero cache sentinels")
    parser.add_argument("--label", help="New build label; existing records are never overwritten")
    args = parser.parse_args()
    build(args.variant, args.extra, args.label, args.edges)
