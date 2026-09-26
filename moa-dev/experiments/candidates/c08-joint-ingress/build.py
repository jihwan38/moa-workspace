#!/usr/bin/env python3
"""Recompile frozen E105 once, retaining descriptors without a host/NPU build."""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def digest(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def build(label):
    record_path = HERE / f"build-{label}.json"
    assert not record_path.exists(), "Never overwrite a prior build"
    spec = importlib.util.spec_from_file_location("planner", ROOT / "moa-dev/experiments/structural_build.py")
    planner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(planner)
    _, config, plan = planner.resolve_plan(HERE.name, "frozen-v2")
    old_path = ROOT / config["frozen_reference_build"]
    old = json.loads(old_path.read_text())
    out = ROOT / "moa-dev/experiments/artifacts/c08-joint-ingress" / label
    out.mkdir(parents=True, exist_ok=False)
    crate = out / "crate"
    archive = subprocess.check_output(["git", "archive", plan["source_base_commit"], "furiosa-opt-gemma4-12B"], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar:
            if member.isfile():
                target = crate / Path(member.name).relative_to("furiosa-opt-gemma4-12B")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tar.extractfile(member).read())
    shutil.copyfile(ROOT / "moa-dev/locks/gemma4-Cargo.lock", crate / "Cargo.lock")
    for target, override in plan["source_overrides"].items():
        shutil.copyfile(ROOT / override["path"], crate / target)
    inputs = {name: digest((crate / name).read_bytes()) for name in plan["source_origin_inputs"]}
    assert inputs == old["source_inputs"], "Frozen E105 input identity must hold before compile"
    remote = "/tmp/moa-c08-ingress/" + label
    subprocess.run(["docker", "exec", "moa2026-dev", "mkdir", "-p", remote], check=True)
    subprocess.run(["docker", "cp", str(crate) + "/.", "moa2026-dev:" + remote], check=True)
    copied = subprocess.check_output(["docker", "exec", "--workdir", remote, "moa2026-dev", "tar", "-cf", "-", *inputs])
    with tarfile.open(fileobj=io.BytesIO(copied)) as tar:
        assert {m.name: digest(tar.extractfile(m).read()) for m in tar if m.isfile()} == inputs
    evidence = HERE / "evidence" / label
    evidence.mkdir(parents=True, exist_ok=False)
    command = ["docker", "exec", "--workdir", remote, "-e", "CARGO_HOME=/home/moa/.cargo",
               "-e", "CARGO_NET_OFFLINE=true", "-e", "CARGO_TARGET_DIR=/tmp/moa-c01-upgate/target",
               "-e", f"FURIOSA_OPT_OUT_DIR={remote}/kernels", "moa2026-dev", "cargo", "furiosa-opt", "compile",
               "ops::sliding_project_qkv", "--exact", "--dump-schedule", "schedule.json", "--dump-graph", "graph.json",
               "--dump-ir", "kernel.ir", "--dump-visa", "kernel.visa"]
    record = {"label": label, "source_plan": plan, "source_inputs": inputs,
              "source_inputs_match_frozen_e105": True, "source_inputs_verified_in_container": True,
              "frozen_reference_build": {"path": str(old_path.relative_to(ROOT)), **digest(old_path.read_bytes())},
              "builder": digest(Path(__file__).read_bytes()), "command": command, "dumps": {},
              "scope": "Exact compile only; no host build, Arena execution, performance claim or production source changes"}
    log = evidence / "compile.log"
    try:
        with log.open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        record["exact_compile_exit_code"] = result.returncode
        assert result.returncode == 0, f"See {log}"
        for name in ("schedule.json", "graph.json", "kernel.ir", "kernel.visa"):
            subprocess.run(["docker", "cp", "moa2026-dev:" + remote + "/" + name, str(evidence / name)], check=True)
            record["dumps"][name] = digest((evidence / name).read_bytes())
        image_path = out / "exact-kernel.bin"
        subprocess.run(["docker", "cp", "moa2026-dev:" + remote + "/kernels/furiosa-opt-gemma4/furiosa_opt_gemma4::ops::sliding_project_qkv.bin", str(image_path)], check=True)
        old_image = ROOT / "moa-dev/experiments/artifacts/c08-fused-qkv/c08-fused-contiguous-v2/exact-kernel.bin"
        expected = {key: old["images"]["sliding_project_qkv"][key] for key in ("bytes", "sha256")}
        assert digest(old_image.read_bytes()) == expected
        record["image"] = digest(image_path.read_bytes())
        record["image_byte_identical_to_e105"] = image_path.read_bytes() == old_image.read_bytes()
        assert record["image_byte_identical_to_e105"], "Do not attribute changed-image descriptors to E105"
        record["compiler_makespan"] = max(n["lifetime"]["end"] for n in json.loads((evidence / "schedule.json").read_text())["instructions"])
        print("Exact E105 image reproduced; compiler makespan", record["compiler_makespan"], flush=True)
    finally:
        if log.exists():
            record["compile_log"] = digest(log.read_bytes())
        record_path.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label")
    build(parser.parse_args().label)
