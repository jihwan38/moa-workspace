#!/usr/bin/env python3
"""Build a benchmark host only after corrected descriptor verification passes."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
from build import HERE, ROOT, digest
from analyze_corrected import result

ARTIFACTS = ROOT / "moa-dev/experiments/artifacts/c08-joint-ingress"
common = SimpleNamespace(ROOT=ROOT, ARTIFACTS=ARTIFACTS, digest=digest)


def package(label):
    gate = result(label)
    assert gate == json.loads((HERE / f"descriptor-analysis-{label}.json").read_text())
    original = json.loads((HERE / f"build-{label}.json").read_text())
    path = HERE / f"build-{label}-repro.json"
    assert not path.exists(), "Never overwrite a host build"
    out = ARTIFACTS / label
    remote = "/tmp/moa-c08-ingress/" + label
    env = ["docker", "exec", "--workdir", remote, "-e", "CARGO_HOME=/home/moa/.cargo",
           "-e", "CARGO_NET_OFFLINE=true", "-e", "CARGO_TARGET_DIR=/tmp/moa-c01-upgate/target",
           "-e", f"FURIOSA_OPT_OUT_DIR={remote}/kernels", "moa2026-dev"]
    record = {**original, "extra_inputs": False, "adopted": False,
              "source_gate": str((HERE / f"descriptor-analysis-{label}.json").relative_to(ROOT)),
              "package_builder": digest(Path(__file__).read_bytes()), "images": {}, "commands": []}
    log = HERE / "evidence" / label / "host-build.log"
    try:
        with log.open("w") as stream:
            for command in [env + ["cargo", "clean", "--release", "-p", "furiosa-opt-gemma4"],
                            env + ["cargo", "furiosa-opt", "build", "--release", "--locked", "--bin", "test_kernels"]]:
                record["commands"].append(command)
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        record["host_build_exit_code"] = 0
        package = out / "package"
        package.mkdir(exist_ok=False)
        subprocess.run(["docker", "cp", "moa2026-dev:/tmp/moa-c01-upgate/target/release/test_kernels", str(package / "test_runtime")], check=True)
        kernels = out / "kernels"
        kernels.mkdir(exist_ok=False)
        subprocess.run(["docker", "cp", "moa2026-dev:" + remote + "/kernels/furiosa-opt-gemma4/.", str(kernels)], check=True)
        host = (package / "test_runtime").read_bytes()
        control = json.loads((HERE.parent / "c07-native-scale-dpe/build-c07-e104-reviewed-v1-repro.json").read_text())
        for kernel in ("sliding_project_qkv", "sliding_attention_output", "decoder_feedforward"):
            raw = (kernels / f"furiosa_opt_gemma4::ops::{kernel}.bin").read_bytes()
            assert host.count(raw) == 1
            expected = original["image"] if kernel == "sliding_project_qkv" else {key: control["images"][kernel][key] for key in ("bytes", "sha256")}
            assert digest(raw) == expected, (kernel, "exact or E104 identity changed")
            record["images"][kernel] = {**digest(raw), "host_occurrences": 1}
        reference_package = ROOT / "moa-dev/experiments/artifacts/c07-native-scale-dpe/c07-e104-reviewed-v1/package"
        for name in ("fixtures.safetensors", "remote_entrypoint.sh"):
            raw = (reference_package / name).read_bytes()
            assert digest(raw) == control["package_files"][name]
            (package / name).write_bytes(raw)
        (package / "test_runtime").chmod(0o755)
        (package / "remote_entrypoint.sh").chmod(0o755)
        record["package_files"] = {file.name: digest(file.read_bytes()) for file in sorted(package.iterdir())}
        record["unchanged_e104_o_ffn_images"] = True
        print("Corrected host and E104 O/FFN image identities verified", flush=True)
    except subprocess.CalledProcessError as error:
        record["host_build_exit_code"] = error.returncode
        raise
    finally:
        record["host_build_log"] = digest(log.read_bytes())
        path.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label")
    package(parser.parse_args().label)
