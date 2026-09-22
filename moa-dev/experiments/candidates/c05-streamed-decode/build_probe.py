#!/usr/bin/env python3
"""Build the projection diagnostic from the exact partials candidate source."""
import io
import json
import re
import subprocess
import sys
import tarfile
from build import HERE, common

ROOT, ARTIFACTS, digest, run = common.ROOT, common.ARTIFACTS, common.digest, common.run


if __name__ == "__main__":
    label = sys.argv[1]
    assert re.fullmatch(r"projection-probe[a-z0-9-]*", label)
    out = ARTIFACTS / label
    out.mkdir(parents=True, exist_ok=False)
    crate = out / "crate"
    mapping = {
        "src/lib.rs": HERE / "probe_lib.rs",
        "src/stream_probe.rs": HERE / "stream_probe.rs",
        "src/bin/probe_stream.rs": HERE / "probe_stream.rs",
        "src/device/mlp.rs": HERE / "partials/mlp.rs",
        **{f: ARTIFACTS / "c05-partials-transpose2/crate" / f for f in
           ("src/axes.rs", "src/device/layout.rs", "Cargo.toml", "Cargo.lock", "rust-toolchain.toml")},
    }
    for name, path in mapping.items():
        target = crate / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    inputs = {name: digest(path.read_bytes()) for name, path in mapping.items()}
    remote = "/tmp/moa-c05/" + label
    run(["docker", "exec", "moa2026-dev", "mkdir", "-p", remote])
    run(["docker", "cp", str(crate) + "/.", "moa2026-dev:" + remote])
    archive = subprocess.check_output(["docker", "exec", "--workdir", remote, "moa2026-dev", "tar", "-cf", "-", *mapping])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        assert {m.name: digest(tar.extractfile(m).read()) for m in tar if m.isfile()} == inputs
    env = ["docker", "exec", "--workdir", remote, "-e", "CARGO_NET_OFFLINE=true",
           "-e", "CARGO_TARGET_DIR=/tmp/moa-c01-probe/target", "-e", f"FURIOSA_OPT_OUT_DIR={remote}/kernels", "moa2026-dev"]
    record = {"source_inputs": inputs, "source_inputs_verified_in_container": True,
              "candidate_build": "build-c05-partials-transpose2-repro.json", "scope": "Projection correctness only; separate diagnostic images, no performance claims."}
    try:
        run(env + ["cargo", "clean", "--release", "-p", "furiosa-opt-gemma4"])
        run(env + ["cargo", "furiosa-opt", "build", "--release", "--locked", "--bin", "probe_stream"], out / "build.log")
        record["build_exit_code"] = 0
        package = out / "package"
        package.mkdir()
        run(["docker", "cp", "moa2026-dev:/tmp/moa-c01-probe/target/release/probe_stream", str(package / "probe_stream")])
        (package / "run.sh").write_text('#!/bin/sh\nset -eu\ncd "$(dirname "$0")"\nchmod +x probe_stream\nexec ./probe_stream\n')
        for p in package.iterdir():
            p.chmod(0o755)
        run(["docker", "cp", "moa2026-dev:" + remote + "/kernels", str(out / "kernels")])
        host = (package / "probe_stream").read_bytes()
        record["images"] = {}
        for path in (out / "kernels").rglob("*.bin"):
            data = path.read_bytes()
            assert host.count(data) == 1
            record["images"][path.name] = dict(digest(data), host_occurrences=1)
        assert len(record["images"]) == 2
        run(["docker", "cp", str(package), "moa2026-dev:" + remote + "/package"])
        record["package_files"] = {p.name: digest(p.read_bytes()) for p in package.iterdir()}
        print("NPU diagnostic package ready", flush=True)
        run(["docker", "exec", "moa2026-dev", remote + "/package/probe_stream", "--oracle-only"], out / "oracle.log")
        record["oracle_exit_code"] = 0
        print((out / "oracle.log").read_text(), flush=True)
    except subprocess.CalledProcessError as error:
        record.update(failed_exit_code=error.returncode, failed_command=error.cmd)
        raise
    finally:
        record["logs"] = {p.name: digest(p.read_bytes()) for p in (out / "build.log", out / "oracle.log") if p.exists()}
        with (HERE / f"build-{label}.json").open("x") as f:
            json.dump(record, f, indent=2)
            f.write("\n")
