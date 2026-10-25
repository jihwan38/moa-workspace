#!/usr/bin/env python3
"""Build in the exclusively owned shared target, then copy this exact output."""
import os
import pathlib
import shutil
import subprocess

subprocess.run(["cargo", "furiosa-opt", "build", "--release", "--bin", "compare_b16_f4"], check=True)
source = pathlib.Path(os.environ["CARGO_TARGET_DIR"]) / "release/compare_b16_f4"
destination = pathlib.Path.cwd() / "target/release/compare_b16_f4_mse_identity"
destination.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(source, destination)
print("BUILD_IDENTITY_COPY", destination, flush=True)
