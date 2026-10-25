#!/usr/bin/env python3
"""Build in our shared target and copy immediately for record_run capture."""
import os
import pathlib
import shutil
import subprocess
subprocess.run(["cargo", "furiosa-opt", "build", "--release", "--bin", "server"], check=True)
source = pathlib.Path(os.environ["CARGO_TARGET_DIR"]) / "release/server"
destination = pathlib.Path.cwd() / "target/release/server"
destination.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(source, destination)
print("BUILD_SERVER_COPY", destination, flush=True)
