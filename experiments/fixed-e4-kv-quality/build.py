#!/usr/bin/env python3
"""Fresh-package wrapper captures these copied artifacts after source verification."""
import os
from pathlib import Path
import shutil
import subprocess
names = ['test_fixed_e4_head_codec', 'compare_fixed_e4_kv', 'compare_fixed_e4_packed']
command = ['cargo', 'furiosa-opt', 'build', '--release']
for name in names:
    command += ['--bin', name]
subprocess.run(command, check=True)
for name in names:
    source = Path(os.environ['CARGO_TARGET_DIR']) / 'release' / name
    target = Path.cwd() / 'target/release' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    print('COPIED_EXACT_BUILD', target, flush=True)
