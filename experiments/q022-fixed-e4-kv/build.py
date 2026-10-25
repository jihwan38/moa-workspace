#!/usr/bin/env python3
import os
from pathlib import Path
import shutil
import subprocess
out = Path(os.environ['FURIOSA_OPT_OUT_DIR'])
assert 'furiosa-kernels-q022' in str(out), 'q022 requires an isolated public SDK kernel directory'
subprocess.run(['cargo','furiosa-opt','build','--release','--bin','server'],check=True)
src=Path(os.environ['CARGO_TARGET_DIR'])/'release/server'
dst=Path.cwd()/'target/release/server'
dst.parent.mkdir(parents=True,exist_ok=True)
shutil.copy2(src,dst)
print('CAPTURE_READY',dst,flush=True)
