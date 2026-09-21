#!/usr/bin/env python3
"""Fuse only ring32 broadcast into the existing shared BF16 TRF load."""
from pathlib import Path
import hashlib
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
BASE=ROOT/'moa-dev/experiments/artifacts/qkv-bf16-shared-trf-e55/crate'
CRATE=ROOT/'moa-dev/experiments/artifacts/qkv-bf16-broadcast-trf-e55/crate'

def once(s,a,b):
    assert s.count(a)==1,(s.count(a),a)
    return s.replace(a,b)

original=(BASE/'src/device/sliding/projection.rs').read_text()
assert hashlib.sha256(original.encode()).hexdigest()=='26b7ada776900a82ccc716deaa26e1f6d7ed55c4e86fce9f1098132e1d766fb5'
projection=once(original,'use crate::axes::{Ds, Dummy2, Gs, H, Ns, Ps, Qs};',
    'use crate::axes::{Ds, Dummy2, Dummy256, Gs, H, Ns, Ps, Qs};')
projection=once(projection,
    '// Both physical clusters and all 256 slices hold the same real BF16 input.\n// Named neutral axes let Q/K/V borrow one TRF without a TRF reshape API.',
    '// Both clusters have a real full-H BF16 vector at every ring32 leader.\n// Sub broadcasts those initialized leaders directly into one shared TRF.')
projection=once(projection,
    '    x: &DmTensor<bf16, Chip, m![2], Replicated, m![H]>,',
    '    x: &DmTensor<bf16, Chip, m![2], m![Dummy256 / 32, 1 # 32], m![H]>,')
projection=once(projection,
    "    let x: DmTensorView<'_, bf16, Chip, InputClusters, Replicated, m![H]> = unsafe { x.view().reshape() };\n    device.sub.begin(x)\n        .fetch::<m![H / 16], m![H % 16]>()",
    "    // Only name the two already initialized cluster copies; Slice padding stays padding.\n    let x: DmTensorView<'_, bf16, Chip, InputClusters, m![Dummy256 / 32, 1 # 32], m![H]> = unsafe { x.view().reshape() };\n    device.sub.begin(x)\n        .fetch::<m![1], m![H]>()\n        .switch::<Replicated, m![1]>(SwitchConfig::CustomBroadcast { ring_size: 32 })")
oldops=(BASE/'src/ops.rs').read_text()
begin=oldops.index('    let x: DmTensor<bf16, Chip, m![2], m![Dummy256 / 32, 32], m![H]> = device')
end=oldops.index('    let x_trf = sliding::projection::prepare_shared_input',begin)
ops=oldops[:begin]+oldops[end:]
CRATE.mkdir(parents=True,exist_ok=True)
for file in ['Cargo.toml','Cargo.lock','rust-toolchain.toml']:shutil.copyfile(BASE/file,CRATE/file)
shutil.copytree(BASE/'src',CRATE/'src',dirs_exist_ok=True,copy_function=shutil.copyfile)
for directory in [CRATE,*CRATE.rglob('*')]:
    if directory.is_dir():directory.chmod(0o755)
for file,old,new,dst in [('projection.rs',original,projection,'src/device/sliding/projection.rs'),('ops.rs',oldops,ops,'src/ops.rs')]:
    (HERE/(file+'.original')).write_text(old)
    (HERE/file).write_text(new)
    (CRATE/dst).write_text(new)
print(CRATE)
