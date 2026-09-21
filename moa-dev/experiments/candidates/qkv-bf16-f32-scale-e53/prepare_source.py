#!/usr/bin/env python3
"""Assemble a private scratch candidate; never write the live crate."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
BASE = ROOT / 'moa-dev/experiments/artifacts/e53-bf16-output-clean-source'
ANCESTOR = ROOT / 'moa-dev/experiments/candidates/qkv-bf16-sharded-rms-e47'
CRATE = ROOT / 'moa-dev/experiments/artifacts/qkv-bf16-f32-scale-e53/crate'


def replace_once(text, old, new):
    assert text.count(old) == 1, (text.count(old), old)
    return text.replace(old, new)


original = (BASE/'src/device/sliding/projection.rs').read_text()
old_bf16 = (ANCESTOR/'projection.rs').read_text()
marker = 'pub(crate) fn project_output('
prefix = old_bf16.split(marker, 1)[0]
assert prefix.count('let contraction: DmTensor<bf16,') == 4
prefix = prefix.replace('let contraction: DmTensor<bf16,', 'let contraction: DmTensor<f32,')
assert prefix.count('        .cast::<bf16, m![1 # 16]>()\n') == 2
prefix = prefix.replace('        .cast::<bf16, m![1 # 16]>()\n', '')
prefix = replace_once(prefix,
    '        .transpose::<m![Qs / 32 % 2], m![Qs / 8 % 4 # 16]>()\n        .commit_trim::<m![Qs / 8 % 4]>()',
    '        .transpose::<m![Qs / 16 % 4], m![Qs / 8 % 2 # 8]>()\n        .commit_trim::<m![Qs / 8 % 2]>()')
prefix = replace_once(prefix,
    '        .transpose::<m![1], m![Ps / 8 % 4 # 16]>()\n        .commit_trim::<m![Ps / 8 % 4]>()',
    '        .transpose::<m![Ps / 16 % 2], m![Ps / 8 % 2 # 8]>()\n        .commit_trim::<m![Ps / 8 % 2]>()')
for axis in ('Qs', 'Ps'):
    prefix = replace_once(prefix,
        f'        .collect::<m![{axis} % 8], m![1 # 16]>()\n        .transpose::<m![{axis} / 4 % 2], m![{axis} % 4 # 16]>()\n        .commit_trim::<m![{axis} % 4]>()',
        f'        .collect::<m![{axis} % 8], m![1 # 8]>()\n        .transpose::<m![{axis} / 2 % 4], m![{axis} % 2 # 8]>()\n        .commit_trim::<m![{axis} % 2]>()')
    prefix = replace_once(prefix,
        f'        .begin(contraction.view())\n        .fetch::<m![1], m![{axis} % 8]>()\n        .fetch_cast::<f32>()',
        f'        .begin(contraction.view())\n        .fetch::<m![1], m![{axis} % 8]>()')
prefix = replace_once(prefix, '.fetch::<m![Qs / 8 % 8], m![1 # 4]>()', '.fetch::<m![Qs / 8 % 8], m![1 # 2]>()')
prefix = replace_once(prefix, '.fetch::<m![Ps / 8 % 4 # 8], m![1 # 4]>()', '.fetch::<m![Ps / 8 % 4 # 8], m![1 # 2]>()')
prefix = prefix.replace('    // Switch swaps the low slice coordinate with the local row coordinate.',
    '    // Raw dot and row transpose stay F32; the first BF16 cast follows channel scale.\n'
    '    // Switch swaps the low slice coordinate with the local row coordinate.')
prefix = prefix.replace('    // Four valid local rows are padded to an eight-way Switch group.',
    '    // Preserve F32 raw dots across Switch; cast only after per-channel scale.\n'
    '    // Four valid local rows are padded to an eight-way Switch group.')
projection = prefix + marker + original.split(marker, 1)[1]

ops_original = (BASE/'src/ops.rs').read_text()
ops_bf16 = (ANCESTOR/'ops.rs').read_text()
start = '#[device(chip = 1)]\npub fn sliding_project_qkv('
end = '#[device(chip = 1)]\npub fn full_project_qkv('
new_function = ops_bf16[ops_bf16.index(start):ops_bf16.index(end)]
ops = ops_original[:ops_original.index(start)] + new_function + ops_original[ops_original.index(end):]

CRATE.mkdir(parents=True, exist_ok=True)
for file in ['Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml']:
    shutil.copyfile(BASE/file, CRATE/file)
shutil.copytree(BASE/'src', CRATE/'src', dirs_exist_ok=True, copy_function=shutil.copyfile)
for path in [CRATE, *CRATE.rglob('*')]:
    if path.is_dir():
        path.chmod(0o755)
(HERE/'projection.rs.original').write_text(original)
(HERE/'projection.rs').write_text(projection)
(HERE/'ops.rs.original').write_text(ops_original)
(HERE/'ops.rs').write_text(ops)
(CRATE/'src/device/sliding/projection.rs').write_text(projection)
(CRATE/'src/ops.rs').write_text(ops)
print('Created QKV BF16/F32-scale candidate; unchanged E53 O suffix and all other sources.')
