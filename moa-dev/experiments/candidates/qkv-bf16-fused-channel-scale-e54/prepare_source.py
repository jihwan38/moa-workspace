#!/usr/bin/env python3
"""Create one isolated candidate from frozen E54; never touch live source."""
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT / 'moa-dev/experiments/artifacts/e54-bf16-qkv-clean-source'
CRATE = ROOT / 'moa-dev/experiments/artifacts/qkv-bf16-fused-channel-scale-e54/crate'

def once(text, old, new):
    assert text.count(old) == 1, (text.count(old), old)
    return text.replace(old, new)

original = (BASE/'src/device/sliding/projection.rs').read_text()
marker = 'pub(crate) fn project_output('
prefix, suffix = original.split(marker, 1)
for axis, cluster, slices, local in [('Qs', 'QueryCluster', 'QueryRows', 8), ('Ps', 'KvCluster', 'KvRows', 4)]:
    raw = f'm![{axis} / 8 % {local}]'
    anchor = f'    let contraction: DmTensor<f32, Chip, {cluster}, {slices}, {raw}> = device'
    scale = f'''    // Compact row scales use the same interleaved row coordinates as the dot.
    // Each dot Time reads one F32 scalar (Q: 8 rows, K/V: 4 rows per slice).
    let weight_scale: DmTensor<bf16, Chip, {cluster}, {slices}, {raw}> = weight_scale.to_dm(&mut device.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, {cluster}, {slices}, {raw}> = device
        .sub.begin(weight_scale.view())
        .fetch::<m![1], {raw}>()
        .fetch_cast::<f32>()
        .collect::<m![1], {raw}>()
        .to_vrf();
    let contraction: DmTensor<bf16, Chip, {cluster}, {slices}, {raw}> = device'''
    prefix = once(prefix, anchor, scale)
    lane = f'        .contract_lane::<{raw}, m![1 # 8]>(LaneMode::Interleaved)'
    prefix = once(prefix, lane, lane + '''
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale_vrf)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()''')
    old_time = f'm![{axis} / 16 % {local//2}]'
    new_time = f'm![{axis} / 32 % 2]' if axis == 'Qs' else 'm![1]'
    prefix = once(prefix,
        f'        .transpose::<{old_time}, m![{axis} / 8 % 2 # 8]>()\n        .commit_trim::<m![{axis} / 8 % 2]>()',
        f'        .transpose::<{new_time}, m![{axis} / 8 % 4 # 16]>()\n        .commit_trim::<m![{axis} / 8 % 4]>()')
    canonical = f'm![{axis} / 8 % 256]' if axis == 'Qs' else 'm![Ps / 32 % 32, Ps / 8 % 4 # 8]'
    prefix = once(prefix,
        f'    let contraction: DmTensor<f32, Chip, {cluster}, {canonical}, m![{axis} % 8]> = device',
        f'    let scaled: DmTensor<bf16, Chip, {cluster}, {canonical}, m![{axis} % 8]> = device')
    rowtime = raw if axis == 'Qs' else 'm![Ps / 8 % 4 # 8]'
    prefix = once(prefix, f'.fetch::<{rowtime}, m![1 # 2]>()', f'.fetch::<{rowtime}, m![1 # 4]>()')
    prefix = once(prefix,
        f'        .collect::<m![{axis} % 8], m![1 # 8]>()\n        .transpose::<m![{axis} / 2 % 4], m![{axis} % 2 # 8]>()\n        .commit_trim::<m![{axis} % 2]>()',
        f'        .collect::<m![{axis} % 8], m![1 # 16]>()\n        .transpose::<m![{axis} / 4 % 2], m![{axis} % 4 # 16]>()\n        .commit_trim::<m![{axis} % 4]>()')
    begin = prefix.index(f'    let weight_scale: DmTensor<bf16, Chip, {cluster}, {canonical}, m![{axis} % 8]>')
    end = prefix.index('    let output:', begin)
    prefix = prefix[:begin] + prefix[end:]
# Correction 1: to_vrf requires one 32B F32 flit. Keep four valid K/V
# values and pad only the collect/VRF mapping; the scalar dot reads rows 0..3.
prefix = once(prefix,
    'VrfTensor<f32, Chip, KvCluster, KvRows, m![Ps / 8 % 4]>',
    'VrfTensor<f32, Chip, KvCluster, KvRows, m![Ps / 8 % 4 # 8]>')
prefix = once(prefix,
    '.collect::<m![1], m![Ps / 8 % 4]>()',
    '.collect::<m![1], m![Ps / 8 % 4 # 8]>()')
# Correction 2: raw row DMA has illegal 2-byte TDMA strides. Read the known
# aligned canonical layout, then route each scale through an inverse Sub Switch.
for axis, cluster, slices, local in [('Qs','QueryCluster','QueryRows',8), ('Ps','KvCluster','KvRows',4)]:
    start = prefix.index('    // Compact row scales', prefix.index('pub(crate) fn project_query' if axis=='Qs' else 'fn project_one_kv_matrix'))
    end = prefix.index('    let contraction:', start)
    canonical = 'm![Qs / 8 % 256]' if axis=='Qs' else 'm![Ps / 32 % 32, Ps / 8 % 4 # 8]'
    localtime = f'm![{axis} / 8 % {local}]' if axis=='Qs' else 'm![Ps / 8 % 4 # 8]'
    vrf = f'm![{axis} / 8 % {local}, 1 # 8]' if axis=='Qs' else 'm![Ps / 8 % 4 # 8, 1 # 8]'
    scale = f'''    // Canonical contiguous scale DMA satisfies TDMA's 8-byte alignment.
    // Inverse row Switch routes the matching scale into each raw dot slice.
    // VRF stores one valid scalar in each 32-byte Time slot; padding is unread.
    let weight_scale: DmTensor<bf16, Chip, {cluster}, {canonical}, m![{axis} % 8]> = weight_scale.to_dm(&mut device.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, {cluster}, {slices}, {vrf}> = device
        .sub.begin(weight_scale.view())
        .fetch::<m![{axis} % 8], m![1 # 4]>()
        .fetch_cast::<f32>()
        .switch::<{slices}, {localtime}>(SwitchConfig::InterTranspose {{ slice1: 8, slice0: 1, time0: 1 }})
        .collect::<{localtime}, m![1 # 8]>()
        .to_vrf();
'''
    prefix = prefix[:start] + scale + prefix[end:]
prefix = prefix.replace('// Raw dot and row transpose stay F32; the first BF16 cast follows channel scale.',
    '// The complete F32 dot is scaled before its first BF16 cast; Switch only moves BF16 values.')
prefix = prefix.replace('// Preserve F32 raw dots across Switch; cast only after per-channel scale.',
    '// Scale the complete F32 dot before BF16 cast, then preserve the scaled value across Switch.')
result = prefix + marker + suffix
assert result.count('pub(crate) fn project_output(') == 1
assert result.split(marker,1)[1] == original.split(marker,1)[1]
CRATE.mkdir(parents=True, exist_ok=True)
for file in ['Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml']:
    shutil.copyfile(BASE/file, CRATE/file)
shutil.copytree(BASE/'src', CRATE/'src', dirs_exist_ok=True, copy_function=shutil.copyfile)
for directory in [CRATE, *CRATE.rglob('*')]:
    if directory.is_dir():
        directory.chmod(0o755)
(HERE/'projection.rs.original').write_text(original)
(HERE/'projection.rs').write_text(result)
(CRATE/'src/device/sliding/projection.rs').write_text(result)
print(CRATE)
