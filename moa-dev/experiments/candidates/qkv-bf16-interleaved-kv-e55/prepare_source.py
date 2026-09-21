#!/usr/bin/env python3
"""Reproduce the interleaved K/V candidate from the frozen E55 source crate."""
from pathlib import Path
import hashlib
import shutil
import difflib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT / 'moa-dev/experiments/artifacts/qkv-bf16-fused-channel-scale-e54/crate'
CRATE = ROOT / 'moa-dev/experiments/artifacts/qkv-bf16-interleaved-kv-e55/crate'
original = (BASE / 'src/device/sliding/projection.rs').read_text()
assert hashlib.sha256(original.encode()).hexdigest() == '380e57d6502e26086bd6d933ce762f4ad5de18587d52a8ef4aac7fa87e3b9c9f'
prefix, remainder = original.split('fn project_one_kv_matrix(', 1)
old_kv, suffix = remainder.split('pub(crate) fn project_output(', 1)
# Preserve the original BF16 row Switch and typed head gather exactly. Only its
# already-committed input is now a view of one half of the combined output.
tail = old_kv[old_kv.index('    // Scale the complete F32 dot'):old_kv.index('\npub(crate) fn project_key_value(')]
tail = tail.replace('.main.begin(contraction.view())', '.main.begin(contraction)')
finish = '''fn finish_one_kv_matrix(
    device: &mut Device,
    contraction: DmTensorView<'_, bf16, Chip, KvCluster, KvRows, m![Ps / 8 % 4]>,
) -> DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Ds]> {
''' + tail
new_kv = '''pub(crate) fn project_key_value(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, m![2], Replicated, m![H]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    v_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Ds]>,
    DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Ds]>,
) {
    let x: DmTensorView<'_, bf16, Chip, KvCluster, KvRows, m![H]> = unsafe { x.view().reshape() };
    let x_trf: TrfTensor<bf16, Chip, KvCluster, KvRows, m![1], m![H]> = device
        .sub
        .begin(x)
        .fetch::<m![H / 16], m![H % 16]>()
        .collect::<m![H / 16], m![H % 16]>()
        .to_trf();

    // Same typed row distribution as E55; the two HBM matrices remain separate.
    let k_weight_f8: DmTensor<f8e4m3, Chip, KvCluster, KvRows, m![Ps / 8 % 4, H]> = k_weight.to_dm(&mut device.tdma);
    let v_weight_f8: DmTensor<f8e4m3, Chip, KvCluster, KvRows, m![Ps / 8 % 4, H]> = v_weight.to_dm(&mut device.tdma);
    let k_scale: DmTensor<bf16, Chip, KvCluster, m![Ps / 32 % 32, Ps / 8 % 4 # 8], m![Ps % 8]> = k_weight_scale.to_dm(&mut device.tdma);
    let v_scale: DmTensor<bf16, Chip, KvCluster, m![Ps / 32 % 32, Ps / 8 % 4 # 8], m![Ps % 8]> = v_weight_scale.to_dm(&mut device.tdma);

    // Dummy2 selects matrix 0=K, 1=V. It remains a Time axis in the
    // inverse scale Switch and in the contraction; it is never reduced.
    let scale_vrf: VrfTensor<f32, Chip, KvCluster, KvRows, m![Dummy2, Ps / 8 % 4 # 8, 1 # 8]> = device
        .sub
        .begin_interleaved::<Dummy2, _, _, _, _, _>(k_scale.view(), v_scale.view())
        .fetch::<m![Dummy2, Ps % 8], m![1 # 4]>()
        .fetch_cast::<f32>()
        .switch::<KvRows, m![Dummy2, Ps / 8 % 4 # 8]>(SwitchConfig::InterTranspose { slice1: 8, slice0: 1, time0: 1 })
        .collect::<m![Dummy2, Ps / 8 % 4 # 8], m![1 # 8]>()
        .to_vrf();

    let contraction: DmTensor<bf16, Chip, KvCluster, KvRows, m![Dummy2, Ps / 8 % 4]> = device
        .main
        .begin_interleaved::<Dummy2, _, _, _, _, _>(k_weight_f8.view(), v_weight_f8.view())
        .fetch::<m![Dummy2, Ps / 8 % 4, H / 16], m![H % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Dummy2, Ps / 8 % 4, H / 16], m![H % 16]>()
        .contract_outer::<m![Dummy2, Ps / 8 % 4, H / 32], m![H % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Dummy2, Ps / 8 % 4]>()
        .contract_lane::<m![Dummy2, Ps / 8 % 4], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Dummy2], m![Ps / 8 % 4 # 16]>()
        .commit_trim::<m![Ps / 8 % 4]>()
        .commit();

    // Each half has four committed BF16 values. Public tile fixes the matrix
    // selector and unpad removes only the resulting outer padding; no data or
    // pointer arithmetic is invented, and existing row/head restoration follows.
    let k = finish_one_kv_matrix(device, contraction.view()
        .tile::<m![Dummy2], 1, m![1 # 2, Ps / 8 % 4]>(0)
        .unpad::<m![Ps / 8 % 4]>());
    let v = finish_one_kv_matrix(device, contraction.view()
        .tile::<m![Dummy2], 1, m![1 # 2, Ps / 8 % 4]>(1)
        .unpad::<m![Ps / 8 % 4]>());
    (k, v)
}

'''
source = prefix + finish + new_kv + 'pub(crate) fn project_output(' + suffix
assert source.split('pub(crate) fn project_query(',1)[1].split('type KvCluster',1)[0] == original.split('pub(crate) fn project_query(',1)[1].split('type KvCluster',1)[0]
assert source.split('pub(crate) fn project_output(',1)[1] == suffix
CRATE.mkdir(parents=True, exist_ok=True)
for name in ['Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml']:
    shutil.copyfile(BASE / name, CRATE / name)
shutil.copytree(BASE / 'src', CRATE / 'src', dirs_exist_ok=True, copy_function=shutil.copyfile)
for p in [CRATE, *CRATE.rglob('*')]:
    if p.is_dir():
        p.chmod(0o755)
(HERE / 'projection.rs.original').write_text(original)
(HERE / 'projection.rs').write_text(source)
(CRATE / 'src/device/sliding/projection.rs').write_text(source)
(HERE / 'candidate.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True), source.splitlines(True), fromfile='a/src/device/sliding/projection.rs', tofile='b/src/device/sliding/projection.rs')))
print(CRATE)
print('projection_sha256', hashlib.sha256(source.encode()).hexdigest())
