#!/usr/bin/env python3
"""Prepare one shared-TRF candidate from frozen E55 fusion/E54 ops."""
from pathlib import Path
import hashlib
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
BASE=ROOT/'moa-dev/experiments/artifacts/qkv-bf16-fused-channel-scale-e54/crate'
CRATE=ROOT/'moa-dev/experiments/artifacts/qkv-bf16-shared-trf-e55/crate'

def once(s,a,b):
    assert s.count(a)==1,(s.count(a),a)
    return s.replace(a,b)

original=(BASE/'src/device/sliding/projection.rs').read_text()
assert hashlib.sha256(original.encode()).hexdigest()=='380e57d6502e26086bd6d933ce762f4ad5de18587d52a8ef4aac7fa87e3b9c9f'
prefix,suffix=original.split('pub(crate) fn project_output(',1)
prefix=prefix.replace('    x: &DmTensor<bf16, Chip, m![2], Replicated, m![H]>,',
    '    x_trf: &SharedInputTrf,')
prefix=once(prefix,
    '    x_trf: &TrfTensor<bf16, Chip, KvCluster, KvRows, m![1], m![H]>,',
    '    x_trf: &SharedInputTrf,')
for cluster,slices in [('QueryCluster','QueryRows'),('KvCluster','KvRows')]:
    start=prefix.index(f"    let x: DmTensorView<'_, bf16, Chip, {cluster}, {slices}, m![H]>")
    end=prefix.index('        .to_trf();',start)+len('        .to_trf();\n\n')
    prefix=prefix[:start]+prefix[end:]
for axis,cluster,slices,local in [('Qs','QueryCluster','QueryRows',8),('Ps','KvCluster','KvRows',4)]:
    anchor=f'    let contraction: DmTensor<bf16, Chip, {cluster}, {slices}, m![{axis} / 8 % {local}]> = device'
    vrf_element=f'm![{axis} / 8 % {local}, 1 # 8]' if axis=='Qs' else 'm![Ps / 8 % 4 # 8, 1 # 8]'
    prefix=once(prefix,anchor,f'''    // Relabel identical physical coordinates; never transpose rows or H.
    let weight_f8_view: DmTensorView<'_, f8e4m3, Chip, InputClusters, Replicated, m![{axis} / 8 % {local}, H]> =
        unsafe {{ weight_f8.view().reshape() }};
    let weight_scale_vrf: VrfTensor<f32, Chip, InputClusters, Replicated, {vrf_element}> =
        unsafe {{ weight_scale_vrf.reshape() }};
    let contraction: DmTensor<bf16, Chip, InputClusters, Replicated, m![{axis} / 8 % {local}]> = device''')
    marker=f'        .fetch::<m![{axis} / 8 % {local}, H / 16], m![H % 16]>()'
    prefix=once(prefix,'        .begin(weight_f8.view())\n'+marker,'        .begin(weight_f8_view)\n'+marker)
    start=prefix.index(f'    let contraction: DmTensor<bf16, Chip, InputClusters, Replicated, m![{axis} / 8 % {local}]>')
    end=prefix.index('        .commit();',start)+len('        .commit();')
    prefix=prefix[:end]+f'''
    // Restore the original row names before the existing BF16 output Switch.
    let contraction: DmTensor<bf16, Chip, {cluster}, {slices}, m![{axis} / 8 % {local}]> =
        unsafe {{ contraction.reshape() }};'''+prefix[end:]
prefix=prefix.replace('>(&x_trf)', '>(x_trf)')
prefix=prefix.replace('project_one_kv_matrix(device, &x_trf,','project_one_kv_matrix(device, x_trf,')
helper='''// Both physical clusters and all 256 slices hold the same real BF16 input.
// Named neutral axes let Q/K/V borrow one TRF without a TRF reshape API.
type InputClusters = m![Dummy2];
pub(crate) type SharedInputTrf = TrfTensor<bf16, Chip, InputClusters, Replicated, m![1], m![H]>;

pub(crate) fn prepare_shared_input(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, m![2], Replicated, m![H]>,
) -> SharedInputTrf {
    let x: DmTensorView<'_, bf16, Chip, InputClusters, Replicated, m![H]> = unsafe { x.view().reshape() };
    device.sub.begin(x)
        .fetch::<m![H / 16], m![H % 16]>()
        .collect::<m![H / 16], m![H % 16]>()
        .to_trf()
}

'''
prefix=once(prefix,'pub(crate) fn project_query(',helper+'pub(crate) fn project_query(')
projection=prefix+'pub(crate) fn project_output('+suffix
ops_original=(BASE/'src/ops.rs').read_text()
ops=once(ops_original,
    '    let q: DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Gs, Ds]> =\n        sliding::projection::project_query(device, &x, q_weight, q_weight_scale);\n    let (k, v) = sliding::projection::project_key_value(device, &x, k_weight, v_weight, k_weight_scale, v_weight_scale);',
    '    let x_trf = sliding::projection::prepare_shared_input(device, &x);\n    let q: DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Gs, Ds]> =\n        sliding::projection::project_query(device, &x_trf, q_weight, q_weight_scale);\n    let (k, v) = sliding::projection::project_key_value(device, &x_trf, k_weight, v_weight, k_weight_scale, v_weight_scale);')
CRATE.mkdir(parents=True,exist_ok=True)
for file in ['Cargo.toml','Cargo.lock','rust-toolchain.toml']:
    shutil.copyfile(BASE/file,CRATE/file)
shutil.copytree(BASE/'src',CRATE/'src',dirs_exist_ok=True,copy_function=shutil.copyfile)
for directory in [CRATE,*CRATE.rglob('*')]:
    if directory.is_dir():directory.chmod(0o755)
for name,old,new,destination in [('projection.rs',original,projection,'src/device/sliding/projection.rs'),('ops.rs',ops_original,ops,'src/ops.rs')]:
    (HERE/(name+'.original')).write_text(old)
    (HERE/name).write_text(new)
    (CRATE/destination).write_text(new)
print(CRATE)
