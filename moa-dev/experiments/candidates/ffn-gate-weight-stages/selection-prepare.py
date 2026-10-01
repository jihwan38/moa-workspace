from pathlib import Path
import hashlib,json,difflib
R=Path('/home/jihwan38/moa-workspace');P=Path('/tmp/moa-ffn-gate-staged-preparation');P.mkdir(exist_ok=True)
src=(R/'moa-dev/experiments/candidates/ffn-down-weight-stages/staged-v1/mlp.rs').read_text();assert hashlib.sha256(src.encode()).hexdigest()=='813aa7068376a41833723748ba31bf43008972bb5ff9b4e1e7cbc3d185a23530'
start=src.index('fn project_rows_with_scale_contraction(');end=src.index('\npub(crate) fn project_up_and_gate(',start);old=src[start:end]
consumers=old[old.index('    static_passes!'):]
producer=old[old.index('    let partials:'):old.index('    static_passes!')]
producer=producer.replace('    let partials: DmTensor<f32, Chip, UpGateCluster, UpGateSlices, m![FfnRow, H / 16]> = device','    device').replace('FfnRow','FfnRow = 15').replace('.commit();','.commit_view(partials.view_mut().tile::<m![FfnRow], 15, m![FfnRow = 15 #{!} 30, H / 16]>(offset));')
helper='''// Prepared source only: compact gate half -> disjoint original full-row partial.
fn project_gate_native_half(
    device: &mut Device,
    x_trf: &TrfTensor<f8e5m2, Chip, UpGateCluster, UpGateSlices, m![1], m![FfnTerm, H]>,
    x_scale_vrf: &VrfTensor<f32, Chip, UpGateCluster, UpGateSlices, m![1 # 8]>,
    weights: &DmTensor<f4e2m1, Chip, UpGateCluster, UpGateSlices, m![FfnRow = 15, H]>,
    partials: &mut DmTensor<f32, Chip, UpGateCluster, UpGateSlices, m![FfnRow, H / 16]>,
    offset: usize,
) {
'''+producer+''' }

fn project_gate_rows_staged(
    device: &mut Device,
    x_trf: &TrfTensor<f8e5m2, Chip, UpGateCluster, UpGateSlices, m![1], m![FfnTerm, H]>,
    x_scale_vrf: &VrfTensor<f32, Chip, UpGateCluster, UpGateSlices, m![1 # 8]>,
    first: &DmTensor<f4e2m1, Chip, UpGateCluster, UpGateSlices, m![FfnRow = 15, H]>,
    second: &DmTensor<f4e2m1, Chip, UpGateCluster, UpGateSlices, m![FfnRow = 15, H]>,
    scales: &DmTensor<f8e4m3, Chip, UpGateCluster, UpGateSlices, m![FfnRow, H / 16]>,
    output: &mut DmTensor<f32, Chip, UpGateCluster, UpGateSlices, m![FfnRow]>,
) {
    let mut partials: DmTensor<f32, Chip, UpGateCluster, UpGateSlices, m![FfnRow, H / 16]> = DmTensor::new();
    project_gate_native_half(device, x_trf, x_scale_vrf, first, &mut partials, 0);
    project_gate_native_half(device, x_trf, x_scale_vrf, second, &mut partials, 15);
'''+consumers
s=src[:end]+'\n'+helper+src[end:]
needle='''    let gate_weight_packed: DmTensor<f4e2m1, Chip, UpGateCluster, UpGateSlices, m![FfnRow, H]> =
        gate_weight_packed.to_dm(&mut device.tdma);'''
replace='''    let gate_weight_first: DmTensor<f4e2m1, Chip, UpGateCluster, UpGateSlices, m![FfnRow = 15, H]> = gate_weight_packed
        .tile::<m![FfnRow], 15, m![Dummy2, FfnSlice, FfnRow = 15 # 30, H]>(0)
        .to_dm(&mut device.tdma);
    let gate_weight_second: DmTensor<f4e2m1, Chip, UpGateCluster, UpGateSlices, m![FfnRow = 15, H]> = gate_weight_packed
        .tile::<m![FfnRow], 15, m![Dummy2, FfnSlice, FfnRow = 15 # 30, H]>(15)
        .to_dm(&mut device.tdma);'''
assert s.count(needle)==1;s=s.replace(needle,replace)
needle='project_rows_with_scale_contraction(device, &x_trf, &x_scale_vrf, &gate_weight_packed, &gate_weight_scale, &mut gate);';assert s.count(needle)==1;s=s.replace(needle,'project_gate_rows_staged(device, &x_trf, &x_scale_vrf, &gate_weight_first, &gate_weight_second, &gate_weight_scale, &mut gate);')
assert s[s.index('fn cast_gathered_rows'):]==src[src.index('fn cast_gathered_rows'):]
assert old in s and consumers in helper
(P/'mlp.rs').write_text(s);(P/'source.diff').write_text(''.join(difflib.unified_diff(src.splitlines(True),s.splitlines(True),fromfile='E150/mlp.rs',tofile='prepared/mlp.rs')))
record={'status':'source-only unselected; no compile/NPU','origin_sha256':hashlib.sha256(src.encode()).hexdigest(),'proposal':{'bytes':len(s.encode()),'sha256':hashlib.sha256(s.encode()).hexdigest()},'intended_change':'Gate only full30FP4 ingress/native -> two15row compact native streams writing original full30F32partial halves, old five6scaleconsumers unchanged','half_weight_bytes_per_slice':28800,'half_partial_bytes_per_slice':14400,'partial_offsets':[0,14400],'cross_half_consumer_rows':[12,17],'added':['One gate DMA/native/lookup and live alias join'],'unchanged':['All logical weight/partial traffic','original up helper','all source after cast_gathered_rows incl E150down','scale values/arithmetic/consumerbody'],'must_verify':['actual15row fetch/partialcommit bounds','TRF CSE remains shared and coefficient lifetime','ready activation gather interleaves between gate DMAs; no source-order guarantee','current source-recipe composition if E151 adopted','source/full/standard/raw-up oracle before adoption']}
(P/'source-contract.json').write_text(json.dumps(record,indent=2)+'\n');print(record['proposal'])
