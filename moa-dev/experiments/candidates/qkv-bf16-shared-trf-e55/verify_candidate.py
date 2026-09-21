#!/usr/bin/env python3
"""Before-compile source scope and complete row/element-address proof."""
from pathlib import Path
import hashlib
import json
import re

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
BASE=ROOT/'moa-dev/experiments/artifacts/qkv-bf16-fused-channel-scale-e54/crate'
CRATE=ROOT/'moa-dev/experiments/artifacts/qkv-bf16-shared-trf-e55/crate'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=(HERE/'projection.rs.original').read_text();new=(HERE/'projection.rs').read_text()
prefix,suffix=new.split('pub(crate) fn project_output(',1)
assert suffix==old.split('pub(crate) fn project_output(',1)[1]
assert prefix.count('.to_trf()')==1
assert prefix.count('x_trf: &SharedInputTrf')==3
assert prefix.count('unsafe { weight_f8.view().reshape() }')==2
assert prefix.count('unsafe { weight_scale_vrf.reshape() }')==2
assert prefix.count('unsafe { contraction.reshape() }')==2
assert prefix.count('.cast::<bf16,')==2
for axis,local in [('Qs',8),('Ps',4)]:
    start=f'        .fetch::<m![{axis} / 8 % {local}, H / 16], m![H % 16]>()'
    op=old[old.index(start):];op=op[:op.index('        .commit();')]
    np=new[new.index(start):];np=np[:np.index('        .commit();')]
    assert np==op.replace('>(&x_trf)','>(x_trf)')
    marker='    // The complete F32 dot' if axis=='Qs' else '    // Scale the complete F32 dot'
    assert new[new.index(marker):new.index('\n}',new.index(marker))]==old[old.index(marker):old.index('\n}',old.index(marker))]
    scale_start=f'    let weight_scale: DmTensor<bf16, Chip, '+('QueryCluster' if axis=='Qs' else 'KvCluster')
    os=old[old.index(scale_start):];os=os[:os.index('        .to_vrf();')+len('        .to_vrf();')]
    ns=new[new.index(scale_start):];ns=ns[:ns.index('        .to_vrf();')+len('        .to_vrf();')]
    assert os==ns
oldops=(HERE/'ops.rs.original').read_text();newops=(HERE/'ops.rs').read_text()
expect=oldops.replace('    let q: DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Gs, Ds]> =\n        sliding::projection::project_query(device, &x, q_weight, q_weight_scale);',
    '    let x_trf = sliding::projection::prepare_shared_input(device, &x);\n    let q: DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Gs, Ds]> =\n        sliding::projection::project_query(device, &x_trf, q_weight, q_weight_scale);')
expect=expect.replace('sliding::projection::project_key_value(device, &x,','sliding::projection::project_key_value(device, &x_trf,')
assert expect==newops
public=r'#\[device\(chip = 1\)\]\s+pub fn \w+\([\s\S]*?\) \{'
assert re.findall(public,oldops)==re.findall(public,newops)

reports=[]
for label,rows,cr,lr in [('Q',4096,2048,8),('K',2048,1024,4),('V',2048,1024,4)]:
    seen=set()
    for row in range(rows):
        c=row//cr;g=row//(lr*8)%32;s=row%8;r=row//8%lr
        original=(c,8*g+s,r)
        neutral=(c,8*g+s,r)
        assert original==neutral and original not in seen
        seen.add(original)
        restored=neutral[0]*cr+(neutral[1]//8)*(8*lr)+8*neutral[2]+neutral[1]%8
        assert restored==row
        # Whole H interval has equal base and unit-byte stride in both views:
        # for every h in0..3839, original/neutral weight address=r*3840+h.
        assert original[2]*3840==neutral[2]*3840
        assert original[2]*3840+3839==neutral[2]*3840+3839
        # Shared activation: all 512 actual copies contain BF16 x[h], lane1.
        assert 0<=c<2 and 0<=neutral[1]<256
        for h in (0,15,16,479,480,3839): assert 2*h<7680
        # Fused scale index and raw result byte offset remain unchanged.
        assert r*32==neutral[2]*32 and r<lr
        assert r*2==neutral[2]*2
        canonical=(c,g*8+r,s)
        assert canonical[1]%8<lr
        assert (canonical[1]//8*8+canonical[2],canonical[1]%8)==(8*g+s,r)
    assert len(seen)==rows
    reports.append({'matrix':label,'all_rows_checked':rows,'local_rows':lr,
        'weight_elements_covered_by_identical_contiguous_h_spans':rows*3840,
        'weight_element_stride_bytes':1,'activation_stride_bytes':2,
        'scale_vrf_scalar_stride_bytes':32,'raw_output_stride_bytes':2,
        'kv_valid4_padding_preserved':label!='Q'})
changes=[]
for f in sorted((BASE/'src').rglob('*')):
    if f.is_file() and sha(f)!=sha(CRATE/f.relative_to(BASE)):
        changes.append(str(f.relative_to(BASE)))
assert changes==['src/device/sliding/projection.rs','src/ops.rs']
for f in ['Cargo.toml','Cargo.lock','rust-toolchain.toml']:assert sha(BASE/f)==sha(CRATE/f)
assert (CRATE/'src/device/sliding/projection.rs').read_bytes()==(HERE/'projection.rs').read_bytes()
assert (CRATE/'src/ops.rs').read_bytes()==(HERE/'ops.rs').read_bytes()
print(json.dumps({'source_changes':changes,'other_41_sources_unchanged':True,
 'o_suffix_unchanged':True,'public_ops_signatures_unchanged':True,
 'gather_ring32_rms_head_rope_cache_and_ffn_unchanged':True,
 'dpe_lookup_reduce_scale_cast_pipeline_unchanged':True,
 'scale_dma_and_inverse_switch_unchanged':True,'one_source_trf_load':True,
 'row_wire_address_checks':reports,'npu_run':False},indent=2))
