#!/usr/bin/env python3
"""Read-only source/proof/checkpoint audit. No build or device actions."""
import hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
TREE=HERE.parents[1]
BASE='5e3ee88365cd30ccf4919d60726d68751d4992a6'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def audit():
    old=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE,'src','Cargo.toml','rust-toolchain.toml'],cwd=TREE,text=True).splitlines()
    hashes={}
    for n in old:
        expected=subprocess.check_output(['git','show',BASE+':'+n],cwd=TREE)
        assert (TREE/n).read_bytes()==expected,n
        hashes[n]=sha(TREE/n)
    plan=json.loads((HERE/'plan.json').read_text())
    for n,s in plan['dependencies'].items():assert sha(TREE/n)==s,n
    for n,s in plan['proof_hashes'].items():assert sha(HERE/'independent-proof'/n)==s,n
    new=sorted(str(p.relative_to(TREE)) for p in (TREE/'src').rglob('*') if p.is_file() and str(p.relative_to(TREE)) not in hashes)
    assert new==['src/bin/support/int4_hadamard64.rs','src/bin/test_int4_hadamard64.rs'],new
    prior=json.loads((TREE/'experiments/int4-weight-mse-diagnostic/build-audit.json').read_text())
    suffixes=['ops_ffn_projection_join::join','ops_group_ffn_projection_storegroup::sliding_projection_owned',
        'ops_int4_ffn_bridges::geglu_bf16','ops_int4_ffn_bridges::prefix_sliding_bf16',
        'ops_prepack_alloc::prepack_down','ops_prepack_alloc::prepack_up','ops_prepack_alloc::reblock_down','ops_prepack_alloc::reblock_up']
    images=[r for r in prior['registry']['records'] if any(r['path'].endswith(s) for s in suffixes)]
    assert len(images)==8
    checks=[]
    for name in ['a003-int4-h64-cpu-proof','a004-int4-h64-hostcheck']:
        folder=HERE/name;r=json.loads((folder/'run.json').read_text());h=json.loads((folder/'source-hashes.json').read_text())
        assert r['exit_code']==0 and r['result']=='command_succeeded'
        for n,s in h.items():assert sha(TREE/n)==s,(name,n)
        checks.append({'name':name,'run_sha256':sha(folder/'run.json'),'log_sha256':sha(folder/'output.log'),'inputs':len(h),'result':'PASS'})
    return {'base':BASE,'functional_source':'afec8a05','old_source_config_unchanged':len(old)+1,
        'source_config_count':len(old)+1+len(new),'new_host_files':{n:sha(TREE/n) for n in new},
        'original_rational_oracle_sha256':sha(TREE/'src/bin/support/rational_a4_oracle.rs'),
        'dependencies':plan['dependencies'],'proof_hashes':plan['proof_hashes'],'checks':checks,
        'expected_reused_images_if_later_built':images,'device_images_verified':False,
        'scope':'Source/CPU/plain-hostcheck only. Eight expected old images: three runtime and five startup. No device build, sampled model result, quality or speed claim.'}
if __name__=='__main__':print(json.dumps(audit(),indent=2))
