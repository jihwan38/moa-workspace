#!/usr/bin/env python3
"""Read-only preservation and diagnostic-body audit; never builds or executes."""
from pathlib import Path
import hashlib,json,re,subprocess
HERE=Path(__file__).resolve().parent;TREE=HERE.parents[1];BASE='3f4b4141e83851c52346de7694181242b4b67a87'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def body(text,name):
    a=text.index('async fn '+name+'(');b=text.index('{',a);d=1;i=b+1
    while d:
        d+=(text[i]=='{')-(text[i]=='}');i+=1
    return re.sub(r',([)\]])',r'\1',re.sub(r'\s+','',text[b:i]))
def main():
    files=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE,'src','Cargo.toml','rust-toolchain.toml'],cwd=TREE,text=True).splitlines()
    for n in files:assert (TREE/n).read_bytes()==subprocess.check_output(['git','show',BASE+':'+n],cwd=TREE),n
    plan=json.loads((HERE/'plan.json').read_text())
    for n,s in plan['dependencies'].items():assert sha(TREE/n)==s
    for n,s in plan['donor_hashes'].items():assert sha(TREE/n)==s
    new=sorted(str(p.relative_to(TREE))for p in(TREE/'src').rglob('*')if p.is_file() and str(p.relative_to(TREE))not in files)
    assert new==['src/bin/support/int4_hadamard64_complete_fixture.rs','src/bin/support/int4_hadamard64_transform.rs','src/bin/test_int4_hadamard64_complete.rs']
    donor=(TREE/'src/bin/test_int4_weight_mse.rs').read_text();probe=(TREE/new[-1]).read_text()
    names=['production_chain','candidate_prefix']
    for n in names:assert body(probe,n)==body(donor,n),n
    assert 'candidate_clocked' not in probe
    return {'base':BASE,'preserved_source_config':len(files)+1,'source_config_count':len(files)+1+len(new),
        'new_host_files':{n:sha(TREE/n)for n in new},'donor_hashes':plan['donor_hashes'],'dependencies':plan['dependencies'],
        'unchanged_launch_bodies':names,'new_device_functions':0,'expected_images_if_built':19,
        'scope':'Source proof only, CPU activation roundtrips explicit; no timing/model/device result.'}
if __name__=='__main__':print(json.dumps(main(),indent=2))
