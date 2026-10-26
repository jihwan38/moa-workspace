#!/usr/bin/env python3
"""Read-only source/ELF/resource binding; never invokes device tools or objcopy."""
import json,hashlib
from pathlib import Path
import registry_audit
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
ROUND2=ROOT.parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    r=json.loads((HERE/'device-build/run.json').read_text())
    h=json.loads((HERE/'device-build/source-hashes.json').read_text())
    assert r['exit_code']==0 and r['result']=='command_succeeded' and r['build_inputs_unchanged']
    assert (HERE/'device-build/source.diff').read_bytes()==b''
    for name,digest in h.items():assert sha(ROOT/name)==digest,name
    assert len(h)==418
    relative='target/release/test_int4_grouped_activation_store'
    a=r['built_artifacts'][relative];elf=Path(a['path'])
    assert sha(elf)==a['sha256']
    registry=registry_audit.read_artifact(elf);assert registry['record_count']==2
    pairs=[]
    for row in registry['records']:
        mode='input' if row['path'].endswith('::quantize_pack_input') else 'hidden'
        assert row['path']==f'furiosa_opt_gemma4::ops_int4_grouped_activation_store::quantize_pack_{mode}'
        image=HERE/f'metadata-resource-images/{mode}.bin'
        fresh=HERE/'kernel-out-a007/furiosa-opt-gemma4'/f"{row['path']}.bin"
        assert row['key']==[] and row['sha256']==sha(image)==sha(fresh)
        pairs.append({'path':row['path'],'key':row['key'],'sha256':row['sha256'],'bytes':row['bytes'],'resource_image':str(image.relative_to(ROOT)),'fresh_build_image':str(fresh.relative_to(ROOT)),'exact':True})
    out={'source_commit':r['commit'],'functional_source':'f6ad9c83','captured_artifact':a,
         'source_inputs':len(h),'source_inputs_unchanged':True,'original_dependencies':{k:h[k] for k in ['Cargo.toml','Cargo.lock','rust-toolchain.toml']},
         'fresh_package':r['fresh_package'],'command':r['command'],'registry':registry,'resource_image_parity':pairs,
         'files':{p.name:sha(p) for p in sorted((HERE/'device-build').iterdir())},
         'probe_scope':json.loads((HERE/'standalone-plan.json').read_text()),
         'status':'BUILD_CAPTURE_BINDING_PASS; device correctness and all timing/quality remain unexecuted; root owns Arena.'}
    print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__':main()
