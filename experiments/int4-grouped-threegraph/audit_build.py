#!/usr/bin/env python3
"""Read-only fresh/captured ELF and donor/resource image binding."""
from pathlib import Path
import hashlib,importlib.util,json
ROOT=Path(__file__).resolve().parents[3]
TREE=ROOT/'int4-grouped-threegraph'
HERE=TREE/'experiments/int4-grouped-threegraph'
RUN=ROOT/'runs/a012-int4-grouped-threegraph-device-build'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
    r=read(RUN/'run.json');assert r['exit_code']==0 and r['build_inputs_unchanged']
    assert r['fresh_package']['contents_changed'] is False
    hashes=read(RUN/'source-hashes.json');assert len(hashes)==424
    for path,h in hashes.items():assert sha(TREE/path)==h,path
    old_inputs=read(ROOT/'runs/a002-int4-grouped-complete-ffn-device-build/source-hashes.json')
    changed={'src/lib.rs','src/ops_int4_ffn_bridges.rs','src/ops_int4_grouped_activation_store.rs','src/ops_int4_grouped_activation_projection.rs'}
    for path,h in old_inputs.items():
        if path not in changed:assert hashes[path]==h,path
    extra={'src/ops_int4_grouped_threegraph.rs','src/i4_grouped_threegraph_raw_adapter.rs','src/bin/test_int4_grouped_threegraph.rs'}
    assert set(hashes)-set(old_inputs)==extra
    assert hashes['Cargo.lock']=='a99652661c588fe5c3a8b23278f2390d8cd2e99f2b33bcab6f52e97dbba04e2e'
    assert hashes['Cargo.toml']=='a4154d908d25a512d031dea859b1b013794f821a47deb739334332bd0e4fdf85'
    parser=ROOT/'b16-f4-mse-lm-fp8/experiments/sdk09-cache-audit/audit_registry.py'
    spec=importlib.util.spec_from_file_location('registry',parser);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    artifact=r['built_artifacts']['target/release/test_int4_grouped_threegraph']
    for key in ['path','source_path']:
        p=Path(artifact[key]);assert sha(p)==artifact['sha256'] and p.stat().st_size==artifact['size']
    current=m.read_artifact(artifact['path'])
    prior_path=ROOT/'runs/a002-int4-grouped-complete-ffn-device-build/built/target/release/test_int4_grouped_complete_ffn'
    prior=m.read_artifact(prior_path)
    assert prior['elf_sha256']=='3444ef561e2d39d6a15ec6340548d07e65b873d2884b62ac04f9ea2ccd66cdd4'
    def index(d):
        assert all(not x['key'] for x in d['records'])
        rows={x['path']:x for x in d['records']};assert len(rows)==len(d['records']);return rows
    old=index(prior);new=index(current);assert len(old)==19 and len(new)==22
    resources=read(HERE/'resource-result.json')
    added={Path(x['path']).stem:x for x in resources['images'] if 'graph' in x}
    assert len(added)==3 and set(new)==set(old)|set(added)
    outarg=next(x for x in r['command'] if x.startswith('FURIOSA_OPT_OUT_DIR='))
    out=ROOT/outarg.split('=',1)[1].removeprefix('/round2/')/'furiosa-opt-gemma4'
    comparisons=[]
    for name,entry in sorted(new.items()):
        expected=old[name] if name in old else added[name]
        assert (entry['sha256'],entry['bytes'])==(expected['sha256'],expected['bytes']),name
        fresh=out/(name+'.bin');assert sha(fresh)==entry['sha256'] and fresh.stat().st_size==entry['bytes'],name
        if name in added:assert sha(TREE/added[name]['path'])==entry['sha256']
        comparisons.append({'path':name,'reference':'actual323213' if name in old else 'same-source resource81fe314b',
            'sha256':entry['sha256'],'bytes':entry['bytes'],'reference_equal':True,'fresh_out_bin_equal':True})
    return {'run':r['name'],'source_commit':r['commit'],'functional_source':'d5e70b98','input_hashes':len(hashes),
        'all_inputs_unchanged':True,'unchanged_old_inputs':len(old_inputs)-len(changed),'only_modified_old_source':sorted(changed),
        'new_inputs':sorted(extra),'built_artifact':artifact,'registry':current,'comparisons':comparisons,
        'prior_artifact':{'path':str(prior_path),'sha256':prior['elf_sha256']},'parser_sha256':sha(parser),
        'original19_equal':True,'ABC3_resource_equal':True,'scope':'Build/image identity only. No actual threegraph correctness/speed/model-quality result.'}
if __name__=='__main__':print(json.dumps(main(),indent=2))
