#!/usr/bin/env python3
"""Captured artifact audit only; never builds or invokes the NPU."""
from pathlib import Path
import argparse,hashlib,importlib.util,json
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
ap=argparse.ArgumentParser();ap.add_argument('--run',default='a004-int4-drift-attribution-device-build');args=ap.parse_args()
tree=Path(__file__).resolve().parents[2];root=tree.parent
parser=root/'b16-f4-mse-lm-fp8/experiments/sdk09-cache-audit/audit_registry.py'
spec=importlib.util.spec_from_file_location('registry',parser);registry=importlib.util.module_from_spec(spec);spec.loader.exec_module(registry)
runroot=root/'runs'/args.run;run=json.loads((runroot/'run.json').read_text());assert run['exit_code']==0 and run['build_inputs_unchanged']
inputs=json.loads((runroot/'source-hashes.json').read_text())
for p,h in inputs.items():assert sha(tree/p)==h,p
oldrun=root/'runs/a005-int4-complete-ffn-device-build';oldinputs=json.loads((oldrun/'source-hashes.json').read_text())
assert len(oldinputs)==411
for p,h in oldinputs.items():assert inputs[p]==h,p
assert set(inputs)-set(oldinputs)=={'src/bin/test_int4_drift_attribution.rs','src/bin/support/int4_drift_metrics.rs'}
art=run['built_artifacts']['target/release/test_int4_drift_attribution'];current=registry.read_artifact(art['path']);assert current['elf_sha256']==art['sha256']
old=registry.read_artifact(oldrun/'built/target/release/test_int4_complete_ffn')
def index(x):
    records=[r for r in x['records'] if not r['key']];out={r['path']:r for r in records};assert len(out)==len(records);return out
assert all(not r['key'] for r in current['records']+old['records'])
ci=index(current);oi=index(old);assert len(oi)==19
checks=[]
for name,r in oi.items():
    assert name in ci and (ci[name]['bytes'],ci[name]['sha256'])==(r['bytes'],r['sha256']),name
    checks.append({'path':name,'status':'equal_to_actual323067','sha256':r['sha256'],'bytes':r['bytes']})
extra={'furiosa_opt_gemma4::ops_native_fp8_p512_blocks::quantize_hidden','furiosa_opt_gemma4::ops_native_fp8_p512_blocks::quantize_ffn'}
assert set(ci)-set(oi)==extra
q020=registry.read_artifact(root/'runs/a003-q020-mse-parallel-server-build/built/target/release/server');qi=index(q020)
for name in sorted(extra):
    entry={'path':name,'current':ci[name],'source':'unchanged existing public quantizer','standalone_actual_gate':'pending'}
    if name in qi:
        assert (ci[name]['bytes'],ci[name]['sha256'])==(qi[name]['bytes'],qi[name]['sha256']),name
        entry.update(status='equal_to_q020_frozen_image',prior=qi[name])
    else:entry['status']='not_in_prior_complete_or_q020_ELF; newly_bound_existing_source'
    checks.append(entry)
print(json.dumps({'run':args.run,'source_commit':run['commit'],'input_hashes':len(inputs),'original411_unchanged':True,
 'parser_sha256':sha(parser),'current':current,'original_complete_elf_sha256':old['elf_sha256'],
 'q020_elf_sha256':q020['elf_sha256'],'comparisons':checks,'actual_attribution_pending':True},indent=2))
