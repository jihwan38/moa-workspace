"""Reproduce model evidence from parent-frozen local runs; no submission or mutation."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = ROOT / 'runs'
FAMILIES = {
 'Tail384': {'source': '9b1e19cd9977244490a160a2c15dd7bebaa1f0be', 'binary': 'compare_tail384', 'worktree':'tail384-model',
  'sha256': '15b0a9415a8d5acef90dc247d0df966608a6cee7430061fde73a94cdc025218e',
  'prefix': 'TAIL384', 'runs': [844,847,848,852,869,870,871,872,873,874,875]},
 'ThreeOwner512': {'source': '180202c32670bb5f48792d6d59ee3f37298b5231', 'binary': 'compare_three_owner512', 'worktree':'three-owner512-model',
  'sha256': '41b6c454ccafe300bfc899f0d58474885c65d6e27251c066319065c867d1ecd2',
  'prefix': 'THREE512', 'runs': [857,866,867,868]},
}
def sha(b): return hashlib.sha256(b).hexdigest()
def fields(line): return dict(re.findall(r'(\w+)=([^\s;]+)', line))
result = {'scope': 'Same-model all-vocabulary and raw-cache gates; sequential completed prepare+layers+finish timings, excluding downloads, interleaves and cache scans. No HTTP/Worker performance claim.',
 'worker_source': '38493b8', 'worker_host_tests': 111, 'families': {}}
for family, config in FAMILIES.items():
 out = {k:v for k,v in config.items() if k not in ['runs','prefix']}
 out['runs'] = []
 canonical = None
 for run in config['runs']:
  dirs = list(RUNS.glob(f'r{run}-*')); assert len(dirs) == 1, dirs
  p = dirs[0]
  meta = json.loads((p/'run.json').read_text())
  status = json.loads((p/'arena-status.json').read_text())
  assert status['status'] == 'SUCCEEDED' and status['exit_code'] == 0, p
  log = (p/'arena.log').read_text()
  assert f'PASS {family} ' in log, p
  manifest = json.loads((p/'source-hashes.json').read_text())
  if canonical is None:
   canonical = manifest
   tracked = 0; generated = []
   for name, expected in manifest.items():
    blob = subprocess.run(['git','show',f"{config['source']}:{name}"], cwd=HERE, capture_output=True)
    if blob.returncode == 0:
     data = blob.stdout; tracked += 1
    else:
     data = (ROOT/config['worktree']/name).read_bytes()
     generated.append(name)
     target = HERE/(family.lower()+'-generated')/name
     target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(data)
    assert sha(data) == expected, (family,name)
   out['tracked_source_files_verified_against_build_commit'] = tracked
   out['untracked_generated_files_verified_against_run_and_preserved'] = generated
   (HERE/f'{family.lower()}-source-hashes.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
  assert manifest == canonical, (family,p,'source manifest changed')
  artifact = Path(meta['submitted_artifact']['path'])
  assert artifact.name == config['binary'] and sha(artifact.read_bytes()) == config['sha256'], p
  exact = [fields(x) for x in log.splitlines() if x.startswith(config['prefix']+'_EXACT ')]
  assert exact and all(int(x['bitdiff']) == 0 and int(x['nonfinite']) == 0 for x in exact), p
  cache = [fields(x) for x in log.splitlines() if x.startswith('KV_ALL_BYTES_EXACT ')]
  case_lines = [x for x in log.splitlines() if x.startswith(config['prefix']+'_CASE_PASS ')]
  assert len(case_lines) == 1, p
  case = fields(case_lines[0]); case = {k:float(v) if k.endswith('_ms') else v for k,v in case.items()}
  reset = fields(next(x for x in log.splitlines() if x.startswith(f'PASS {family} ')))
  out['runs'].append({'run':p.name,'job':status['id'],'submitted_commit':meta['commit'],
   'artifact_path':str(artifact.relative_to(ROOT)), 'artifact_sha256_verified': config['sha256'],
   'log_sha256':sha((p/'arena.log').read_bytes()),'source_manifest_sha256':sha((p/'source-hashes.json').read_bytes()),
   'logit_comparisons':len(exact),'logit_values':sum(int(x['values']) for x in exact),
   'raw_cache_comparisons':len(cache),'raw_cache_bytes':sum(int(x['compared_bytes']) for x in cache),
   'case':case,'reset':{k:float(v) for k,v in reset.items() if k.endswith('_ms')}})
 out['manifest_files_verified'] = len(canonical)
 out['total_logit_values'] = sum(x['logit_values'] for x in out['runs'])
 out['total_raw_cache_bytes'] = sum(x['raw_cache_bytes'] for x in out['runs'])
 result['families'][family] = out
result['total_runs'] = sum(len(v['runs']) for v in result['families'].values())
result['total_logit_values'] = sum(v['total_logit_values'] for v in result['families'].values())
result['total_raw_cache_bytes'] = sum(v['total_raw_cache_bytes'] for v in result['families'].values())
(HERE/'model-results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='families'},indent=2))
for k,v in result['families'].items():
 print(k, 'runs',len(v['runs']),'sources',v['manifest_files_verified'],
  'logits',v['total_logit_values'],'cache_bytes',v['total_raw_cache_bytes'])
