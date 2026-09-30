#!/usr/bin/env python3
"""Prepare the original full2 collector config from a newly built package and explicit job id."""
import argparse, hashlib, json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4]
def digest(raw):return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build-label',default='e144-boundary-v1');p.add_argument('--job',type=int,required=True);p.add_argument('--job-name',default='beef-e144-boundary-v1');a=p.parse_args()
assert a.job>0
bp=HERE/f'build-{a.build_label}.json';b=json.loads(bp.read_text());assert b['status']=='READY_NOT_NPU_TESTED'
selected=ROOT/b['selected_build'];assert digest(selected.read_bytes())==b['selected_build_digest']
assert b['production_inputs_match'] and b['host_check']['full_input_sha256_passed']==38
image=ROOT/b['image']['path'];assert digest(image.read_bytes())=={k:b['image'][k] for k in ('bytes','sha256')}
base=HERE.parent.relative_to(ROOT)
c={'job_id':a.job,'build_record':str(bp.relative_to(ROOT)),'output':str(base/f'boundary-suite/result-{a.job}.json'),'log':str(base/f'boundary-suite/arena-{a.job}.log'),'archive':f'moa-dev/experiments/artifacts/qkv-fullh-weight-switch/boundary-job-{a.job}.tar','expected_job_name':a.job_name,'package':b['package'],'image':b['image']['path'],'image_sha256':b['image']['sha256'],'cpu':b['cpu_reference_source'],'variant':json.loads(selected.read_text())['label']}
with (HERE/'collector-config.json').open('x') as f:json.dump(c,f,indent=2);f.write('\n')
print('Config prepared; no request or submission executed')
