#!/usr/bin/env python3
"""Read exact immutable compile schedules; estimates do not establish hardware speed."""
import collections,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
TREE=HERE.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(name):
 p=HERE/f'{name}-a1.json';x=json.loads(p.read_text());t=x['timeline'];d=x['description']
 tensors=[(k,v)for k,vs in t['tensors'].items()for v in vs]
 trfs=[{'id':k,'shape':d['tensors'][k]['properties']['shape'],**v}for k,v in tensors if v['allocation']['buffer_type']=='Trf']
 return {'file':p.name,'sha256':sha(p),'estimated_end':max(v['lifetime']['end']for vs in t['operators'].values()for v in vs),
 'dram_stack':[k for k,v in tensors if v['allocation']['buffer_type'].startswith('DramStack')],
 'trf_epochs':len(trfs),'trfs':trfs,'command_counts':{kind:sum(len(t['operators'].get(k,[]))for k,op in d['operators'].items()if op['kind']==kind)for kind in ['rngd_cmd.tu_exec','rngd_cmd.sto_trf','rngd_cmd.sto_vrf','dma.dto_s','dma.sto_d','dma.sto_s']}}
parts={name:read(name)for name in ['control','core','startup','quant']}
assert all(not p['dram_stack']for p in parts.values())
assert parts['core']['trf_epochs']==1 and parts['control']['trf_epochs']==32
payload=262144*3840+262144*4
out={'scope':'Compiler raw estimates only; no NPU speed or old-LM equality claim.',
 'parts':parts,'complete_estimate_change':(parts['core']['estimated_end']+parts['quant']['estimated_end'])/parts['control']['estimated_end']-1,
 'new_persistent_bytes':payload,'runtime_input_scratch_bytes':16*3840+16*8*4,
 'source_hashes':{name:sha(TREE/name)for name in ['src/ops_b16_lm_fp8.rs','src/ops_b16_lm_storegroup.rs','src/ops_query_b16_native_fp8.rs']}}
(HERE/'resource-audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items()if k not in ['parts','source_hashes']},indent=2))
