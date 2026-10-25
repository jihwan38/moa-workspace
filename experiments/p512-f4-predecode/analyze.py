#!/usr/bin/env python3
"""Reproduce all three complete resource schedules and preserve the declared gate."""
from pathlib import Path
import json,hashlib,collections
tree=Path(__file__).resolve().parents[2]; root=tree.parent
out={'source':'eb6af906','scope':'compiler resources only, no hardware or modelquality claim','graphs':{}}
for name,run in [('candidate','r1291-p512-f4-predecode-resource'),('original','r1293-p512-f4-predecode-original-resource'),('inline','r1294-p512-f4-predecode-inline-resource')]:
    d=root/'runs'/run;record=json.loads((d/'run.json').read_text());assert record['result']=='command_succeeded' and record['build_inputs_unchanged']
    hashes=json.loads((d/'source-hashes.json').read_text())
    for rel,want in hashes.items(): assert hashlib.sha256((tree/rel).read_bytes()).hexdigest()==want,rel
    p=Path(__file__).with_name(name+'-schedule.json');j=json.loads(p.read_text());end=0;sums=collections.Counter()
    for key,events in j['timeline']['operators'].items():
        for event in events:
            life=event['lifetime'];end=max(end,life['end']);sums[j['description']['operators'][key]['kind']]+=life['end']-life['begin']
    stack=[e['allocation'] for entries in j['timeline']['tensors'].values() for e in entries if 'DramStack' in e['allocation']['buffer_type']];assert not stack
    out['graphs'][name]={'estimated_cycles':end,'overlapping_operator_duration_sums':dict(sums),'dram_stack':stack,'schedule_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_hashes_verified':len(hashes),'run':run}
c,o,i=[out['graphs'][n]['estimated_cycles'] for n in ['candidate','original','inline']]
out['gain_vs_original']=1-c/o;out['gain_vs_inline']=1-c/i
out['resource_gate_passed']=out['gain_vs_original']>=.1 and c<i
out['decision']='PASS resource only' if out['resource_gate_passed'] else 'HOLD: not faster than held inline F4; do not start NPU or host integration'
Path(__file__).with_name('result.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='graphs'}))
