#!/usr/bin/env python3
"""Verify the frozen schedule's actual tensor dependency graph and write times."""
from pathlib import Path
import hashlib,json,sys
p=Path(__file__).parent
schedule=Path(sys.argv[1] if len(sys.argv)>1 else '/tmp/moa-ffn-native-fp8-tail-zero-schedule.json')
e=json.loads((p/'evidence.json').read_text())
assert hashlib.sha256(schedule.read_bytes()).hexdigest()==e['schedule_sha256']
j=json.loads(schedule.read_text());ins={i['index']:i for i in j['instructions']};prod={}
for i in j['instructions']:
    for t in i['output_tensors']:
        assert t not in prod
        prod[t]=i['index']
def parents(idx, seen=None):
    seen=set() if seen is None else seen
    for t in ins[idx]['input_tensors']:
        if t in prod and prod[t] not in seen:
            parent=prod[t];seen.add(parent);parents(parent,seen)
    return seen
checks=[]
for label,init,copy,write,consumer in [('gate_weight','25','57','183','300'),('up_weight','149','161','319','374'),('gate_scale','121','128','189','223'),('up_scale','164','173','322','364')]:
    ps=parents(consumer)
    assert {init,copy,write} <= ps
    assert ins[copy]['lifetime']['begin'] >= ins[init]['lifetime']['end']
    assert ins[consumer]['lifetime']['begin'] >= max(ins[copy]['lifetime']['end'],ins[write]['lifetime']['end'])
    checks.append({'buffer':label,'zero_memset':{'id':init,**ins[init]['lifetime']},'zero_copy':{'id':copy,**ins[copy]['lifetime']},'valid30_write':{'id':write,**ins[write]['lifetime']},'first_consumer':{'id':consumer,**ins[consumer]['lifetime']},'all_three_producers_in_consumer_ancestry':True})
(p/'dependency-evidence.json').write_text(json.dumps(checks,indent=2)+'\n')
print('PASS: all four consumers depend on zero initialization, tail copy and valid30 write, and start after both writes')
