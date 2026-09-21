"""Summarize an exact compile; includes physical padding in tensor sizes."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


def main():
    path=Path(sys.argv[1]);s=json.loads(path.read_text())
    ops=s['instructions'];tensors={t['index']:t for t in s['tensors']}
    totals=Counter()
    for op in ops:totals[op['tpe']]+=op['lifetime']['end']-op['lifetime']['begin']
    stages={}
    selectors=[('native_first_contraction','.begin(weights.view())'),('scale_bf16_lookup','.main.begin(scales.view())'),('scale_trf_load','.sub.begin(scales.view())'),('second_contraction','.begin(partials.view())'),('gather256','.switch::<Slice, m![FfnRow, FfnSlice]>'),('up_weight_hbm','up_weight_packed.to_dm('),('gate_weight_hbm','gate_weight_packed.to_dm('),('up_scale_hbm','up_weight_scale.to_dm('),('gate_scale_hbm','gate_weight_scale.to_dm('),('up_typed_gather','up.to_dm('),('gate_typed_gather','gate.to_dm(')]
    for name,marker in selectors:
        stages[name]=[{'index':i['index'],'type':i['tpe'],'begin':i['lifetime']['begin'],'end':i['lifetime']['end'],'cycles':i['lifetime']['end']-i['lifetime']['begin']} for i in ops if marker in i['description']]
    storage=[]
    for op in ops:
        if '.begin(partials.view())' not in op['description']:continue
        t=tensors[str(op['output_tensors'][0])]
        assert t['size']==122880 and 'Element: (FfnRow, 1 # 4)' in t['shape']
        consumers=[i for i in ops if int(t['index']) in i['input_tensors'] and '.switch::<Slice, m![FfnRow, FfnSlice]>' in i['description']]
        assert len(consumers)==1;g=consumers[0]
        assert g['lifetime']['begin']>=op['lifetime']['end']
        output=tensors[str(g['output_tensors'][0])]
        assert 'Slice: 1 # 256' in output['shape']
        assert output['size']==2*256*30*256*2
        storage.append({'second_dpe':op['index'],'scalar_tensor':t,'gather':g['index'],'gather_tensor':output,'gather_waits_for_scalar_commit':True})
    assert len(storage)==2
    summary={'makespan':max(i['lifetime']['end'] for i in ops),'instruction_count':len(ops),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'instruction_duration_sum_by_type':dict(totals),'stages':stages,'scalar_storage':storage,'row2_row30_makespan':170176,'e39_tailzero_row256_makespan':175459,'warning':'Static schedule only; overlap means stage durations are not additive runtime cycles. Gather Slice1#256 has one live slice and256 physical slice slots.'}
    assert summary['makespan']==179710
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
