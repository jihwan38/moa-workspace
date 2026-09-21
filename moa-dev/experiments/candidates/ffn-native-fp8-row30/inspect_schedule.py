"""Extract immutable exact-compile evidence without storing the large raw schedule."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def summarize(path):
    schedule = json.loads(Path(path).read_text())
    ops = schedule['instructions']
    totals = Counter()
    for op in ops:
        totals[op['tpe']] += op['lifetime']['end']-op['lifetime']['begin']
    selectors = {
        'weight_initialization': lambda i: '_weight_dense.view_mut().memset(' in i['description'],
        'scale_initialization': lambda i: '_scale_dense.view_mut().memset(' in i['description'],
        'up_gate_weight_hbm_dma': lambda i: any(n+'_weight_packed.to_dm(' in i['description'] for n in ['up','gate']),
        'up_gate_scale_hbm_dma': lambda i: i['tpe']=='DmaLoad' and any(n+'_weight_scale.to_dm' in i['description'] for n in ['up','gate']),
        'native_first_contraction':lambda i: '.begin(weights.view())' in i['description'],
        'scale_lookup_bf16':lambda i: i['tpe']=='Main' and '.begin(scales.view())' in i['description'],
        'scale_load_trf':lambda i: i['tpe']=='Sub' and '.begin(scales.view())' in i['description'],
        'second_contraction':lambda i: '.begin(partials.view())' in i['description'],
        'row_pair_switch_gather':lambda i: '.switch::<m![FfnSlice / 2, 1 # 2]' in i['description'],
    }
    stages = {name:[{'index':i['index'],'begin':i['lifetime']['begin'],'end':i['lifetime']['end'],
                   'cycles':i['lifetime']['end']-i['lifetime']['begin']} for i in ops if pred(i)] for name,pred in selectors.items()}
    return {
        'path':str(path),'sha256':sha(path),'makespan':max(i['lifetime']['end'] for i in ops),
        'instruction_count':len(ops),'instruction_duration_sum_by_type':dict(totals),
        'stages':stages,
    },schedule


def main():
    base,_ = summarize(sys.argv[1])
    candidate,raw = summarize(sys.argv[2])
    assert base['makespan']==182868 and candidate['makespan']==170176
    assert not candidate['stages']['weight_initialization'] and not candidate['stages']['scale_initialization']
    tensors={t['index']:t for t in raw['tensors']}
    storage=[]
    for op in raw['instructions']:
        if '.begin(partials.view())' in op['description']:
            t=tensors[str(op['output_tensors'][0])]
            assert 'Element: (FfnRow, 1 # 4)' in t['shape']
            assert t['size']==2*256*30*8
            gather = [i for i in raw['instructions'] if int(t['index']) in i['input_tensors']
                      and '.switch::<m![FfnSlice / 2, 1 # 2]' in i['description']]
            assert len(gather)==1
            g=gather[0]
            assert g['lifetime']['begin']>=op['lifetime']['end']
            out=tensors[str(g['output_tensors'][0])]
            assert out['size']==2*256*60*2  # physical slice-size256 includes one padded side
            storage.append({'producer_index':op['index'],'tensor_index':t['index'],
                            'tensor_shape':t['shape'],'bytes':t['size'],
                            'gather_index':g['index'],'gather_waits_for_producer':True,
                            'gather_shape':out['shape'],'gather_physical_bytes':out['size']})
    assert len(storage)==2
    print(json.dumps({'baseline':base,'candidate':candidate,
                      'makespan_saved':base['makespan']-candidate['makespan'],
                      'relative_change':candidate['makespan']/base['makespan']-1,
                      'scalar_storage_evidence':storage,
                      'limitation':'Static compiler schedule only; duration sums overlap and are not runtime cycles.'},indent=2))


if __name__=='__main__':main()
