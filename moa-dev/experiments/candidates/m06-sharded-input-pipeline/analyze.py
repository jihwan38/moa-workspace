#!/usr/bin/env python3
"""Read emitted dataflow and retain static times separately from NPU screens."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def digest(raw):return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def analyze(path,sharded):
    s=json.loads(path.read_text());ins=s['instructions'];ts={str(t['index']):t for t in s['tensors']}
    main=[n for n in ins if n['tpe']=='Main'];dma=[n for n in ins if n['tpe'].startswith('Dma')]
    norm=[n for n in main if f"src/device/shared/rmsnorm.rs:{261 if sharded else 95}:" in n['description']]
    maxima=[n for n in main if '0x7fff_ffff' in n['description']]
    quant=[n for n in main if 'FpBinaryOp::DivF, &scale_vrf' in n['description']]
    broadcasts=[n for n in main if 'CustomBroadcast { ring_size: 32 }' in n['description'] and 'FfnTerm' in n['description']]
    fp8=[n for n in main if 'fetch_table_lookup::<f8e' in n['description']]
    dots=[n for n in main if 'contract_outer' in n['description']]
    weights=[n for n in dma if n['tpe']=='DmaLoad' and 'src/device/shared/mlp.rs:' in n['description'] and any(x in n['description'] for x in ['weight_packed.to_dm','weight.to_dm'])]
    packing=[n for n in dma if '.to_dm_view' in n['description'] and 'FfnTerm' in n['description']]
    assert len(norm)==len(maxima)==len(broadcasts)==1
    assert len(quant)==len(fp8)==len(dots)==len(weights)==3
    def detail(nodes):return [{'index':n['index'],'type':n['tpe'],**n['lifetime'],'description':n['description'],'outputs':[ts.get(str(i), {'unlisted_view_index':str(i),'allocated_size':None}) for i in n['output_tensors']]} for n in nodes]
    return {'schedule':digest(path.read_bytes()),'instruction_count':len(ins),'dma_count':len(dma),
            'makespan':max(n['lifetime']['end'] for n in ins),'pre_rms':detail(norm),'local_amax':detail(maxima),
            'pre_rms_to_amax_gap':maxima[0]['lifetime']['begin']-norm[0]['lifetime']['end'],
            'quantization':detail(quant),'input_broadcast':detail(broadcasts),'packing_dma':detail(packing),
            'weight_loads':detail(weights),'fp8_expansion':detail(fp8),'dots':detail(dots),
            'fp8_logical_write_bytes':3*15360*3840,
            'limits':'Static schedule; an executable may still fail on NPU. Not a measured internal latency/stall model.'}
def result():
    paths={'E104':(HERE.parent/'e104-weight-path/evidence/profile-v1/schedule.json',False),
           'M03':(HERE.parent/'m03-producer-local-ffn/evidence/m03-native-local-v1/schedule.json',False),
           'M05':(HERE.parent/'m05-local-activation-ready/evidence/m05-local-prepare-v1/schedule.json',False),
           'M06_v4':(HERE/'evidence/m06-sharded-packed-v4/schedule.json',True),
           'M06_v5':(HERE/'evidence/m06-sharded-dma-v5/schedule.json',True)}
    return {k:analyze(p,sharded) for k,(p,sharded) in paths.items()}
if __name__=='__main__':
    r=result();(HERE/'static-analysis.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:{'makespan':v['makespan'],'dma_count':v['dma_count'],'pre_rms_to_amax_gap':v['pre_rms_to_amax_gap'],'quantization_ends':[n['end'] for n in v['quantization']],'broadcast_end':v['input_broadcast'][0]['end'],'packing_dmas':len(v['packing_dma'])} for k,v in r.items()},indent=2))
