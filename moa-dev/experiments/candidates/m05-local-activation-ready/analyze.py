#!/usr/bin/env python3
"""Recompute complete schedules; only NPU screen reports measure performance."""
import importlib.util,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
spec=importlib.util.spec_from_file_location('m03_analysis',HERE.parent/'m03-producer-local-ffn/analyze.py')
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
digest=common.digest
PATHS={'E104':HERE.parent/'e104-weight-path/evidence/profile-v1/schedule.json',
       'M03':HERE.parent/'m03-producer-local-ffn/evidence/m03-native-local-v1/schedule.json',
       'M05':HERE/'evidence/m05-local-prepare-v1/schedule.json'}
def analyze():
    result={k:common.analyze(p) for k,p in PATHS.items()}
    for name,path in PATHS.items():
        s=json.loads(path.read_text())
        result[name]['input_preparation']=[{'index':n['index'],'type':n['tpe'],**n['lifetime'],'description':n['description']} for n in s['instructions'] if n['tpe']=='Main' and ('src/device/shared/rmsnorm.rs:95:' in n['description'] or '0x7fff_ffff' in n['description'] or 'FpBinaryOp::DivF, &scale_vrf' in n['description'] or ('CustomBroadcast { ring_size: 32 }' in n['description'] and 'FfnTerm' in n['description']))]
    result['coordinates']=common.coordinates()
    return result
if __name__=='__main__':
    r=analyze();(HERE/'static-analysis.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:{f:r[k][f] for f in ['makespan','dma_count','pre_rms_to_amax_gap']} for k in PATHS},indent=2))
