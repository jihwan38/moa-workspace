#!/usr/bin/env python3
import hashlib,importlib.util,json
from pathlib import Path
ROOT=Path('/home/jihwan38/moa-ffn-direct-copy-repair');P=ROOT/'moa-dev/experiments/candidates/ffn-independent-direct-copy-repair';checker=P/'independent-audit.py'
z=importlib.util.spec_from_file_location('whole',checker);m=importlib.util.module_from_spec(z);z.loader.exec_module(m)
e=P/'validation/probe-evidence-v1';image=ROOT/'moa-dev/experiments/artifacts/ffn-independent-direct-copy-repair/e141-direct-probe-v1/kernels/furiosa-opt-gemma4/furiosa_opt_gemma4::stream_probe::upgate.bin'
g=m.h.Graph(json.loads((e/'graph.json').read_text()));s=json.loads((e/'schedule.json').read_text());r=m.prefix_and_bitmap(g,s,image)
source=e/'production-mlp.rs';frozen=P/'direct-six-row-copy-v1/mlp.rs';assert source.read_bytes()==frozen.read_bytes()
paths={'source':source,'frozen_production':frozen,'graph':e/'graph.json','schedule':e/'schedule.json','exact_image':image,'checker':checker}
out={'passed':True,'inputdigests':{k:m.record(p) for k,p in paths.items()},'prefix_copy_and_bitmap':r,'scope':'Actual repaired production prefix, three compact480B writes, disjoint1440B term assembly, live ring32 control bitmap outside write bounds. No oracle arithmetic or runtime success inferred before NPU.'}
Path('/tmp/moa-e141-probe-copy-review.json').write_text(json.dumps(out,indent=2)+'\n');print('PASS',r['bitmap']['physical_bounds_per_slice'])
