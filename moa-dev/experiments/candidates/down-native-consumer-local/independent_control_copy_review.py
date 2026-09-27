#!/usr/bin/env python3
import hashlib,importlib.util,json
from pathlib import Path
ROOT=Path('/home/jihwan38/moa-down-native-local');P=ROOT/'moa-dev/experiments/candidates/down-native-consumer-local'
z=importlib.util.spec_from_file_location('h',P.parent/'ffn-native-down-weighted-residual/verify_graph.py');h=importlib.util.module_from_spec(z);z.loader.exec_module(h)
def req(b,s):
 if not b:raise ValueError(s)
def record(p):
 b=p.read_bytes();return {'path':str(p.relative_to(ROOT)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def strip_copies(s):
 start=s.index('    let mut packed: DmTensor<f8e5m2, Chip, m![2], FfnQuantSlices')
 end=s.index('    (broadcast_ffn_quantized(device, &packed)',start)
 return s[:start]+'    /* exact up-term copies checked separately */\n'+s[end:]
def verify_prefix(g,ops):
 checks=[];allbytes=[]
 for term,n in enumerate([30,35,40]):
  o=ops[n];req(o['kind']=='TuExec' and 'context: Sub' in o['description'],'exactSubcopy')
  d=o['description'];req('ve_pass:' not in d and 'cast_compaction:' not in d and 'tensor_pass:' not in d,'rawcopy no arithmetic')
  req(h.offsets(h.sequence(d,'fetch_sequencer'))==list(range(480)),'copy reads480')
  req(h.offsets(h.sequence(d,'commit_projection'))==list(range(480)) and 'commit_in_size: 32B (=32EA)' in d,'copy writes480')
  producer=g.producers[o['inputs'][0]][1];req(h.offsets(h.sequence(producer['description'],'commit_projection'))==list(range(480)),'producer initializes480')
  view=g.view(o['inputs'][1]);req((view['stride_elements'],view['axis_extent'],view['window'],view['start'])==(480,3,1,term),'term exactoffset')
  req(g.root_output(view['parent'])==40,'samepackedroot')
  bounds=[480*term,480*(term+1)];allbytes.append(tuple(bounds))
  checks.append({'operator':o['name'],'context':'Sub','read_bytes_per_slice':480,'commit_packets':15,'commit_packet_bytes':32,'destination_bounds_per_slice':bounds,'root_bytes_per_slice':1440})
 h.full_coverage(allbytes,1440,'packed no overrun')
 req(g.tensor(40)['size']==1440*512,'packed allocation extent')
 return checks

label='e140-control-copy-fix';paths={'source':P/'control-copy-fix/mlp.rs','graph':P/'evidence'/label/'graph.json','schedule':P/'evidence'/label/'schedule.json','build_record':P/f'build-{label}-repro.json'}
g=h.Graph(json.loads(paths['graph'].read_text()));ops={int(o['name'].split('#O')[-1]):o for o in g.ops.values()};copies=verify_prefix(g,ops)
old=(P.parent/'ffn-native-down-weighted-residual/weighted-v3/mlp.rs').read_text();new=paths['source'].read_text();req(strip_copies(old)==strip_copies(new),'control changes outside three copies')
b=json.loads(paths['build_record'].read_text());req(b['exact_compile_exit_code']==0 and b['source_inputs_verified_in_container'],'exactcompile')
for name in ['graph','schedule']:
 r=record(paths[name]);req({k:r[k] for k in ['bytes','sha256']}==b[name],name+' exactbinding')
r=record(paths['source']);req({k:r[k] for k in ['bytes','sha256']}==b['source_inputs']['src/device/shared/mlp.rs'],'source exactbinding')
result={'passed':True,'inputdigests':{k:record(v) for k,v in paths.items()},'only_source_change':'three up-quantizer padded typed DMAs replaced by exact Sub480B copies; all down and public arithmetic byte-identical to E137 FFN/E134weighted-v3','copies':copies,'scope':'Actual emitted read/commit and indexed destination bounds. Numeric hardware and whole-kernel behavior require separate NPU validation.','limitations':['No causal numerical attribution before repaired whole result','No claim original E137 typed DMA had valid bounds merely because graded cases passed']}
Path('/tmp/moa-e140-control-copy-review.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS control exact three480B copies')
