import hashlib,json,re
from pathlib import Path
p=Path('/home/jihwan38/moa-down-native-local/moa-dev/experiments/candidates/down-native-consumer-local');v=p/'evidence/e140-local-v1/failure.visa.txt';s=v.read_text();records=[]
for m in re.finditer(r'StoVrf \{ args:.*?result: TensorVar\((\d+)\) \}',s):
 b=m[0];tid=int(m[1]);shape=re.search(r'vrf_shape: (.*?)vrf_element_type: (\w+)',b)
 assert shape
 if tid in [559,568,602,610,618,626,634,657] or 'FfnRow = 6' in shape[1]:
  users=[]
  for x in re.finditer(r'inputs: \[([^\]]*)\], output: (.{0,120})',s):
   ids=[int(z) for z in re.findall(r'TensorVar\((\d+)\)',x[1])]
   if tid in ids:users.append({'inputs':ids,'output_prefix':x[2].split('}),')[0]})
  records.append({'tensor_var':tid,'vrf_shape':shape[1].strip().rstrip(','),'type':shape[2],'direct_tu_users':users})
assert {559,568,602,657}.issubset({x['tensor_var'] for x in records})
def rec(f):b=f.read_bytes();return {'path':str(f),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
r={'scope':'preallocation vISA shape and data-dependency audit only, no scheduled lifetimes or physical addresses exist','inputdigests':[rec(v),rec(p/'local-v1/mlp.rs'),rec(p/'build-e140-local-v1-repro.json')],'observations':records,'allocation_request_bytes_total':2949120,'allocation_request_bytes_per_slice':5760,'failed_request_identifies_unique_buffer':False,'repair_selection':{'consumer_k_unchanged':1920,'quant_chunk':480,'chunks':4,'reason':'960 residual3840 + observed5760 operand exceed8192 even before scalar.480 residual1920 restores E134 operand footprint and admits observed5760 + scalar32 + term96 + longglobal32=7840.','actual_cause_or_compile_success_proven':False,'new_packed_offsets':'1920*term +480*chunk, exact480B E5 commits; no typedDMA into paddedviews','remaining_risks':['postRMS gamma1920 may require rescheduling','compiler may overlap different chunk VRFs despite lexical blocks','extra quantizer tasks, IndexWrite dependencies and narrowview lowering may offset readiness gain']}}
Path('/tmp/moa-e140-vrf-failure-review.json').write_text(json.dumps(r,indent=2)+'\n');print('PASS preallocation evidence extraction; selected concrete capacity repair remains uncompiled')
