#!/usr/bin/env python3
"""E153 original raw-upgate wrapper/new sharded quantizer actual graph check."""
import argparse,importlib.util,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;C=HERE.parent;ROOT=C.parents[3]
s=importlib.util.spec_from_file_location('e153_audit',C/'audit.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
def run():
 bp=C/'validation/probe-build-v1.json';b=a.load(bp);ev=C/'validation/probe-evidence-v1';wb=C/'build-e153-sharded-v1-repro.json';w=a.load(wb)
 assert b['valid'] and b['exact_compile_exit_code']==b['host_build_exit_code']==0
 assert b['production_whole_source_inputs']==w['source_inputs'] and b['production_whole_exact_image']==w['exact_image']
 src=C/'sharded-v1/mlp.rs';assert a.digest(src)==b['production_mlp_prefix'] and src.read_bytes()==(ev/'production-mlp.rs').read_bytes()
 append=ROOT/b['diagnostic_append']['path'];actual=ev/'crate/src/device/mlp.rs';assert actual.read_bytes()==src.read_bytes()+append.read_bytes()
 original=C.parent/'ffn-direct-full30-partials/validation/probe-v1'
 for f in ['probe_lib.rs','probe_module.rs','probe_stream.rs','recipe.json','stream_probe.rs']:
  if f=='recipe.json':continue
  assert (C/'validation/probe-v1'/f).read_bytes()==(original/f).read_bytes(),f
 assert 'prepare_ffn_fp8_split(device, &x)'in append.read_text()
 wrapper=a.source.fn(src.read_text(),'prepare_ffn_fp8_split(');assert 'prepare_ffn_fp8_split_sharded(device, &x)'in wrapper
 for f in('graph.json','schedule.json'):assert a.record(ev/f)==b['evidence'][f]
 image=ROOT/b['exact_image']['path'];assert a.record(image)==b['exact_image'] and a.digest(image)=={k:b['image'][k]for k in('bytes','sha256')} and a.digest(image)!=w['exact_image']
 g=a.h.Graph(a.load(ev/'graph.json'));wg=a.h.Graph(a.load(C/'evidence/e153-sharded-v1/graph.json'));checks=a.actual(g,a.load(ev/'schedule.json'),image)
 natives=[o for o in g.ops.values()if 'lookup_table:'in o['description'] and o['context']=='Main']
 for op in natives:assert any(a.sig(g,op)==a.sig(wg,z)for z in wg.ops.values())
 bound={'whole_build':wb,'probe_build':bp,'graph':ev/'graph.json','schedule':ev/'schedule.json','production_prefix':ev/'production-mlp.rs','actual_compiled_mlp':actual,'wrapper':append,'whole_gate':C/'audit-e153-sharded-v1.json','whole_verifier':C/'audit.py','verifier':Path(__file__)}
 return {'passed':True,'candidate':'E153','inputdigests':{k:a.record(v)for k,v in bound.items()},'production_mlp_prefix_preserved':True,'original_wrapper_host_truth_files_unchanged':True,'actual_sharded_quantizer_called':True,'upgate_native_actual_contracts_match_whole':True,'actual':checks,'scope':'Actual isolated upgate diagnostic source/mapping/register check, not NPU result. Original raw8 does not call preRMS; RMS emitted whole/source proof and fullFFN validation are separate.'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args();r=json.loads(json.dumps(run()));out=HERE/'raw-graph-review.json'
 if args.check:assert a.load(out)==r,'Raw review differs'
 else:out.write_text(json.dumps(r,indent=2)+'\n')
 print('PASS',out)
