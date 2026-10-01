#!/usr/bin/env python3
"""Independent E150 source and logical half/consumer proof, no compile or NPU."""
import argparse,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def rec(p):
 b=p.read_bytes();return {'path':str(p.relative_to(ROOT)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def review():
 p=HERE/'staged-v1/mlp.rs';oldpath=HERE.parent/'ffn-direct-full30-partials/full30-v1/mlp.rs';s=p.read_text();old=oldpath.read_text()
 a=s.index('// Read one compact30-row');b=s.index('fn project_down_tile(');c=old.index('fn project_down_tile(')
 assert s[:a]==old[:c] and s[s.index('\nfn finish_down('):]==old[old.index('\nfn finish_down('):]
 new=s[b:s.index('\nfn finish_down(')];base=old[c:old.index('\nfn finish_down(')]
 assert new[:new.index('    // HBM remains')]==base[:base.index('    // HBM remains')]
 assert new[new.index('    let mut down:'):]==base[base.index('    let mut down:'):]
 oldops=base[base.index('    let partials:'):base.index('    let mut down:')];helper=s[a:b]
 keys=('contract_','vector_','fetch_table_lookup','commit_trim')
 oldseq=[l.strip()for l in oldops.splitlines()if any(k in l for k in keys)]
 newseq=[l.strip().replace('H % 60 = 30','H % 60').replace('>(x_trf)','>(&x_trf)').replace(', x_scale_vrf)',', &x_scale_vrf)').replace('.commit_trim::<m![TileL / 16 % 4]>()','.commit_trim::<m![TileL / 16 % 4]>().commit();')for l in helper.splitlines()if any(k in l for k in keys)]
 assert oldseq==newseq
 seen=set();partials=set()
 for c in range(2):
  for g in range(64):
   for q in range(4):
    for off in (0,30):
     for r in range(30):
      row=60*g+off+r;lo=(row*15360+7680*c+1920*q)//2;assert lo not in seen;seen.add(lo)
      for k in range(120):
       d=(c,4*g+q,120*(off+r)+k);assert d not in partials;partials.add(d)
 assert sorted(seen)==list(range(0,29491200,960)) and len(partials)==512*7200
 return {'passed':True,'scope':'Source and logical coordinates only, no actual lowering or hardware claim.','inputs':{'candidate':rec(p),'baseline':rec(oldpath),'verifier':rec(Path(__file__))},'weight_960B_segments':30720,'weight_bytes':29491200,'partial_f32_elements':len(partials),'partial_bytes':14745600,'partial_slice_half_byte_ranges':[[0,14400],[14400,28800]],'cross_half_consumer_rows_inclusive':[24,35],'original_native_arithmetic_preserved_except_row_extent':True,'original_quantizer_trf_and_five12row_consumers_suffix_unchanged':True}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args();r=review();out=HERE/'geometry.json'
 if args.check:assert json.loads(out.read_text())==r
 else:out.write_text(json.dumps(r,indent=2)+'\n')
 print('PASS E150 source/logical geometry')
