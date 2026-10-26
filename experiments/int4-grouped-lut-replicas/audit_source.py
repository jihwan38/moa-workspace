#!/usr/bin/env python3
"""Read-only unchanged-body and full coordinate/relative-offset proof."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE='8afe741d0e9056f3205d7507e2e2fff365ec6a69'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 old=(ROOT/'src/ops_int4_grouped_activation_store.rs').read_text()
 new=(ROOT/'src/ops_int4_grouped_lut_replicas.rs').read_text()
 restored=new.replace('//! Explicit adjacent-replica LUT addressing; all rational codes/scales/packing unchanged.','//! Same per960 quantizer with one real group-leader metadata store repair.')
 restored=restored.replace('axes![Nibbles = 8, GroupWords = 120, Replica = 16];\npub type ReplicaBases = m![P512 % 16, Record];\npub type ReplicatedTable = m![Mantissa, Replica, Record];','axes![Nibbles = 8, GroupWords = 120];')
 restored=restored.replace('    table: &HbmTensor<i32, Chip, ReplicatedTable>,\n    replica_bases: &HbmTensor<i32, Chip, ReplicaBases>,','    table: &HbmTensor<i32, Chip, m![Mantissa, Record]>,')
 for rows,t in [('InputRows',4),('HiddenRows',16)]:
  begin=restored.index('    // Explicit immutable coordinate source')
  end=restored.index(f'    let index_records: DmTensor<i32, Chip, Cluster, {rows}, m![P512 % {t}, 1 # 8]>',begin)
  block=restored[begin:end]
  assert f'let replica_dm: DmTensor<i32, Chip, Cluster, {rows}, m![P512 % {t}, Record]>' in block
  assert 'replica_bases.to_dm(&mut device.tdma)' in block
  assert f'.tile::<m![Record], 1, m![P512 % {t}, Record = 1 # 8]>(0)' in block
  restored=restored[:begin]+restored[end:]
 restored=restored.replace('.vector_logic(LogicBinaryOpI32::LeftShift, 9i32)\n        .vector_clip(ClipBinaryOpI32::AddFxp, &replica_vrf)','.vector_logic(LogicBinaryOpI32::LeftShift, 5i32)')
 restored=restored.replace('// are relative offsets512*mantissa+32*replica,0..65504, not device addresses.','// are relative table byte positions0..4064, not fabricated device addresses.')
 restored=restored.replace('.gather::<m![Mantissa, Replica], m![Record]>()','.gather::<m![Mantissa], m![Record]>()')
 assert restored==old,'Unaccounted device body change'
 paths=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE,'src','Cargo.toml','rust-toolchain.toml'],cwd=ROOT,text=True).splitlines()
 for p in paths:
  original=subprocess.check_output(['git','show',f'{BASE}:{p}'],cwd=ROOT)
  current=(ROOT/p).read_bytes()
  if p=='src/lib.rs':assert current==original+b'\npub mod ops_int4_grouped_lut_replicas;\n'
  else:assert current==original,p
 proof=[]
 for groups in [4,16]:
  seen=set();counts=[0]*16;lookup_checks=0
  for c in range(2):
   for slice in range(256):
    g,r=divmod(slice,256//groups)
    for j in range(groups):
     p=256*c+groups*r+j
     replica=4*(r%4)+j if groups==4 else j
     assert replica==p%16
     slot=(c*groups+g)*256+groups*r+j
     assert slot not in seen;seen.add(slot);counts[replica]+=1
     for m in range(128):
      offset=512*m+32*replica
      assert offset==32*(16*m+p%16)
      assert offset%32==0 and 0<=offset<=65504 and offset+32<=65536
      assert divmod(offset//32,16)==(m,p%16)
      lookup_checks+=1
  assert seen==set(range(512*groups)) and len(set(counts))==1
  proof.append({'groups':groups,'unique_token_groups':len(seen),'all_mantissa_offset_checks':lookup_checks,'replica_counts':counts,'scope':'Independent integer-coordinate proof of declared mapping; not proof of actual device DMA values.'})
 plan=json.loads((ROOT/'experiments/int4-grouped-lut-replicas/plan.json').read_text())
 for p,h in plan['dependencies'].items():assert sha(ROOT/p)==h,p
 result={'base':BASE,'old_source_inputs_unchanged_except_lib_export':len(paths),'exact_reverse_delta_PASS':True,'mapping':proof,'dependencies':plan['dependencies'],'source_hashes':{p:sha(ROOT/p)for p in ['src/ops_int4_grouped_activation_store.rs','src/ops_int4_grouped_lut_replicas.rs','src/bin/support/int4_grouped_activation_fixture.rs','src/bin/support/rational_a4_oracle.rs','src/bin/support/int4_lut_replicas_fixture.rs','src/bin/test_int4_grouped_lut_replicas.rs']},'no_claim':'No device values, runtime gain, causal contention, whole FFN or model quality established.'}
 print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
