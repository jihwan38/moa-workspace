#!/usr/bin/env python3
"""Logical geometry only: root provisional O ingress; no compiler/NPU proof."""
import argparse,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[3]
source=HERE/'ingress-v1/projection.rs'
seen=set();dst=set();stage_counts={}
for offset in (0,60):
 n=0
 for c in range(2):
  for g in range(16):
   for a in range(4):
    for b in range(4):
     for r in range(15):
      h=1920*c+120*g+offset+4*r+b
      for q in range(4):
       src_slice=16*g+4*a+b;src_time=15*q+r
       dst_slice=16*g+4*a+q;dst_time=4*r+b
       k=1024*a+256*q
       assert 0<=h<3840 and 0<=k<=3840
       assert src_time//15==q and src_time%15==r
       assert dst_time//4==r and dst_time%4==b
       assert 1920*c+120*(dst_slice//16)+offset+dst_time==h
       assert 256*(dst_slice%16)==k
       assert (h,k)not in seen;seen.add((h,k))
       d=(c,dst_slice,offset+dst_time);assert d not in dst;dst.add(d)
       # Packet's u=0..255 is unchanged by InterTranspose; address increments by one.
       lo=h*4096+k;assert 0<=lo and lo+256<=3840*4096
       assert r*1024+q*256+256<=(r+1)*1024
       n+=1
 stage_counts[str(offset)]=n
assert seen=={(h,k)for h in range(3840)for k in range(0,4096,256)}
s=source.read_bytes()
required=['m![H / 120 % 16, Qs / 1024, H % 120 = 60 % 4]','m![H % 120 = 60 / 4, Qs % 1024]','.fetch::<m![Qs / 256 % 4, H % 120 = 60 / 4], m![Qs % 256]>()','InterTranspose { slice1: 4, slice0: 1, time0: 15 }']
assert all(x in s.decode()for x in required)
out={'passed':True,'scope':'Logical mapping/source-text review only; exact descriptor, lowering, numerical and NPU gates remain required.','provisional_source':{'path':str(source.relative_to(ROOT)),'bytes':len(s),'sha256':hashlib.sha256(s).hexdigest()},'packets':len(seen),'packet_bytes':256,'weight_bytes':len(seen)*256,'per_stage_packets':stage_counts,'source_coordinate':'h=1920*c+120*g+offset+4*r+b; k=1024*a+256*q+u','switch':'slice16*g+4*a+b,time15*q+r -> slice16*g+4*a+q,time4*r+b; u unchanged','logical_ingress_contiguous_region_bytes':1024,'emitted_descriptor_packet_bytes':'unmeasured; E144 fullH3840 precedent still encodes 256B packets','additional_work':'Two Main Switch stages consume all 15MiB once; strided packet fetch changes. No additional full-weight DM materialization in this source.'}
out['verifier']={'path':str(Path(__file__).resolve().relative_to(ROOT)),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args();target=HERE/'geometry.json'
if args.check: assert json.loads(target.read_text())==out
else: target.write_text(json.dumps(out,indent=2)+'\n')
print('PASS E149 logical61440packet geometry')
