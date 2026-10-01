#!/usr/bin/env python3
"""CPU-only source/coordinate preparation. Does not compile or invoke NPU."""
import argparse, difflib, hashlib, json
from pathlib import Path
P=Path(__file__).resolve().parent
DEFAULT=Path('/home/jihwan38/moa-ffn-down-weight-stages/moa-dev/experiments/candidates/ffn-down-weight-stages/staged-v1/mlp.rs')
EXPECTED='813aa7068376a41833723748ba31bf43008972bb5ff9b4e1e7cbc3d185a23530'
def digest(b): return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def main():
 a=argparse.ArgumentParser();a.add_argument('--source',type=Path,default=DEFAULT);a.add_argument('--check',action='store_true');args=a.parse_args()
 old=args.source.read_bytes();assert hashlib.sha256(old).hexdigest()==EXPECTED
 s=old.decode();before=s
 oldload='    let scales: DmTensor<f8e4m3, Chip, UpGateCluster, DownColumns, m![H % 60, TileL / 16 % 120]> = scale.to_dm(&mut device.tdma);'
 newload='''    // Source slice 4*g+b owns rows 60*g+4*r+b, r<15, with each
    // cluster's full 480-byte scale row. No intermediate DM scatter.
    let scales: DmTensor<f8e4m3, Chip, UpGateCluster, m![H / 60, H % 4], m![H / 4 % 15, TileL / 16]> = scale.to_dm(&mut device.tdma);'''
 assert s.count(oldload)==1;s=s.replace(oldload,newload)
 oldread='''            .begin(scales.view().tile::<m![H % 60], 12, m![H % 60 = 12 # 60, TileL / 16 % 120]>(12 * i))
            .fetch::<m![H % 60 = 12], m![TileL / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 12, TileL / 128 % 15], m![TileL / 16 % 8]>()'''
 newread='''            .begin(scales.view().tile::<m![H / 4 % 15], 3, m![H / 4 % 15 = 3 # 15, TileL / 16]>(3 * i))
            .fetch::<m![TileL / 1920, H / 4 % 15 = 3], m![TileL / 16 % 120]>()
            .fetch_cast::<f32>()
            // Actual InterTranspose order is (row-high3,row-low4).
            // The four old source slices supply 12 consecutive output rows.
            .switch::<DownColumns, m![H % 60 = 12]>(SwitchConfig::InterTranspose { slice1: 4, slice0: 1, time0: 3 })
            .collect::<m![H % 60 = 12, TileL / 128 % 15], m![TileL / 16 % 8]>()'''
 assert s.count(oldread)==1;s=s.replace(oldread,newread)
 # Every HBM source byte is read once; every destination scale element equals
 # the old DownColumns coordinate. These are intended topology coordinates,
 # not a replacement for inspecting emitted lowering/physical descriptors.
 seen=bytearray(3840*960);dest=set();segments=0
 for c in range(2):
  for g in range(64):
   for b in range(4):
    for r in range(15):
     h=60*g+4*r+b;start=h*960+c*480
     assert start>=0 and start+480<=len(seen)
     assert not any(seen[start:start+480]);seen[start:start+480]=b'\1'*480;segments+=1
   for i in range(5):
    for q in range(4):
     for a3 in range(3):
      for b in range(4):
       row=4*a3+b;h=60*g+12*i+row
       src_h=60*g+4*(3*i+a3)+b;assert src_h==h
       # Four 120-element packets from old sources b give one consumer row12.
       for u in range(120):
        source=h*960+c*480+120*q+u
        expect=(60*g+12*i+row)*960+c*480+120*q+u
        assert source==expect
       key=(c,4*g+q,12*i+row);assert key not in dest;dest.add(key)
 assert all(seen) and segments==7680 and len(dest)==512*60
 b=s.encode();report={'status':'source_preparation_cpu_mapping_passed_target_not_compiled','source':{'path':str(args.source),**digest(old)},'candidate':digest(b),'changed_regions':2,'unchanged_arithmetic':True,'coverage':{'hbm_bytes':len(seen),'row_segments':segments,'source_row_segment_bytes':480,'destination_row_packets':len(dest),'destination_packet_elements':120,'destination_rows_per_slice':60,'consumer_streams':5,'consumer_rows':12},'costs':{'hbm_bytes_unchanged':3686400,'dm_payload_bytes_unchanged':3686400,'f32_switch_logical_payload_added':14745600,'f32_switch_logical_payload_added_mib':14.0625,'vrf_each_consumer_bytes_unchanged':5760,'new_dm_copy_stages':0,'sub_switch_streams_added':5},'unverified':['named tile output factoring into H%60=12','480-byte F32 packet Switch lowering','actual HBM packet extent/pitch and live allocations','bitmap/config and all VRF lifetimes','NPU numerical and whole-path performance']}
 outputs={'mlp.rs':b,'source.diff':''.join(difflib.unified_diff(before.splitlines(True),s.splitlines(True),fromfile='E150/staged-v1/mlp.rs',tofile='prepared-scale-switch/mlp.rs')).encode(),'source-contract.json':(json.dumps(report,indent=2)+'\n').encode()}
 for name,data in outputs.items():
  p=P/name
  if args.check: assert p.read_bytes()==data,name
  else:p.write_bytes(data)
 print(json.dumps({'passed':True,'check':args.check,'candidate':report['candidate'],'coverage':report['coverage']}))
if __name__=='__main__':main()
