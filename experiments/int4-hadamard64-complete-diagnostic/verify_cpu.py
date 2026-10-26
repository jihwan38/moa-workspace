#!/usr/bin/env python3
"""Read-only parity of both complete-diagnostic CPU transforms to frozen vectors."""
import argparse,hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
def main():
    ap=argparse.ArgumentParser();ap.add_argument('driver',type=Path);args=ap.parse_args()
    raw=(HERE.parent/'int4-hadamard64-diagnostic/independent-proof/vectors.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='994a071a81a0be51e6a1807bd2945c8fd228034654219a6c5569f3eeeab106e0'
    vectors=json.loads(raw)['vectors'];fields=[str(len(vectors))]
    for v in vectors:fields +=[str(v['base_channel']),*map(str,v['input_f32_bits'])]
    lines=subprocess.check_output([str(args.driver.resolve())],input=' '.join(fields),text=True).splitlines();assert len(lines)==16
    for i,(line,v)in enumerate(zip(lines,vectors)):
        n,bits=line.split();assert int(n)==i;assert list(map(int,bits.split(',')))==v['rotated_bf16_bits']
    print(json.dumps({'result':'BOTH_ROTATION_CPU_IMPLEMENTATIONS_PASS','vectors':16,'bf16_bits_each':1024,
        'driver_sha256':hashlib.sha256(args.driver.read_bytes()).hexdigest(),'scope':'fixed vectors and independent CPU stage-coordinate mapping; no actual full-chain result'},indent=2))
if __name__=='__main__':main()
