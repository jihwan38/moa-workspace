#!/usr/bin/env python3
"""Read-only fixed Fraction/vector parity for supplied CPU executable. No build."""
import argparse, hashlib, json, subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
SHA='994a071a81a0be51e6a1807bd2945c8fd228034654219a6c5569f3eeeab106e0'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('driver',type=Path);args=ap.parse_args()
    raw=(HERE/'independent-proof/vectors.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==SHA
    vectors=json.loads(raw)['vectors'];fields=[str(len(vectors))]
    for v in vectors:fields += [str(v['base_channel']),*map(str,v['input_f32_bits'])]
    result=subprocess.run([str(args.driver.resolve())],input=' '.join(fields),text=True,capture_output=True,check=True)
    lines=result.stdout.splitlines();assert len(lines)==len(vectors)
    for i,(line,v) in enumerate(zip(lines,vectors)):
        index,signs,f32,bf16=line.split();assert int(index)==i
        assert list(map(int,signs.split(',')))==v['signs'],('sign',i)
        assert list(map(int,f32.split(',')))==v['rotated_f32_bits'],('F32',i)
        assert list(map(int,bf16.split(',')))==v['rotated_bf16_bits'],('BF16',i)
    print(json.dumps({'result':'FIXED_VECTOR_CPU_PASS','vectors':len(vectors),'signs':len(vectors)*64,
        'f32_bits':len(vectors)*64,'bf16_bits':len(vectors)*64,'vector_sha256':SHA,
        'driver_sha256':hashlib.sha256(args.driver.read_bytes()).hexdigest(),
        'scope':'fixed CPU vectors only; dense_exact_f64 is not used as an exact oracle; no device/actual-model/range-universal claim'},indent=2))
if __name__=='__main__':main()
