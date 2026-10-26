#!/usr/bin/env python3
"""Compare a plain CPU Rust helper to independently frozen Fraction vectors.

The caller supplies the host-only executable. No build, network or NPU action.
"""
import argparse, hashlib, json, subprocess
from fractions import Fraction
from pathlib import Path

HERE=Path(__file__).resolve().parent
VECTOR_SHA='315c55a62ff5826a3778b5441c9175e05f1ad18ba806500fd154e4b3c3aabdfc'
def value(bits):
    sign=-1 if bits&32768 else 1
    e=(bits>>7)&255;m=bits&127
    if e==0:return sign*Fraction(m)*Fraction(2)**-133
    assert e!=255
    return sign*Fraction(128+m)*Fraction(2)**(e-134)
def original_codes(bits):
    values=[value(b) for b in bits];a=max(max(map(abs,values)),Fraction(2)**-112)
    out=[]
    for x in values:
        q=7*abs(x)/a
        k=(q.numerator*2+q.denominator)//(2*q.denominator)
        out.append(-k if x<0 else k)
    return out
def main():
    ap=argparse.ArgumentParser();ap.add_argument('driver',type=Path);args=ap.parse_args()
    path=HERE/'independent-fraction/fraction-vectors.json';raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==VECTOR_SHA
    data=json.loads(raw);fixtures=data['fixtures']
    fields=[str(len(fixtures))]
    for f in fixtures:
        fields += [str(f['original_scale_bits']),*map(str,f['bf16_bits']),*map(str,original_codes(f['bf16_bits']))]
    result=subprocess.run([str(args.driver.resolve())],input=' '.join(fields),text=True,capture_output=True,check=True)
    lines=result.stdout.splitlines();assert len(lines)==len(fixtures)+1
    assert list(map(int,lines[0].removeprefix('DOMAIN ').split(',')))==data['domain_bf16_bits']
    for i,(line,f) in enumerate(zip(lines[1:],fixtures)):
        index,selected,scale,baseline,sse,norm,packed=line.split()
        assert (int(index),int(selected),int(scale))==(i,f['selected_index'],f['selected_scale_bits']),f['name']
        assert (baseline,sse,packed)==(f['baseline_sse_units'],f['selected_sse_units'],f['packed_hex']),f['name']
        assert int(sse)<=int(baseline)
        expected_norm=sum((value(b)*2**38)**2 for b in f['bf16_bits'])
        assert expected_norm.denominator==1 and int(norm)==expected_norm.numerator
    print(json.dumps({'decision':'CPU_FRACTION_VECTORS_PASS','fixtures':len(fixtures),'codes':960*len(fixtures),
        'packed_bytes':480*len(fixtures),'domain_magnitudes':len(data['domain_bf16_bits']),
        'vector_sha256':VECTOR_SHA,'driver_sha256':hashlib.sha256(args.driver.read_bytes()).hexdigest(),
        'scope':'provided-format/synthetic CPU proof; no supplied model execution, device or task-quality claim'},indent=2))
if __name__=='__main__':main()
