#!/usr/bin/env python3
"""Bind the independent quantizer audit's BF16 inputs to the host oracle checksum."""
import argparse, hashlib, importlib.util, json, struct
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[4]
def file(p):
    raw=p.read_bytes();return {'path':str(p.relative_to(ROOT)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def calculate():
    model=HERE/'probe-v2/quantizer_audit.py';spec=importlib.util.spec_from_file_location('e128_quantizer_model',model);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    built=HERE/'probe-build-v2b.json';b=json.loads(built.read_text());assert b['valid'];m.audit()
    cases=[]
    for case,expected_hash in b['oracle']['input_checksums']:
        raw=bytearray()
        for k in range(3840):
            x=m.activation(int(case),k);bits=struct.unpack('<I',struct.pack('<f',x))[0]
            assert bits&0xffff==0 # Every selected activation is exactly BF16.
            raw.extend(struct.pack('<H',bits>>16))
        h=0xcbf29ce484222325
        for v in raw:h=((h^v)*0x100000001b3)&((1<<64)-1)
        assert f'{h:016x}'==expected_hash
        cases.append({'case':int(case),'input_elements':3840,'fingerprint':expected_hash,'model_matches_actual_host_input':True})
    return {'valid':True,'model_source':file(model),'build':file(built),'checker':file(Path(__file__)),'cases':cases,'scope':'Independent Python E5M2 model input bytes match all eight actual Rust BF16 input buffers; hardware correctness remains a separate probe result.'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();r=calculate();out=HERE/'probe-model-binding.json'
    if a.check:assert json.loads(out.read_text())==r
    else:
        with out.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print('All8 independent quantizer-model inputs match actual host BF16 bytes PASS')
