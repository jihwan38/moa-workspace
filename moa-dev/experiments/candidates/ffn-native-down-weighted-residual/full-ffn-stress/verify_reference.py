#!/usr/bin/env python3
"""Read-only provenance/fixture check; does not import Torch or execute a kernel."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
BASE='cc82b9ccff22795165faffd3ba9a68cc557da001'

def check_hash(path,record):
    raw=path.read_bytes()
    assert len(raw)==record['bytes'] and hashlib.sha256(raw).hexdigest()==record['sha256'],path

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');ap.parse_args()
    build=json.loads((HERE/'reference-build.json').read_text())
    ref=json.loads((HERE/'reference.json').read_text())
    assert build['exit_code']==0 and build['cpu_only'] and not build['npu_executed']
    for name,record in build['source_files'].items():check_hash(HERE/name,record)
    for name,record in build['outputs'].items():check_hash(HERE/name,record)
    for name,record in ref['source_files'].items():check_hash(HERE/name,record)
    for name in ('generate_references.py','fixture_prng.py','reference/gemma4.py'):
        frozen=(HERE/'upstream'/name).read_bytes()
        original=subprocess.check_output(['git','show',BASE+':furiosa-opt-gemma4-12B/scripts/'+name],cwd=ROOT)
        assert frozen==original,name
    expected=(HERE/'expected-bf16.bin').read_bytes()
    assert len(expected)==7680 and set(struct.unpack('<3840H',expected))=={0x3ecf}
    assert ref['expected']['unique']==[{'value':.404296875,'count':3840}]
    assert ref['intermediates']['post_geglu']==[
        {'value':2.**-22,'count':15358},{'value':.5,'count':1},{'value':1024.,'count':1}]
    assert len(ref['inputs'])==13 and not ref['npu_executed'] and not ref['official_prng_seed']
    assert ref['unchanged_tolerance']=={'atol':.01,'rtol':.01}
    print('PASS actual official Torch CPU full-FFN fixture provenance: 3840 x BF16 0x3ecf; no NPU or PRNG-seed claim')
