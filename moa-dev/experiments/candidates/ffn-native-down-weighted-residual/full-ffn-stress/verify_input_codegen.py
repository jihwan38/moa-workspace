#!/usr/bin/env python3
"""Independent Rust-host byte reconstruction vs actual Torch reference hashes.
Only a tiny CPU C checksum helper is compiled in a temporary directory. No Cargo,
NPU compiler, shared target, runtime kernel edit, or device access is involved.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import tempfile

HERE=Path(__file__).resolve().parent
OUT=HERE/'input-codegen-verification.json'
H,L=3840,15360

def info(p):
    raw=p.read_bytes();return dict(path=p.name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

def build_arrays():
    # Independent packed-byte implementation of frozen Rust lines13-32.
    row_bytes=H//2
    up=bytearray((b'\x01'+bytes(row_bytes-1))*L)
    gate=bytearray((b'\x02'+bytes(row_bytes-1))*L)
    for matrix in (up,gate):
        matrix[:row_bytes]=bytes(row_bytes)
        matrix[:1365]=b'\x77'*1365
        matrix[1365]=6
    a=7680*row_bytes
    up[a:a+row_bytes]=bytes(row_bytes)
    up[a:a+2]=b'\x77'*2;up[a+2]=0x47
    gate[a:a+row_bytes]=bytes(row_bytes)
    gate[a:a+341]=b'\x77'*341;gate[a+341]=6
    left=b'\x70'+b'\x77'*3839+bytes(3840)
    right=bytes(3840)+b'\x70'+b'\x77'*3839
    down=left*1920+right*1920
    local=bytearray(b'\x40'*(L*H//16))
    local_f32=bytearray(struct.pack('<f',2.)*(L*H//16))
    for row in (0,7680):
        b,e=row*H//16,(row+1)*H//16
        local[b:e]=b'\x48'*(H//16)
        local_f32[4*b:4*e]=struct.pack('<f',4.)*(H//16)
    ones=struct.pack('<H',0x3f80)*H
    arrays={'residual':ones,'pre_ff_rms_weight':ones,'post_ff_rms_weight':ones,
            'up_weight_packed':up,'gate_weight_packed':gate,'down_weight_packed':down,
            'up_weight_scale':local,'gate_weight_scale':local,
            'down_weight_scale':b'\x57'*(H*L//16),
            'up_global_scale':struct.pack('<f',2.**-11),
            'gate_global_scale':struct.pack('<f',2.**-11),
            'down_global_scale':struct.pack('<f',2.**-11),
            'layer_scalar':struct.pack('<H',0x3ec0)*8}
    return arrays,local_f32

def calculate():
    reference=json.loads((HERE/'reference.json').read_text())
    arrays,local_f32=build_arrays()
    checks={}
    for name,data in arrays.items():
        actual=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        expected={k:reference['inputs'][name][k] for k in ('bytes','sha256')}
        assert actual==expected,(name,actual,expected)
        checks[name]={**actual,'matches_actual_torch_reference_bytes':True}
    with tempfile.TemporaryDirectory(prefix='moa-e134-cpu-fnv-') as tmp:
        library=Path(tmp)/'fnv64.so'
        command=['cc','-O2','-shared','-fPIC',str(HERE/'fnv64.c'),'-o',str(library)]
        subprocess.run(command,check=True,capture_output=True)
        fn=ctypes.CDLL(str(library)).fnv64
        fn.argtypes=[ctypes.c_void_p,ctypes.c_size_t];fn.restype=ctypes.c_uint64
        def checksum(raw):
            b=bytes(raw);return f'{fn(ctypes.c_char_p(b),len(b)):016x}'
        # Independently check helper's constants/update against Python on small bytes.
        for raw in (b'',b'a',bytes(range(256))):
            want=0xcbf29ce484222325
            for v in raw:want=((want^v)*0x100000001b3)&((1<<64)-1)
            assert checksum(raw)==f'{want:016x}'
        fnvs={name:checksum(arrays[name+'_weight_packed']) for name in ('up','gate','down')}
        fnvs['local']=checksum(local_f32)
        fnvs['expected']=checksum((HERE/'expected-bf16.bin').read_bytes())
    expected_line='FULL_STRESS_INPUT '+' '.join(f'{k}={v}' for k,v in fnvs.items())
    actual_preparation={}
    for side in ('e128','e133','e134'):
        path=HERE/'evidence'/side/'prepare.log'
        lines=re.findall(r'^FULL_STRESS_INPUT .+$',path.read_text(),re.M)
        assert lines==[expected_line] and path.read_text().count('FULL_STRESS_PREPARED')==1
        actual_preparation[side]={**info(path),'path':str(path.relative_to(HERE)),
                                  'matches_independent_torch_bound_fnv':True,'input_line':lines[0]}
    host=(HERE/'full_ffn_stress.rs').read_text()
    assert 'delta>0.01+0.01*e.abs()' in host
    assert 'mismatches+=usize::from(!a.is_finite()||delta>0.01+0.01*e.abs());' in host
    assert 'launch(ops::decoder_feedforward,(&mut d,&mut residual,&pre,&up,&gate,&down,&ups,&gates,&downs,&us,&gs,&ds,&post,&layer))' in host
    assert 'include_bytes!("expected.bf16.bin")' in host
    return dict(schema_version=1,valid=True,cpu_only=True,npu_executed=False,
        source_files={p.name:info(p) for p in (HERE/'full_ffn_stress.rs',HERE/'reference.json',HERE/'expected-bf16.bin',HERE/'fnv64.c',Path(__file__))},
        tensor_byte_checks=checks,original_tensor_count=13,all_reconstructed_inputs_match_torch_reference=True,
        expected_host_fnv64=fnvs,expected_host_input_line=expected_line,
        actual_prepared_host_logs=actual_preparation,actual_prepare_checksums_all_match=True,
        acceptance=dict(atol=.01,rtol=.01,all_output_elements=H,nonfinite_rejected=True,
                        expected_from_actual_official_torch=True,
                        process_exit_alone_is_acceptance=False,
                        note='The executable prints diagnostics without failing the E133 process; run.py separately requires control and repair acceptance. These are additional full-FFN stress inputs, not official PRNG inputs.'),
        observation_limits='Packed up/gate/down, decoded local-scale F32 bytes and expected are printed by the actual host and can be compared with this expected line. Remaining constant tensors are validated by frozen source review and independent reconstruction; the host does not print their runtime hashes.',
        method='Independent stdlib packed-byte constructor for the frozen Rust expressions, SHA256 equality with actual dense official Torch generator inputs, and CPU C FNV1a equality with the Rust printed-checksum algorithm.')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');ap.add_argument('--prepare-log',type=Path,action='append',default=[]);args=ap.parse_args()
    value=calculate()
    if args.check:assert json.loads(OUT.read_text())==value,'input verification record changed'
    else:OUT.write_text(json.dumps(value,indent=2)+'\n')
    for path in args.prepare_log:
        lines=re.findall(r'^FULL_STRESS_INPUT .+$',path.read_text(),re.M)
        assert lines==[value['expected_host_input_line']],(str(path),lines,value['expected_host_input_line'])
        assert path.read_text().count('FULL_STRESS_PREPARED')==1
    print(value['expected_host_input_line'])
    print(f'PASS: 13/13 input byte hashes match actual Torch reference; 3 frozen actual prepare logs + {len(args.prepare_log)} explicit logs compared; unchanged tolerance/all-element/nonfinite rules reviewed')
