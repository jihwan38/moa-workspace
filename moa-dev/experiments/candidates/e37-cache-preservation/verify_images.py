#!/usr/bin/env python3
"""Parse ELF64 kernel registries and compare images without executing binaries."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

EXPECTED_NAMES={'sliding_project_qkv','sliding_attention_output','decoder_feedforward'}


def read_registry(path):
    b=Path(path).read_bytes()
    assert b[:6]==b'\x7fELF\x02\x01', 'expected ELF64 little-endian'
    shoff=struct.unpack_from('<Q',b,40)[0]
    entsize,count,strings=struct.unpack_from('<HHH',b,58)
    headers=[struct.unpack_from('<IIQQQQIIQQ',b,shoff+i*entsize) for i in range(count)]
    strings_header=headers[strings]
    names=b[strings_header[4]:strings_header[4]+strings_header[5]]
    sections={names[h[0]:].split(b'\0',1)[0].decode():h for h in headers}
    h=sections['furiosa_kernels'];registry_base=h[4];reg=b[h[4]:h[4]+h[5]]
    pos=0
    def u64():
        nonlocal pos
        value=struct.unpack_from('<Q',reg,pos)[0];pos+=8;return value
    images={}
    for _ in range(u64()):
        n=u64();name=reg[pos:pos+n].decode();pos+=n
        key=[u64() for _ in range(u64())]
        n=u64();start=pos;image=reg[pos:pos+n];pos+=n
        short=name.rsplit('::',1)[-1]
        assert short in EXPECTED_NAMES and short not in images
        assert not key and n>0 and len(image)==n
        # Exact byte extent from the registry, not a substring match guessed by size.
        images[short]={'path':name,'key':key,'offset':registry_base+start,'image':image}
    assert pos==len(reg) and set(images)==EXPECTED_NAMES
    return b,images


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('frozen_e37_elf');parser.add_argument('guard_elf')
    parser.add_argument('--kernel-out',type=Path)
    args=parser.parse_args()
    before,old=read_registry(args.frozen_e37_elf)
    after,new=read_registry(args.guard_elf)
    records={}
    for name in sorted(EXPECTED_NAMES):
        a=old[name];b=new[name]
        assert a['path']==b['path'] and a['key']==b['key']
        assert a['image']==b['image'], f'{name}: device image changed'
        records[name]={'bytes':len(b['image']),'sha256':hashlib.sha256(b['image']).hexdigest(),
                       'frozen_e37_elf_offset':a['offset'],'guard_elf_offset':b['offset'],
                       'byte_identical':True,'registry_path':b['path']}
        if args.kernel_out:
            args.kernel_out.mkdir(parents=True,exist_ok=True)
            (args.kernel_out/(name+'.bin')).write_bytes(b['image'])
    assert b'CACHE_PRESERVATION total_mismatches=' in after
    assert b'CACHE_CHECK stage=' in after
    print(json.dumps({'status':'PASS','method':'ELF64 furiosa_kernels registry parse; exact device-image equality',
                      'frozen_e37_binary':{'path':args.frozen_e37_elf,'bytes':len(before),'sha256':hashlib.sha256(before).hexdigest()},
                      'guard_binary':{'path':args.guard_elf,'bytes':len(after),'sha256':hashlib.sha256(after).hexdigest()},
                      'kernel_byte_identity':records,'guard_messages_present':True,
                      'binary_executed':False},indent=2))


if __name__=='__main__':main()
