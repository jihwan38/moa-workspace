#!/usr/bin/env python3
"""Read ELF64/bincode registry without executing either NPU binary."""
import struct,json,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[2]/'artifacts'/'fp8-packet-probe'
out={}
for kind in ['scalar','blocks']:
    p=root/kind/'test_runtime';b=p.read_bytes()
    assert b[:6]==b'\x7fELF\x02\x01'
    shoff=struct.unpack_from('<Q',b,40)[0]
    entsize,num,strings=struct.unpack_from('<HHH',b,58)
    headers=[struct.unpack_from('<IIQQQQIIQQ',b,shoff+i*entsize) for i in range(num)]
    h=headers[strings];names=b[h[4]:h[4]+h[5]]
    sections={names[h[0]:].split(b'\x00',1)[0].decode():b[h[4]:h[4]+h[5]] for h in headers}
    reg=sections['furiosa_kernels'];pos=0
    def u64():
        global pos
        n=struct.unpack_from('<Q',reg,pos)[0];pos+=8;return n
    entries=[]
    for _ in range(u64()):
        n=u64();name=reg[pos:pos+n].decode();pos+=n
        key=[u64() for _ in range(u64())]
        size=u64();image=reg[pos:pos+size];pos+=size
        entries.append({'path':name,'key':key,'image_bytes':size,'image_sha256':hashlib.sha256(image).hexdigest()})
    assert len(entries)==1 and entries[0]['path'].endswith('packet_probe::fp8_'+kind)
    assert entries[0]['image_bytes']>0 and pos==len(reg)
    out[kind]={'binary_sha256':hashlib.sha256(b).hexdigest(),'binary_bytes':len(b),'registry':entries,'registry_bytes':len(reg)}
print(json.dumps(out,indent=2))
Path(__file__).with_name('registry-summary.json').write_text(json.dumps(out,indent=2)+'\n')
