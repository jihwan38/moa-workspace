#!/usr/bin/env python3
"""Read ELF64 kernel registry without executing the diagnostic binary."""
import hashlib,json,struct,sys
from pathlib import Path
p=Path(sys.argv[1]);blob=p.read_bytes()
assert blob[:6]==b'\x7fELF\x02\x01'
shoff=struct.unpack_from('<Q',blob,40)[0];esize,count,names_index=struct.unpack_from('<HHH',blob,58)
headers=[struct.unpack_from('<IIQQQQIIQQ',blob,shoff+i*esize) for i in range(count)]
h=headers[names_index];names=blob[h[4]:h[4]+h[5]]
sections={names[h[0]:].split(b'\0',1)[0].decode():h for h in headers}
h=sections['furiosa_kernels'];registry=blob[h[4]:h[4]+h[5]];pos=0

def u64():
    global pos
    value=struct.unpack_from('<Q',registry,pos)[0];pos+=8;return value

entries=[]
for _ in range(u64()):
    n=u64();name=registry[pos:pos+n].decode();pos+=n
    key=[u64() for _ in range(u64())];n=u64();offset=h[4]+pos;image=registry[pos:pos+n];pos+=n
    assert len(image)==n and n>0 and not key
    entries.append({'path':name,'key':key,'image_bytes':n,'image_sha256':hashlib.sha256(image).hexdigest(),'elf_offset':offset})
assert pos==len(registry)
assert len(entries) == 1 and entries[0]['path'] == 'furiosa_opt_gemma4::ops::sliding_attention_output'
assert entries[0]['image_sha256'] == '359473814839effc16ade242bad3c598a93610601bc650f550ff44228898d88b'
assert blob.count(image) == 1
print(json.dumps({'binary_path':str(p),'binary_sha256':hashlib.sha256(blob).hexdigest(),'binary_bytes':len(blob),'registry':entries,'binary_executed':False},indent=2))
