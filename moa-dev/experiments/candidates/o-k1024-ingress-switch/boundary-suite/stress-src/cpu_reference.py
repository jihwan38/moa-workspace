"""Recheck the immutable E47 case7 payload with original BF16 Torch O/RMS/residual."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from safetensors.numpy import load_file
scripts,fixture,out=map(Path,sys.argv[1:])
sys.path.insert(0,str(scripts));import generate_references as official
from fixture_prng import checksum
assert hashlib.sha256(fixture.read_bytes()).hexdigest()=='35df91a34ca00b077b35327c513bb36783355363855e214674ddaa04a47ec907'
torch.set_num_threads(4);torch.set_grad_enabled(False);torch.use_deterministic_algorithms(True)
f=load_file(str(fixture));arrays={n:f['case7.input.'+n] for n in ['x','o_weight_scale','post_attn_rms_weight','residual']}
q=np.arange(4096);codes=np.broadcast_to(np.where(q%256<2,0x38,0x77).astype(np.uint8),(3840,4096)).copy();arrays['o_weight']=codes.reshape(-1)
for n,a in arrays.items():assert int(checksum(a))==int(f['case7.checksum.'+n][0])
def bf(a):return torch.from_numpy(a.copy().view(np.int16)).view(torch.bfloat16)
x=bf(arrays['x']);scale=bf(arrays['o_weight_scale']);gamma=bf(arrays['post_attn_rms_weight']);residual=bf(arrays['residual'])
assert bool((x.reshape(16,256)[:,0]==1).all()) and bool((x.reshape(16,256)[:,1]==-1).all()) and bool((x.reshape(16,256)[:,2:]==2**-22).all())
assert bool((scale==2**-10).all()) and bool((gamma==.5).all()) and bool((residual==.5).all())
weights=torch.from_numpy(codes).view(torch.float8_e4m3fn).float()
raw=x.float()@weights.T; projected=official.scaled_projection(x,weights,scale)
normalized=official.rms_norm(3840,gamma)(projected);expected=(residual+normalized).to(torch.bfloat16)
assert bool((raw==.2325439453125).all()) and bool((projected==.0002269744873046875).all())
assert bool((expected==.609375).all()) and torch.equal(expected.float(),torch.from_numpy(f['case7.expected']))
record={'passed':True,'case':7,'cases':1,'outputs':3840,'oracle':'Original stored BF16 input, E4 weight, F32 dot/channel scale, BF16 projection, official RMSNorm, BF16 residual.', 'public_atol':.05,'public_rtol':.01,'raw_dot':float(raw[0]),'projected_bf16':float(projected[0]),'expected':.609375,'expected_bits':'0x3f1c','all_intermediates_finite':all(bool(torch.isfinite(t).all()) for t in [raw,projected,normalized,expected]),'input_sha256':{n:hashlib.sha256(a.tobytes()).hexdigest() for n,a in arrays.items()},'input_checksums':{n:int(checksum(a)) for n,a in arrays.items()},'scope':'Finite structured case within generator component ranges, not an observed PRNG seed or arbitrary-input proof.'}
out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
