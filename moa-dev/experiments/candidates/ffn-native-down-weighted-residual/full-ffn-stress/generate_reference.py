#!/usr/bin/env python3
"""Run the unchanged official PyTorch full-FFN reference on a constructed fixture."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'upstream'))
import generate_references as official

H,L=3840,15360

def digest(raw):return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def input_info(array,shape,dtype):
    raw=array.tobytes(order='C')
    # Same byte-based checksum as test_kernels.rs prng::checksum(offset=0).
    padded=raw+b'\0'*((-len(raw))%4)
    words=np.frombuffer(padded,dtype='<u4').astype(np.uint64)
    factors=np.arange(len(words),dtype=np.uint64)*2+1
    check=int(np.sum(words*factors,dtype=np.uint64))&0xffffffff
    return dict(shape=list(shape),dtype=dtype,**digest(raw),prng_checksum_u32=check,
                prng_checksum_hex=f'{check:08x}')
def bfraw(tensor):return tensor.detach().contiguous().view(torch.int16).numpy().astype('<u2',copy=False)
def unique(tensor):
    values,counts=torch.unique(tensor.detach().float(),return_counts=True)
    return [dict(value=float(v),count=int(n)) for v,n in zip(values,counts)]

torch.set_num_threads(2)
torch.set_grad_enabled(False)
inputs={}
residual=torch.ones(H,dtype=torch.bfloat16)
pre=torch.ones(H,dtype=torch.bfloat16);post=torch.ones(H,dtype=torch.bfloat16)
for name,t in (('residual',residual),('pre_ff_rms_weight',pre),('post_ff_rms_weight',post)):
    inputs[name]=input_info(bfraw(t),t.shape,'bf16_le')
for name in ('up','gate','down'):
    inputs[name+'_global_scale']=input_info(np.array([2.**-11],dtype='<f4'),(1,),'f32_le_reciprocal')
layer=torch.full((8,),.375,dtype=torch.bfloat16)
inputs['layer_scalar']=input_info(bfraw(layer),(8,),'bf16_le')

model=official.gemma4.MLP(official.gemma4.Gemma4Config()).to(torch.bfloat16)
for name in ('up','gate','down'):
    shape=(L,H) if name!='down' else (H,L)
    codes=np.zeros(shape,dtype=np.uint8)
    scale=np.full((shape[0],shape[1]//16),2 if name!='down' else 15,dtype=np.float32)
    if name=='up':
        codes[:,0]=1
        codes[0,:]=0;codes[0,:2730]=7;codes[0,2730]=6
        codes[7680,:]=0;codes[7680,:5]=7;codes[7680,5]=4
    elif name=='gate':
        codes[:,0]=2
        codes[0,:]=0;codes[0,:2730]=7;codes[0,2730]=6
        codes[7680,:]=0;codes[7680,:682]=7;codes[7680,682]=6
    else:
        codes[:1920,1:7680]=7
        codes[1920:,7681:15360]=7
    if name!='down':scale[[0,7680],:]=4
    packed=codes[:,0::2]|(codes[:,1::2]<<4)
    inputs[name+'_weight_packed']=input_info(packed,shape,'f4e2m1_low_nibble_first')
    scale_bits=torch.from_numpy(scale).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    assert np.all(np.isin(scale_bits,[0x40,0x48,0x57]))
    inputs[name+'_weight_scale']=input_info(scale_bits,scale.shape,'f8e4m3_bits')
    # Match official gen_decoder_feedforward: FP4 values * repeated local scale,
    # then BF16 installed into actual nn.Linear; do not substitute a scalar dot.
    values=official.F4_MAGNITUDES[torch.from_numpy(codes).long()]
    folded=values*torch.from_numpy(scale).repeat_interleave(16,dim=-1)
    linear=getattr(model,name+'_proj')
    linear.weight.copy_(folded.to(torch.bfloat16))
    setattr(model,name+'_proj',official.GloballyScaled(linear,2.**-11))
    del codes,scale,packed,scale_bits,values,folded

observed={}
handles=[]
for name in ('up','gate','down'):
    wrapped=getattr(model,name+'_proj')
    handles.append(wrapped.linear.register_forward_hook(lambda mod,args,out,n=name: observed.__setitem__(n+'_raw',out.detach().clone())))
    handles.append(wrapped.register_forward_hook(lambda mod,args,out,n=name: observed.__setitem__(n+'_scaled',out.detach().clone())))
handles.append(model.down_proj.register_forward_pre_hook(lambda mod,args: observed.__setitem__('post_geglu',args[0].detach().clone())))
hidden=official.rms_norm(H,pre)(residual)
mlp_out=model(hidden)
normalized=official.rms_norm(H,post)(mlp_out)
expected=((residual+normalized).to(torch.bfloat16)*layer[0]).to(torch.bfloat16)
for handle in handles:handle.remove()
assert torch.equal(hidden,torch.ones_like(hidden))
assert float(observed['post_geglu'][0])==1024
assert float(observed['post_geglu'][7680])==.5
assert torch.all(observed['post_geglu'][1:7680]==2.**-22)
assert torch.all(observed['post_geglu'][7681:]==2.**-22)
assert torch.all(expected==.404296875),unique(expected)
raw=bfraw(expected).tobytes()
(HERE/'expected-bf16.bin').write_bytes(raw)
source_files={str(p.relative_to(HERE)):digest(p.read_bytes()) for p in [Path(__file__),*sorted((HERE/'upstream').rglob('*.py'))]}
result=dict(schema_version=1,method='Actual official gemma4.MLP + GloballyScaled + RMSNorm on CPU with dense locally-scaled BF16 nn.Linear weights; no candidate numerical reference substitution.',
    source_revision='cc82b9ccff22795165faffd3ba9a68cc557da001',
    official_prng_seed=False,component_wise_generator_range=True,npu_executed=False,
    source_files=source_files,torch_version=torch.__version__,numpy_version=np.__version__,cpu_threads=torch.get_num_threads(),
    inputs=inputs,intermediates={n:unique(t) for n,t in observed.items()},post_rms=unique(normalized),
    expected=dict(path='expected-bf16.bin',shape=[H],dtype='bf16_le',**digest(raw),unique=unique(expected),unique_bits=sorted(set(bfraw(expected).tolist())),
                  prng_checksum_u32=input_info(bfraw(expected),(H,),'bf16_le')['prng_checksum_u32']),
    unchanged_tolerance=dict(atol=.01,rtol=.01),
    scope='A deliberately correlated, component-wise in-range construction. Actual reference CPU execution does not establish that a public PRNG seed generates these tensors or that an NPU image passes/fails it.')
(HERE/'reference.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
