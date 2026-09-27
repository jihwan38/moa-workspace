#!/usr/bin/env python3
"""CPU-only, source-bound E133 scope calculation; not an NPU/public-fixture test."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = HERE / "numerical-scope-analysis.json"
BASE = "cc82b9ccff22795165faffd3ba9a68cc557da001"
FP4 = (0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.)
H, L, HALF = 3840, 15360, 7680

def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]

def bf(x):
    bits = struct.unpack('<I', struct.pack('<f', x))[0]
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack('<f', struct.pack('<I', bits))[0]

def e5_rne(x):
    """Finite E5M2 RNE model, rejecting overflow rather than assuming saturation."""
    if x == 0:
        return x
    a = abs(x)
    step = 2.**(-16 if a < 2.**-14 else math.floor(math.log2(a))-2)
    q = round(a / step) * step
    if q > 57344:
        raise ValueError('E5 overflow')
    return math.copysign(q, x)

def quantized(x, amax):
    scale = 2.**(math.floor(math.log2(max(amax, 2.**-112)))-14)
    u = f32(x / scale)
    hi = e5_rne(u)
    lo = e5_rne(f32(u-hi))
    third = e5_rne(f32(u-f32(hi+lo)))
    restored = f32(f32(f32(hi+lo)+third)*scale)
    return dict(scale=scale, normalized=u, terms=[hi,lo,third], restored=restored)

def raw_dot(terms, scale):
    """A row recipe is count/value pairs followed by zero weights to H."""
    assert scale in (2.,4.) and sum(n for n,_ in terms) <= H
    total=0.
    for count,value in terms:
        assert value in FP4 and bf(value*scale)==value*scale
        for _ in range(count):
            exact=total+value*scale
            assert f32(exact)==exact
            total=exact
    return total

def geglu(up_raw, gate_raw):
    up, gate = bf(bf(up_raw)/2048), bf(bf(gate_raw)/2048)
    # Public PyTorch MLP has a BF16 GELU result before the BF16 product.
    tanh_gelu=.5*gate*(1+math.tanh(math.sqrt(2/math.pi)*(gate+.044715*gate**3)))
    public=bf(bf(tanh_gelu)*up)
    # Production uses F32 erf-based GELU, followed by one BF16 commit.
    doubled=f32(f32(1+f32(math.erf(f32(gate*f32(1/math.sqrt(2))))))*gate)
    production=bf(f32(f32(up*doubled)/2))
    assert public==production
    return dict(up_raw=up_raw, gate_raw=gate_raw, up=up, gate=gate,
                public_tanh_bf16=public, production_erf_cpu_bf16=production)

def epilogue(values):
    """Equal-sized output halves, all-one RMS weights/residual; F32 RMS, BF16 commits."""
    mean=f32(sum(f32(x*x) for x in values)/len(values))
    denom=f32(math.sqrt(f32(mean+f32(1e-6))))
    normalized=[bf(f32(x/denom)) for x in values]
    result=[bf(f32(bf(f32(1+x))*.375)) for x in normalized]
    return dict(mean_square=mean, rms=denom, normalized=normalized, final=result)

def digest(path, revision=None):
    data = subprocess.check_output(['git','show',f'{revision}:{path}'],cwd=ROOT) if revision else (ROOT/path).read_bytes()
    return dict(path=path, git_revision=revision, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

def calculate():
    # f32_uniform's U01 maximum rounds to BF16 1; this is a per-component range fact.
    assert bf((2**24-1)/2**24)==1
    assert bf(f32(1/math.sqrt(f32(1+f32(1e-6)))))==1
    ordinary=geglu(raw_dot([(1,.5)],2.),raw_dot([(1,1.)],2.))
    large=geglu(raw_dot([(2730,6.),(1,4.)],4.),raw_dot([(2730,6.),(1,4.)],4.))
    small=geglu(raw_dot([(5,6.),(1,2.)],4.),raw_dot([(682,6.),(1,4.)],4.))
    tiny=ordinary['public_tanh_bf16']
    assert tiny==2.**-22 and large['public_tanh_bf16']==1024 and small['public_tanh_bf16']==.5
    qs=[quantized(tiny,1024),quantized(tiny,.5)]
    assert qs[0]['restored']==0 and qs[1]['restored']==tiny
    # All down row scales 15; first/second H halves select only first/second L half.
    # The amax anchor in each L half is zero-weighted in down.
    raw=(HALF-1)*tiny*6*15
    assert raw==f32(raw)
    raw_bf=bf(raw)
    y=bf(f32(raw_bf/2048))
    ref=epilogue([y,y]); candidate=epilogue([0.,y])
    errors=[abs(a-b) for a,b in zip(ref['final'],candidate['final'])]
    tolerances=[.01+.01*abs(v) for v in ref['final']]
    mismatch_halves=sum(a>b for a,b in zip(errors,tolerances))
    assert mismatch_halves==1
    contract_paths=[
        'furiosa-opt-gemma4-12B/README.md',
        'furiosa-opt-gemma4-12B/src/bin/test_kernels.rs',
        'furiosa-opt-gemma4-12B/src/ops.rs',
        'furiosa-opt-gemma4-12B/src/device/shared/rmsnorm.rs',
        'furiosa-opt-gemma4-12B/src/device/shared/residual.rs',
        'furiosa-opt-gemma4-12B/scripts/generate_references.py',
        'furiosa-opt-gemma4-12B/scripts/reference/gemma4.py',
    ]
    return dict(
        schema_version=1,
        scope='Source-grounded scalar CPU model of one component-wise in-range full-FFN construction, not a replay of the public PRNG or an NPU fixture.',
        analysis_reproduced=True, npu_executed=False, actual_public_prng_seed_found=False,
        public_grading_failure_demonstrated=False, arbitrary_bf16_accuracy_proven=False,
        source_inputs=[digest(p,BASE) for p in contract_paths]+[
            digest(str((HERE/'down-e5-v2/mlp.rs').relative_to(ROOT))),
            digest(str((HERE/'down-oracle/results.json').relative_to(ROOT))),
            digest(str(Path(__file__).resolve().relative_to(ROOT)))],
        construction=dict(H=H,L=L,half_L=HALF,residual=1.,pre_rms_weight=1.,post_rms_weight=1.,
                          raw_global_scales=[2048.,2048.,2048.],global_reciprocal=2.**-11,layer_scalar=.375,
                          ordinary_rows='all L rows except 0 and 7680',ordinary=ordinary,
                          row0=large,row7680=small,
                          up_gate_local_scales=[2.,4.], down_local_scale=15.,
                          down_rule='h<1920: value6 at 1<=l<7680, else0; h>=1920: value6 at 7681<=l<15360, else0',
                          fp4_values_are_format_values_not_nibble_codes=True),
        candidate_tiny_quantization=qs,
        output=dict(reference_raw_down=raw,reference_raw_down_bf16=raw_bf,
                    reference_after_global_bf16=y,reference=ref,candidate=candidate,
                    absolute_errors_by_half=errors,tolerances_by_half=tolerances,
                    predicted_failing_outputs=(H//2)*mismatch_halves),
        limits=[
            'No complete PyTorch dense fixture or NPU execution was run for this construction.',
            'Scalar BF16/F32/RNE/transcendental model does not emulate all hardware reduction or unary approximations.',
            'All scalar values belong to generator component ranges; correlated tensors need not occur for any public seed.',
            'This is a concrete numerical risk, not a claim that nine validated standard/extra inputs or official grading failed.',
            'Seven mapping cases and measured raw helper case4 remain independent evidence with their original scope.',
        ],
        repair_hypothesis='Preserve FfnTerm through native F32 contraction, restore differently scaled terms independently then sum. Derive a safe final-residual multiplier over the actual normalized domain; changing the quantizer alone is wrong because current contract_time sums equal-scale terms.',
    )

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');args=ap.parse_args()
    result=calculate()
    if args.check:
        assert json.loads(OUT.read_text())==result, 'saved CPU analysis/source digests changed'
        print('PASS: numerical scope CPU analysis and source digests reproduce (no NPU claim)')
    else:
        OUT.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result['output'],indent=2))
