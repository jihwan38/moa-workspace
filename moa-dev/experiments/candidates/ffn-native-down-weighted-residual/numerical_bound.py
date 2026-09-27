#!/usr/bin/env python3
"""CPU-only exhaustive representation analysis; no kernel execution or NPU claim."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
SOURCE=HERE.parent/'ffn-native-down-e5'
OUT=HERE/'numerical-bound.json'
spec=importlib.util.spec_from_file_location('scope', SOURCE/'numerical_scope_analysis.py')
model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)
f32,e5=model.f32,model.e5_rne

def terms(u):
    hi=e5(u);lo=e5(f32(u-hi));r=f32(u-f32(hi+lo))
    return hi,lo,r

def reconstruct(u, factor):
    hi,lo,r=terms(u);last=e5(f32(r*factor))
    return f32(f32(hi+lo)+f32(last/factor))

def scaled_terms(u, lo_factor=16., third_factor=256.):
    hi=e5(u)
    r1=f32(u-hi)
    lo=e5(f32(r1*lo_factor))
    restored_lo=f32(lo/lo_factor)
    r3=f32(u-f32(hi+restored_lo))
    third=e5(f32(r3*third_factor))
    restored=f32(f32(hi+restored_lo)+f32(third/third_factor))
    return dict(hi=hi, first_residual=r1, scaled_lo=lo, restored_lo=restored_lo,
                third_residual=r3, scaled_third=third, restored=restored)

def calculate():
    # Power-of-two division of BF16 has at most8 significant bits. F32 underflow
    # can reduce that precision, so enumerate through its smallest subnormal.
    domain={0.}
    for exponent in range(-149,15):
        for significand in range(128,256):
            value=f32(significand*2.**(exponent-7))
            if value<=32640:
                domain.add(value)
    domain=sorted(domain)
    bfdomain=[struct.unpack('<f',struct.pack('<I',bits<<16))[0] for bits in range(0x4700)]
    assert all(u in domain for u in bfdomain)
    max_residual=max(abs(terms(u)[2]) for u in domain)
    largest_safe_power2=2.**math.floor(math.log2(57344/max_residual))
    assert max_residual==128 and largest_safe_power2==256
    witnesses=[dict(u=u,hi=terms(u)[0],lo=terms(u)[1],residual=terms(u)[2])
               for u in domain if abs(terms(u)[2])==max_residual]
    metrics={}
    for factor in (1,256):
        errors=[abs(reconstruct(u,factor)-u) for u in domain]
        worst=max(errors)
        metrics[str(factor)]={
            'exact_nonnegative_values':sum(e==0 for e in errors),
            'inexact_nonnegative_values':sum(e!=0 for e in errors),
            'max_absolute_normalized_error':worst,
            'worst_examples':[dict(u=u,restored=reconstruct(u,factor),hi=terms(u)[0],lo=terms(u)[1],residual=terms(u)[2])
                              for u,e in zip(domain,errors) if e==worst][:8],
            'negative_domain_checked':all(reconstruct(-u,factor)==-reconstruct(u,factor) for u in domain),
            'max_absolute_cast_input':max(abs(terms(u)[2]*factor) for u in domain),
        }
    s=2.**-4; x=2.**-22;u=f32(x/s)
    assert reconstruct(u,1)*s==0 and reconstruct(u,256)*s==x
    bad=False
    try:e5(max_residual*512)
    except ValueError:bad=True
    assert bad
    combined=[scaled_terms(u) for u in domain]
    combined_errors=[abs(c['restored']-u) for u,c in zip(domain,combined)]
    combined_max=max(combined_errors)
    assert max(abs(c['first_residual']) for c in combined)==2048
    assert max(abs(c['third_residual']) for c in combined)==128
    assert combined_max==2.**-25
    assert all(c['restored']==u for u,c in zip(domain,combined) if u>=2.**-17)
    assert all(scaled_terms(-u)['restored']==-c['restored'] for u,c in zip(domain,combined))
    inexact=[u for u,e in zip(domain,combined_errors) if e]
    metrics['lo16_third256']=dict(
        exact_nonnegative_values=sum(e==0 for e in combined_errors),
        inexact_nonnegative_values=len(inexact),
        max_absolute_normalized_error=combined_max,
        max_abs_first_residual=2048.,max_abs_third_residual=128.,
        max_scaled_lo_input=32768.,max_scaled_third_input=32768.,
        largest_safe_power2_lo_multiplier=16.,largest_safe_power2_third_multiplier=256.,
        exact_for_all_domain_values_at_or_above=2.**-17,
        largest_inexact_positive=max(inexact),negative_domain_checked=True,
        worst_examples=[dict(u=u,**c) for u,c,e in zip(domain,combined,combined_errors) if e==combined_max][:8],
        normalized_error_bound_is_not_output_error_bound=True)
    original_scope=model.calculate()
    # The repaired fixture preserves every tiny and amax anchor. Therefore the
    # down input, and all modeled subsequent values, are identical to reference.
    for amax in (1024.,.5):
        scale=2.**(math.floor(math.log2(amax))-14)
        for activation in (x,amax):
            assert f32(reconstruct(f32(activation/scale),256)*scale)==activation
            assert f32(scaled_terms(f32(activation/scale))['restored']*scale)==activation
    source_files=[]
    for p in (SOURCE/'down-e5-v2/mlp.rs',SOURCE/'numerical_scope_analysis.py',Path(__file__)):
        raw=p.read_bytes();source_files.append(dict(path=str(p.relative_to(ROOT)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    return dict(schema_version=1, cpu_only=True, npu_executed=False,
                sources=source_files,
                domain=dict(nonnegative_count=len(domain),finite_bf16_nonnegative_subset_count=len(bfdomain),
                            min_positive=domain[1],max=domain[-1],sign_symmetry=True,
                            explanation='All nonnegative F32 values with <=8 significant binary bits, through F32 subnormals, <=32640; a conservative superset of normalized finite BF16 under the actual power-of-two scale.'),
                max_abs_third_residual=max_residual,
                max_residual_witnesses=witnesses,
                e5_max_finite=57344,
                largest_safe_power2_multiplier=largest_safe_power2,
                next_power2_512_overflows=True,
                metrics=metrics,
                fixture=dict(original_activation=x,cluster0_scale=s,normalized=u,
                             current_restored=f32(reconstruct(u,1)*s),
                             proposed_third_only_restored=f32(reconstruct(u,256)*s),
                             proposed_lo16_third256_restored=f32(scaled_terms(u)['restored']*s),
                             predicted_repaired_final_halves=original_scope['output']['reference']['final'],
                             modeled_final_tolerance_failures=0),
                consumer_requirement='Keep FfnTerm through native F32 dot; For third-only multiply term0,term1,term2 by s,s,s/256; preferred lo16/third256 uses s,s/16,s/256, before F32 term reduction. Current contract_time removes FfnTerm before scale, so a quantizer-only change is invalid.',
                precision_limits=[
                    'An overflow-safe final residual multiplier is not an exact encoding of every BF16 value.',
                    'Values sufficiently far below the cluster amax still underflow even after multiplication by256.',
                    'When hi and lo underflow, only the final E5 mantissa remains, causing additional rounding; lo16 reduces this hole but does not remove every tiny-value error.',
                    'No target lowering, hardware cast/subnormal policy, native summation order, full-FFN correctness or performance has been tested for the repair.',
                    'The normalized error bound must be multiplied by s; it is not a global absolute output-error bound.',
                    'The scalar reachability construction remains a correlated component-wise in-range fixture, not an observed public PRNG seed.',
                ])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');args=ap.parse_args()
    result=calculate()
    if args.check:
        assert json.loads(OUT.read_text())==result
        print('PASS CPU bound reproduces: max residual128, max safe power2 multiplier256, fixture restored; no general exactness claim')
    else:
        OUT.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:result[k] for k in ('domain','max_abs_third_residual','largest_safe_power2_multiplier','metrics','fixture')},indent=2))
