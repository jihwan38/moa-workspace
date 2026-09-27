#!/usr/bin/env python3
"""Read-only repository audit; CPU scalar model, not NPU/whole-model validation."""
import bisect,hashlib,json,math,struct,subprocess
from pathlib import Path
ROOT=Path('/home/jihwan38/moa-workspace')
SDK='/home/moa/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/furiosa-opt-std-0.8.1/src/float/f8e4.rs'
def f32(x):
 try:return struct.unpack('<f',struct.pack('<f',x))[0]
 except OverflowError:return math.copysign(math.inf,x)
def bits(x):return struct.unpack('<I',struct.pack('<f',x))[0]
def bf16(x):
 b=bits(f32(x));b=(b+0x7fff+((b>>16)&1))>>16
 return struct.unpack('<f',struct.pack('<I',(b&65535)<<16))[0]
def val(code):
 e=(code>>3)&15;m=code&7
 return (m*2.**-9 if e==0 else (1+m/8)*2.**(e-7))*(-1 if code&128 else 1)
positive=[val(c) for c in range(127)]
def e4(x):
 if math.isnan(x):return math.nan
 a=abs(f32(x));i=bisect.bisect_left(positive,a)
 if i==0:q=0.
 elif i>=len(positive):q=positive[-1]
 else:
  dl=a-positive[i-1];du=positive[i]-a
  q=positive[i-1] if dl<du or (dl==du and (i-1)%2==0) else positive[i]
 return math.copysign(q,x)
def sdk_code(x):
 # Literal translation of installed SDK f8_e4_from_f32, independently compared to lattice RNE.
 n=bits(f32(x));exp=(n>>23)&255;man=n&0x7fffff
 if exp>120:
  valid=(n>>20)&0x7ff;v=valid+int((n&0xfffff)+(valid&1)>0x80000)
  a=0x7e if v>=((135<<3)|6) else v-(120<<3)
 elif exp==120:a=8 if man>=0x700000 else 7 if man>0x500000 else 6 if man>=0x300000 else 5 if man>0x100000 else 4
 elif exp==119:a=4 if man>=0x600000 else 3 if man>0x200000 else 2
 elif exp==118:a=2 if man>=0x400000 else 1
 elif exp==117 and man!=0:a=1
 else:a=0
 return ((n>>31)<<7)|a
def terms(u,factors):
 total=0.;records=[]
 for factor in factors:
  r=f32(u-total);cast_in=f32(r*factor);q=e4(cast_in);restored=f32(q/factor)
  total=f32(total+restored)
  records.append(dict(residual=r,cast_input=cast_in,quantized=q,restored=restored))
 return total,records

def domain():
 d={0.}
 for exponent in range(-149,8):
  for significand in range(128,256):
   x=f32(significand*2.**(exponent-7))
   if x<=255:d.add(x)
 return sorted(d)

def analyze(ds, factors):
 maxinputs=[0.]*len(factors);maxres=[0.]*len(factors);worst=0.;worsts=[];bad=[];checks=0
 for u in ds:
  restored,rec=terms(u,factors);err=abs(restored-u)
  for i,r in enumerate(rec):
   maxinputs[i]=max(maxinputs[i],abs(r['cast_input']));maxres[i]=max(maxres[i],abs(r['residual']))
   assert e4(r['cast_input'])==val(sdk_code(r['cast_input']));checks+=1
  if err>0:bad.append(u)
  if err>worst:worst=err;worsts=[]
  if err==worst and len(worsts)<5:worsts.append(dict(u=u,restored=restored,error=err,terms=rec))
  assert terms(-u,factors)[0]==-restored
 exact_cutoff=0. if not bad else 2.**math.ceil(math.log2(max(bad)))
 assert all(terms(u,factors)[0]==u for u in ds if u>=exact_cutoff)
 return dict(factors=factors,coefficients=[1/f for f in factors],max_cast_inputs=maxinputs,max_residuals=maxres,
             max_normalized_abs_error=worst,largest_inexact=max(bad) if bad else None,
             sufficient_exact_threshold=exact_cutoff,exact_count=len(ds)-len(bad),inexact_count=len(bad),
             worst_examples=worsts,sdk_rne_point_comparisons=checks,negative_symmetry_checked=True)

def generator_support():
 # BF16(RNE) of [-1+2*k/2^24]: nonnegative values below2^-16 are n*2^-23;
 # above2^-16 every BF16 lattice point is reached. +1 occurs via rounding.
 out={n*2.**-23 for n in range(128)}
 lo=bits(2.**-16)>>16;hi=bits(1.)>>16
 out.update(struct.unpack('<f',struct.pack('<I',b<<16))[0] for b in range(lo,hi+1))
 for x in out:
  k=round((x+1)*2**23);k=min(k,2**24-1)
  assert bf16(f32(-1+f32(2*f32(k/2**24))))==x
 return sorted(out)

def generator_grid_check(factors):
 vals=generator_support();tested=0;bad=[]
 # Every possible nonzero amax binade in the public O activation support.
 for ae in range(-23,1):
  s=2.**(ae-7)
  for x in vals:
   if x<2.**(ae+1):
    u=f32(x/s);got=f32(terms(u,factors)[0]*s);tested+=1
    if got!=x:bad.append(dict(x=x,scale=s,got=got))
 return dict(nonnegative_support_values=len(vals),positive_amax_binades=24,admissible_value_scale_pairs=tested,
             mismatches=len(bad),examples=bad[:8],scope='Scalar representation only; all pointwise reachable O BF16 values under all admissible power-of-two amax binades. Not all jointly possible PRNG seeds or NPU arithmetic.')

def small_fixtures(factors):
 cases=[]
 for label,anchor,tiny,gamma in [('e47_case7',1.,2.**-22,.5),('residual_scale_prior_counterexample',.53125,2.**-23,.875)]:
  scale=2.**(math.floor(math.log2(anchor))-7)
  xs=[anchor,-anchor]+[tiny]*254;ws=[1.,1.]+[240.]*254
  rows=[terms(f32(x/scale),factors) for x in xs]
  restored=[f32(t[0]*scale) for t in rows]
  raw=sum(x*w for x,w in zip(xs,ws))*16
  # Exact dyadic sums are kept in f64 for semantic term-boundary isolation.
  td=[f32(sum(rows[j][1][i]['quantized']*ws[j] for j in range(256))) for i in range(4)]
  norm=f32(sum(f32(td[i]/factors[i]) for i in range(4)))
  common=f32(norm*scale)*16
  weighted=f32(sum(f32(td[i]*f32(scale/factors[i])) for i in range(4)))*16
  cases.append(dict(name=label,anchor=anchor,tiny=tiny,input_scale=scale,
                    activation_exact=all(a==b for a,b in zip(xs,restored)),raw_reference=raw,
                    raw_common_scale_after_term_sum=common,raw_per_term_scale_before_sum=weighted,
                    term_dots_one256=td,channel_scale=2.**-10,gamma=gamma,residual=.5))
 return cases

def record(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

def calculate():
 ds=domain();fixed=[1.,16.,256.,4096.]
 # Derive each later multiplier from the measured finite normalized BF16-significand residual set.
 prefix=[1.,16.];bounds=[]
 for step in (2,3):
  mr=max(abs(f32(u-terms(u,prefix)[0])) for u in ds)
  fac=2.**math.floor(math.log2(448/mr));bounds.append(dict(term=step,max_residual=mr,chosen_largest_safe_power2=fac,max_input=mr*fac,next_power2_max_input=mr*fac*2))
  prefix.append(fac)
 sdk=subprocess.check_output(['docker','exec','moa2026-dev','cat',SDK])
 for c in range(127):assert e4(val(c))==val(c)
 schemes={'requested_fixed':analyze(ds,fixed),'residual_bound_derived':analyze(ds,prefix)}
 result=dict(cpu_only=True,no_npu_or_compile=True,domain=dict(count=len(ds),min=ds[1],max=ds[-1],definition='Nonnegative F32 values with at most8 significand bits through smallest F32 subnormal, <=255; negatives checked by symmetry.'),
             sdk=dict(container='moa2026-dev',version='0.8.1',path=SDK,sha256=hashlib.sha256(sdk).hexdigest(),max_finite=448.,min_subnormal=2.**-9),schemes=schemes,
             derived_multiplier_evidence=bounds,
             public_o_generator_scalar_checks={k:generator_grid_check(v['factors']) for k,v in schemes.items()},
             semantic_fixture_models={k:small_fixtures(v['factors']) for k,v in schemes.items()},
             source_evidence=[record(ROOT/p) for p in ['furiosa-opt-gemma4-12B/scripts/fixture_prng.py','furiosa-opt-gemma4-12B/scripts/generate_references.py','furiosa-opt-gemma4-12B/scripts/reference/gemma4.py','furiosa-opt-gemma4-12B/src/bin/test_kernels.rs','docs/research/output-boundary-design-audit.md']],
             limits=['Not a NPU floating-point reducer/FTZ/lookup emulator.','Power-of-two scale floor still required for all-zero/tiny inputs.','General BF16 normalized values below exact threshold can lose information; F32 x/s can itself underflow.','Per-term coefficient s/f can underflow at extreme scales and term dot*s can overflow before cancellation; common scale after term reduction has distinct limits.','No general output/RMS error bound or whole-kernel correctness is established.','Public O pointwise generator support is narrower than arbitrary BF16 values merely lying numerically within [-1,1].','QKV post-RMS activation has no inherited24-bit O-grid guarantee.'])
 return result
if __name__=='__main__':
 r=calculate();Path('/tmp/moa-e4-fourterm-audit.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps({k:r[k] for k in ['domain','schemes','derived_multiplier_evidence','public_o_generator_scalar_checks','semantic_fixture_models']},indent=2))
