#!/usr/bin/env python3
"""Independent rational ceiling proof and general BF16 fixture error survey."""
from fractions import Fraction
from pathlib import Path
import json, math, random, struct
FLOOR=Fraction(1,2**112)
def rational(b):
    sign=-1 if b&0x8000 else 1
    e=(b>>7)&255;m=b&127
    if e==255:raise ValueError('nonfinite')
    if e==0:return Fraction(0)  # every BF16 subnormal is F32 subnormal: FTZ
    return sign*Fraction(128+m)*Fraction(2)**(e-134)
def frombits(b):return struct.unpack('<f',struct.pack('<I',b<<16))[0]
def bf16(x):
    b=struct.unpack('<I',struct.pack('<f',x))[0]
    return ((b+0x7fff+((b>>16)&1))>>16)&65535
def scale_bits(b):
    b=max(b&0x7fff,15<<7)<<16  # floor2^-112
    return ((b+0x1fffff)&0x7f800000)-0x1000000
def golden_exp(a):
    a=max(a,FLOOR)
    # Integer/rational comparisons; no hardware-style bit formula or division rounding.
    e=a.numerator.bit_length()-a.denominator.bit_length()-3
    while Fraction(7)*Fraction(2)**e < a:e+=1
    while Fraction(7)*Fraction(2)**(e-1)>=a:e-=1
    return e
def qfrac(x,e):
    v=x/(Fraction(2)**e);n=abs(v.numerator);d=v.denominator
    q=(2*n+d)//(2*d)
    return max(-7,min(7,-q if v<0 else q))
# Exhaust every finite positive BF16 maximum, including zero/subnormals and threshold neighbors.
count=0
for b in range(0x7f80):
    e=golden_exp(rational(b));sb=scale_bits(b)
    assert sb==(e+127)<<23,(b,e,sb)
    rb=0x7f000000-sb
    assert rb==(-e+127)<<23 and 0<rb<0x7f800000
    assert 0<sb<0x7f800000
    count+=1
# Every finite signed BF16 value under every possible normal power2 scale this contract emits.
# Product rounding cannot affect integer rounding: BF16 has<=8 significant bits; normal
# power2 scaling is exact. Any flushed result is far below the firsthalf boundary.
products=0
for e in range(-114,127):
 for b in range(0x10000):
    if (b&0x7f80)==0x7f80:continue
    x=rational(b)
    if abs(x)>7*(Fraction(2)**e):continue
    v=float(x)*math.ldexp(1.,-e)
    if abs(v)<2.**-126:v=math.copysign(0.,v)
    q=math.floor(abs(v)+.5)*(1 if v>=0 else -1)
    assert q==qfrac(x,e),(e,b,q,qfrac(x,e))
    products+=1
rng=random.Random(20261025)
results=[]
for width in [3840,15360]:
 for name in ['gaussian','laplace','block_scales','sparse_outliers']:
  vals=[]
  for r in range(16):
   row=[]
   for k in range(width):
    if name=='gaussian':x=rng.gauss(0,1)
    elif name=='laplace':x=(-1 if rng.random()<.5 else 1)*(-math.log(max(rng.random(),2.**-24)))
    elif name=='block_scales':x=rng.gauss(0,1)*2.**((k//960+r)%7-3)
    else:x=rng.gauss(0,.25) if k%997 else rng.choice([-1,1])*16
    row.append(bf16(x))
   vals.append(row)
  norm=sum(frombits(b)**2 for row in vals for b in row)
  metrics={}
  for block in [width,960]:
   err=0.;zero=0;codes=0
   for row in vals:
    for offset in range(0,width,block):
     sub=row[offset:offset+block];ab=max(b&0x7fff for b in sub);e=(scale_bits(ab)>>23)-127;s=2.**e
     for b in sub:
      x=frombits(b);q=math.floor(abs(x/s)+.5)*(1 if x>=0 else -1);q=max(-7,min(7,q));err+=(q*s-x)**2;zero+=q==0;codes+=1
   metrics[str(block)]={'relative_l2':math.sqrt(err/norm),'zero_fraction':zero/codes,'scale_count':len(vals)*width//block}
  results.append({'fixture':name,'width':width,'rows':16,'metrics':metrics})
result={'status':'PASS','finite_abs_bf16_scale_patterns':count,'admissible_finite_bf16_power2_products':products,'scale_exp_range':[-114,126],'rounding':'nearest ties away from zero','ftz':'BF16/F32 subnormal inputs before max and multiplication; tiny products also flush, integercode remains0','fixture_survey':results,'limitations':['No provided weight files or captured model activations available locally. Fixtures are generic deterministic distributions, not model-quality evidence.','Row vs960block scale changes consumer metadata and dot scaling, not just quantizer cost.','Old amax/7 DivF guard failure remains unrepaired and unadopted.']}
Path(__file__).with_name('cpu-proof.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
