#!/usr/bin/env python3
"""Read-only exact E146 v1 constant-bool tensor-phi evidence; no compiler calls."""
import json,re,hashlib
from pathlib import Path
C=Path(__file__).resolve().parent
R=C.parents[3]
p=C/'evidence/e146-qk-half-v1/failure.visa'
s=p.read_text(); src=C/'qk-half-v1/projection.rs'; source=src.read_text()
log=C/'evidence/e146-qk-half-v1/compile.log'
def rec(p):
 b=p.read_bytes(); return {'path':str(p.relative_to(R)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def hit(t):
 assert s.count(t)==1,(t,s.count(t))
 n=s.index(t); return {'offset_bytes':n,'exact_text':t,'context':s[max(0,n-130):n+len(t)+200]}
phis=[]
for projection,const,lhs,rhs1,rhs2 in [('K','true',399,395,398),('V','false',429,425,428)]:
 condition=f'Branch {{ condition: Const(Bool({const})), then_branch:'
 phi=f'PhiFunction {{ lhs: TensorVar({lhs}), rhs1: TensorVar({rhs1}), rhs2: TensorVar({rhs2}) }}'
 assert 'QkvKvRow' in s[s.index(phi):s.index(phi)+650]
 phis.append({'projection':projection,'branch':hit(condition),'phi':hit(phi),'lhs':lhs,'staged_write_rhs':rhs1,'full_native_rhs':rhs2,'next_consumer':'LowLevelReshape to canonical head mapping'})
assert 'k_weight, k_weight_scale, true)' in source and 'v_weight, v_weight_scale, false)' in source
assert 'staged: bool' in source
assert 'T170 := Scratchpad()' in log.read_text() and 'wait_targets: [T185], size: 0' in log.read_text()
assert 'Scratchpad' not in s and 'TensorVar(170)' not in s
r={'valid_evidence':True,'candidate':'E146','variant':'qk-half-v1','inputdigests':[rec(q) for q in [src,p,log]],'source_condition':'K uses true; V uses false; a tensor-valued if staged selects staged vs single-full native.', 'observed_visa_branches':phis,'scheduler_failure':{'tensor':'T170','wait_target':'T185','requested_bytes':0,'visa_to_scheduler_identity_available':False},'full_raw_declared_shape':{'Q':'[Chip1,ClusterDummy2,SliceGroup64×(1#4),ElementRow32] F32','K':'[Chip1,ClusterDummy2,SliceGroup64×(1#4),ElementRow16] F32','Q_live_bytes':16384,'K_live_bytes':8192},'inference':'Removing the tensor-returning constant-bool branch by separate K/V functions eliminates these two observed phi nodes without extra memory traffic. It is a smaller supported source-expression repair to test, not a proved cause.', 'limits':['The pre-scheduling vISA dump does not expose scheduler T170/T185 identity or allocation wait graph.','A size0 AllocRequest does not demonstrate an empty logical F32 tensor.','No claim that the scheduler Scratchpad is produced by a phi until the split-function exact build tests that mechanism.']}
out=Path(__file__).with_suffix('.json');out.write_text(json.dumps(r,indent=2)+'\n');print(out)
