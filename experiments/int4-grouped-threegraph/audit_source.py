#!/usr/bin/env python3
"""Read-only body/control/dependency preservation proof; no build or device work."""
from pathlib import Path
import hashlib,json,re,subprocess
ROOT=Path(__file__).resolve().parents[2]
BASE='d21224649eef82f5c3bb293aac44b2d1c28b58e8'
def old(path):return subprocess.check_output(['git','show',BASE+':'+path],cwd=ROOT).decode()
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def function(s,name):
 m=re.search(r'(?:pub(?:\(crate\))? )?(?:async )?fn '+name+r'\(',s);assert m,name
 start=m.start();b=s.index(' {',start) if ' { ' in s[start:start+10] else -1
 # All target functions have an ordinary body; find the signature closing parenthesis,
 # then return type if present, and the first opening body brace.
 i=s.index('(',m.start());level=1;i+=1
 while level:
  if s[i]=='(':level+=1
  elif s[i]==')':level-=1
  i+=1
 b=s.index('{',i);j=b+1;level=1
 while level:
  if s[j]=='{':level+=1
  elif s[j]=='}':level-=1
  j+=1
 return s[start:b],s[b:j]
proof=json.loads((ROOT/'experiments/int4-grouped-threegraph/extraction.json').read_text())
for item in proof:
 a=old(item['file']);b=(ROOT/item['file']).read_text()
 _,body=function(a,item['function']);_,got=function(b,item['helper'])
 assert body==got and sha(body)==item['body_sha256'],item
 _,wrapper=function(b,item['function'])
 assert re.sub(r'\s+','',wrapper)=='{'+item['helper']+'('+','.join(item['arguments'])+');}'
# Reconstruct each donor by replacing the new wrapper plus helper with its original
# function, proving no extra source edits outside extraction.
for file in {x['file'] for x in proof}:
 s=(ROOT/file).read_text();expected=old(file)
 for item in [x for x in proof if x['file']==file]:
  ws,wb=function(s,item['function']);hs,hb=function(s,item['helper'])
  os,ob=function(expected,item['function'])
  assert ws==os
  s=s.replace(ws+wb+'\n\n'+hs+hb,os+ob,1)
 assert s==expected,file
unchanged=['src/ops_norm_blocked_residual.rs','src/ops_group_ffn_projection_storegroup.rs',
 'src/bin/test_int4_grouped_complete_ffn.rs','src/bin/support/int4_grouped_complete_fixture.rs',
 'src/bin/support/int4_grouped_activation_fixture.rs','src/host/int4_complete_ffn_weights.rs',
 'src/ops_int4_grouped_activation_reference.rs','src/i4_grouped_activation_raw_adapter.rs','Cargo.toml']
for path in unchanged:assert (ROOT/path).read_text()==old(path),path
probe=(ROOT/'src/bin/test_int4_grouped_threegraph.rs').read_text()
donor=old('src/bin/test_int4_grouped_complete_ffn.rs')
reused=['candidate_prefix','candidate_rest','candidate_chain','candidate_clocked','production_chain','intended_step']
for name in reused:assert function(probe,name)==function(donor,name),name
assert hashlib.sha256((ROOT/'Cargo.lock').read_bytes()).hexdigest()=='a99652661c588fe5c3a8b23278f2390d8cd2e99f2b33bcab6f52e97dbba04e2e'
print(json.dumps({'base':BASE,'exact_extracted_bodies':proof,'unchanged':unchanged,'unchanged_probe_functions':reused,
 'all11_snapshot_bytes':63979520,'three_vs_seven_equality_bytes':4*63979520,
 'three_raw_vs_typed_equality_bytes':4*63979520,'prelaunch_negative_checks':2*(67+28),
 'scope':'source identity only; no compiler or runtime parity implied'},indent=2))
