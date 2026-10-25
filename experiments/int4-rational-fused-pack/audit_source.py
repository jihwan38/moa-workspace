#!/usr/bin/env python3
"""Read-only exact extraction audit. No compiler, device, or numerical claims."""
from pathlib import Path
import hashlib, json, re, subprocess
ROOT=Path(__file__).resolve().parents[2]
def sha(x): return hashlib.sha256(x.encode() if isinstance(x,str) else x).hexdigest()
def old(rev,path): return subprocess.check_output(["git","-C",str(ROOT),"show",f"{rev}:{path}"]).decode()
def source(name): return (ROOT/"src"/name).read_text()
f=source("ops_int4_rational_fused_pack.rs")
h=source("ops_int4_rational_lut_p512_compact.rs")
l=source("ops_int4_rational_lut_hidden.rs")
q=lambda t:t[t.index("    let x:"):t.index("    codes.view().to_hbm_view")]
normalized=re.sub(r"\bH\b","K",q(h))
assert normalized==re.sub(r"\bL\b","K",q(l))
assert f[f.index("    let x:"):f.index("    (codes, scale)")]==normalized
s=lambda t:t[t.index("    // Exact scale bits only:"):t.rfind("\n}")]
scale=s(h)
assert scale==s(l)
assert scale in f
orig=source("ops_int4_pack_p512.rs")
assert orig==old("60705649","src/ops_int4_pack_p512.rs")
assert orig.replace("pub(crate) fn pack_dm<","fn pack_dm<",1)==old("bf4bf767","src/ops_int4_pack_p512.rs")
body=orig[orig.index("    // Public AxisName"):orig.index("\n#[device(chip = 1)]")][:-2]
removed="    let input: DmTensor<i32, Chip, Cluster, Rows, m![K]> = input.to_dm(&mut device.tdma);\n"
assert body.count(removed)==1
pack=body.replace(removed,"")
assert pack in f
staged=old("60705649","src/ops_int4_pack_staged_store.rs")
staging=staged[staged.index("    let staged: DmTensor",staged.index("pub fn pack_hidden")):staged.rfind("\n}")]
assert f.count(staging)==2
assert "m![H / 960, P512 % 256 / 4]" not in f
controls=["ops_int4_rational_lut_p512_compact.rs","ops_int4_rational_lut_hidden.rs"]
for name in controls: assert source(name)==old("814c33b7","src/"+name)
for name in ["Cargo.toml","rust-toolchain.toml"]:
 assert (ROOT/name).read_text()==old("814c33b7",name)
result={"result":"SOURCE_EXTRACTION_PASS_NOT_DEVICE_EQUIVALENCE","quantizer_prefix_generic_axis_sha256":sha(normalized),"scale_store_body_sha256":sha(scale),"pack_body_without_input_hbm_load_sha256":sha(pack),"hidden_real_staging_body_sha256":sha(staging),"original_pack_import_sha256":sha(orig),"candidate_sha256":sha(f),"controls":{name:sha(source(name)) for name in controls},"changes":["H/L axis replaced by generic K only within identical quantization prefix","Return codes/scale DM instead of canonical codes HBM write","Pack helper consumes same canonical codes DM, removes only input HBM read","Original H output store retained; validated hidden staging retained","Unchanged scale-store body"]}
print(json.dumps(result,indent=2))
