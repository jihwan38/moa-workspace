#!/usr/bin/env python3
"""Read-only source boundary audit; no compiler or device invocation."""
from pathlib import Path
import hashlib, json, subprocess
root=Path(__file__).resolve().parents[2]
base='1a835e18fa44d6fa308fde4eb4788eab56a7202f'
def old(p): return subprocess.check_output(['git','show',base+':'+p],cwd=root)
def sha(b): return hashlib.sha256(b).hexdigest()
files=subprocess.check_output(['git','ls-tree','-r','--name-only',base,'src','Cargo.toml','rust-toolchain.toml'],cwd=root).decode().splitlines()
for p in files: assert (root/p).read_bytes()==old(p),p
a=old('src/bin/test_int4_complete_ffn.rs').decode()
b=(root/'src/bin/test_int4_drift_attribution.rs').read_text()
shared=a[a.index('async fn upload'):a.index('#[tokio::main]')]
assert b[b.index('async fn upload'):b.index('#[path="support/int4_drift_metrics.rs"]')].strip()==shared.strip()
assert '#[device' not in b
assert '#[device' not in (root/'src/bin/support/int4_drift_metrics.rs').read_text()
lock=sha((root/'Cargo.lock').read_bytes())
assert lock=='a99652661c588fe5c3a8b23278f2390d8cd2e99f2b33bcab6f52e97dbba04e2e'
images=json.loads((root/'experiments/int4-complete-ffn/a005-image-audit.json').read_text())['current']
print(json.dumps({'base':base,'all_prior_source_files_unchanged':len(files),'shared_complete_helpers_sha256':sha(shared.encode()),
 'independent_quotient_oracle_sha256':sha((root/'src/bin/support/rational_a4_oracle.rs').read_bytes()),
 'original_lock_sha256':lock,'new_device_functions':0,'expected_reused_actual_images':images['records'],
 'existing_source_images_not_bound_by_prior_complete_ELF':['furiosa_opt_gemma4::ops_native_fp8_p512_blocks::quantize_hidden','furiosa_opt_gemma4::ops_native_fp8_p512_blocks::quantize_ffn'],
 'future_device_artifact_parity':'required; not yet built'},indent=2))
