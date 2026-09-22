#!/usr/bin/env python3
"""Generate the joint Q/K/V projection while retaining public signatures and epilogues."""
import json
import subprocess
from build import HERE, common

variant = HERE / "c03-fused-contiguous"
variant.mkdir(exist_ok=True)
sources = {"projection.rs": "src/device/sliding/projection.rs", "rmsnorm.rs": "src/device/sliding/rmsnorm.rs", "rope.rs": "src/device/sliding/rope.rs", "ops.rs": "src/ops.rs"}
original = {}
for name, path in sources.items():
    original[name] = subprocess.check_output(["git", "show", common.BASE + ":" + common.CRATE + "/" + path], cwd=common.ROOT).decode()
    (variant / name).write_text(original[name])
(variant / "projection.rs").write_text(original["projection.rs"] + "\n" + (HERE / "fused.rs").read_text())
old = """    let q: DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Gs, Ds]> =
        sliding::projection::project_query(device, &x, q_weight, q_weight_scale);
    let (k, v) = sliding::projection::project_key_value(device, &x, k_weight, v_weight, k_weight_scale, v_weight_scale);"""
assert original["ops.rs"].count(old) == 1
new = """    let (q, k, v) = sliding::projection::project_qkv_fused(
        device, &x, q_weight, k_weight, v_weight, q_weight_scale, k_weight_scale, v_weight_scale);"""
(variant / "ops.rs").write_text(original["ops.rs"].replace(old, new))
record = dict(base=common.BASE, runtime_changes=[sources[n] for n in sources if (variant / n).read_text() != original[n]],
              variant="c03-fused-contiguous", files={sources[n]: common.digest((variant / n).read_bytes()) for n in sources})
(HERE / "generated.json").write_text(json.dumps(record, indent=2) + "\n")
