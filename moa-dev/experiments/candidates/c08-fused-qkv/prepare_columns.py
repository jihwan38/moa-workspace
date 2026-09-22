#!/usr/bin/env python3
"""Generate an eight-column QKV design from E104, independently of failed fusion."""
import json
import subprocess
from build import HERE, common

variant = HERE / "c03-columns8"
variant.mkdir(exist_ok=True)
sources = {"projection.rs": "src/device/sliding/projection.rs", "rmsnorm.rs": "src/device/sliding/rmsnorm.rs", "rope.rs": "src/device/sliding/rope.rs", "ops.rs": "src/ops.rs"}
original = {}
for name, path in sources.items():
    original[name] = subprocess.check_output(["git", "show", common.BASE + ":" + common.CRATE + "/" + path], cwd=common.ROOT).decode()
    (variant / name).write_text(original[name])
(variant / "projection.rs").write_text(original["projection.rs"] + "\n" + (HERE / "column_split.rs").read_text())
ops = original["ops.rs"]
start = ops.index("    let x: DmTensor<bf16, Chip, m![2], m![Dummy256 / 32, 1 # 32], m![H]> = x.to_dm(&mut device.tdma);")
end = ops.index("\n    let q: DmTensor", start)
ops = ops[:start] + ops[end:]
ops = ops.replace("sliding::projection::project_query(device", "sliding::projection::project_query_columns8(device", 1)
ops = ops.replace("sliding::projection::project_key_value(device", "sliding::projection::project_key_value_columns8(device", 1)
(variant / "ops.rs").write_text(ops)
record = dict(base=common.BASE, runtime_changes=[sources[n] for n in sources if (variant / n).read_text() != original[n]],
              variant=variant.name, files={sources[n]: common.digest((variant / n).read_bytes()) for n in sources})
(HERE / "generated-columns.json").write_text(json.dumps(record, indent=2) + "\n")
