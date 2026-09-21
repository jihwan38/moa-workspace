"""Verify the intended packed component/row maps and preserved source tails."""
from pathlib import Path
import json

p = Path(__file__).parent
base = p.parent / "qkv-fp8-direct-packed-quant"
source = (p / "projection.rs").read_text()
original = (base / "projection.rs").read_text()
assert (p / "ops.rs").read_bytes() == (base / "ops.rs").read_bytes()
assert source.split("type KvCluster", 1)[0] == original.split("type KvCluster", 1)[0]
assert source.split("pub(crate) fn project_output", 1)[1] == original.split("pub(crate) fn project_output", 1)[1]
key = source.split("fn project_one_kv_matrix(", 1)[1].split("fn project_value_two_dot(", 1)[0]
old_key = original.split("fn project_one_kv_matrix(", 1)[1].split("pub(crate) fn project_key_value(", 1)[0]
assert key == old_key
value = source.split("fn project_value_two_dot(", 1)[1].split("pub(crate) fn project_key_value(", 1)[0]
assert value.split("    // Four valid local rows", 1)[1] == key.split("    // Four valid local rows", 1)[1]

# Same BF16/scaling mathematics as E19, with local row Ps%4 renamed Ps/8%4.
e19 = (p.parent / "qkv-fp8-split/projection.rs").read_text()
e19 = e19.split("fn project_one_kv_matrix(", 1)[1]
dot = e19[e19.index("    let weight_f8:"):e19.index("    let weight_scale: DmTensor")]
dot = dot.replace("Ps % 4", "Ps / 8 % 4").replace("(lo_trf)", "(&lo_trf)").replace("(hi_trf)", "(&hi_trf)")
assert dot == value[value.index("    let weight_f8:"):value.index("    // Four valid local rows")]

component_addresses = [[component * 3840 + h for h in range(3840)] for component in range(2)]
assert set(component_addresses[0]).isdisjoint(component_addresses[1])
assert sorted(component_addresses[0] + component_addresses[1]) == list(range(7680))
seen = set()
for row in range(2048):
    cluster = row // 1024
    group = (row // 32) % 32
    local = (row // 8) % 4
    low_slice = row % 8
    physical_slice = group * 8 + low_slice
    seen.add((cluster, physical_slice, local))
    assert cluster * 1024 + group * 32 + local * 8 + low_slice == row
    # Existing Switch moves local4 to padded slice-low4#8 and low_slice to Time8.
    restored_row = cluster * 1024 + (group * 4 + local) * 8 + low_slice
    assert restored_row == row
assert len(seen) == 2048
print(json.dumps({
    "ops_q_k_o_quantizer_unchanged": True,
    "v_switch_scale_head_tail_identical_to_k": True,
    "v_dot_matches_e19_after_local_row_axis_rename": True,
    "packed_component_addresses_checked": 7680,
    "kv_row_bijection_checked": 2048,
    "scope": "Source and intended logical mapping only; not compiler/NPU proof",
}, indent=2))
