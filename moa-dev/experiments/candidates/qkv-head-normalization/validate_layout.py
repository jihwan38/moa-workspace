"""Check every renamed projection output index; this is not an NPU oracle."""
from pathlib import Path

root = Path(__file__).parent
query_indices = set()
kv_indices = set()
for cluster in range(2):
    for row_slice in range(256):
        for element in range(8):
            original_row = 2048 * cluster + 8 * row_slice + element
            ns = 4 * cluster + row_slice // 64
            gs = (row_slice // 32) % 2
            ds = 8 * (row_slice % 32) + element
            renamed_row = (ns * 2 + gs) * 256 + ds
            assert original_row == renamed_row
            query_indices.add(renamed_row)
        for element in range(4):
            original_row = 1024 * cluster + 4 * row_slice + element
            ns = 4 * cluster + row_slice // 64
            ds = 4 * (row_slice % 64) + element
            renamed_row = ns * 256 + ds
            assert original_row == renamed_row
            kv_indices.add(renamed_row)

assert query_indices == set(range(4096))
assert kv_indices == set(range(2048))
original_projection = (root / "projection.original.rs").read_text()
candidate_projection = (root / "projection.rs").read_text()
assert original_projection.split("pub(crate) fn project_output(", 1)[1] == candidate_projection.split(
    "pub(crate) fn project_output(", 1
)[1]
original_rope = (root / "rope.original.rs").read_text()
candidate_rope = (root / "rope.rs").read_text()
assert original_rope.split("    let first_half_q", 1)[1] == candidate_rope.split("    let first_half_q", 1)[1]
print("PASS: all Q/KV indices and unchanged O/RoPE suffixes")
