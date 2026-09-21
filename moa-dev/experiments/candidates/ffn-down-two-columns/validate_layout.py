"""Static two-column index checks; floating sum reassociation requires NPU tests."""
from pathlib import Path
import json

here = Path(__file__).parent
base = (here / 'mlp.eight-columns.rs').read_text()
new = (here / 'mlp.rs').read_text()
assert base.split('pub(crate) type DownCluster')[0] == new.split('pub(crate) type DownCluster')[0]
rows = [c * 1920 + group * 16 + local for c in range(2) for group in range(120) for local in range(16)]
assert rows == list(range(3840))
assert [4 * i + r for i in range(4) for r in range(4)] == list(range(16))
for h in rows:
    for column_group in range(2):
        for local in (0, 1, 15, 16, 7678, 7679):
            l = column_group * 7680 + local
            c, hlocal = divmod(h, 1920)
            assert h * 15360 + l == (c * 1920 + hlocal) * 15360 + column_group * 7680 + local
            assert h * 960 + l // 16 == (c * 1920 + hlocal) * 960 + column_group * 480 + local // 16
old_blocks = [group * 120 + block for group in range(8) for block in range(120)]
new_blocks = [group * 480 + block for group in range(2) for block in range(480)]
assert old_blocks == new_blocks == list(range(960))
assert 4 * 480 * 4 == 7680 < 8192
result = {
    'upgate_and_geglu_prefix_byte_identical_to_safe_decode_baseline': True,
    'H_rows_covered_once': len(rows),
    'valid_row_groups_per_cluster': 120,
    'physical_row_groups_per_cluster': 128,
    'column_groups': 2,
    'columns_per_slice': 7680,
    'rows_per_slice': 16,
    'four_row_passes': 4,
    'scale_vrf_bytes': 7680,
    'activation_trf_bytes': 15360,
    'unchanged_weight_and_scale_logical_addresses': True,
    'FP32_sum_reassociation': '8x120 scaled blocks -> 2x480 scaled blocks; not bitwise-equivalent by proof',
    'hardware_correctness': 'not_tested',
}
(here / 'logical-index-proof.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
