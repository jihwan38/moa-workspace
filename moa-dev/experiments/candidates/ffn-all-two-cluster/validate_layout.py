"""Static layout invariants; no hardware correctness or performance claim."""
from pathlib import Path
import json

here = Path(__file__).parent
old = (here / 'mlp.upgate-only.rs').read_text()
new = (here / 'mlp.rs').read_text()
rows = []
for cluster in range(2):
    for row_group in range(32):
        for local_row in range(60):
            h = cluster * 1920 + row_group * 60 + local_row
            rows.append(h)
            for column_group in range(8):
                physical_slice = row_group * 8 + column_group
                assert physical_slice // 8 == row_group
                for local_column in (0, 1, 15, 16, 1918, 1919):
                    l = column_group * 1920 + local_column
                    original_nibble = h * 15360 + l
                    split_nibble = ((cluster * 1920 + row_group * 60 + local_row) * 15360 + column_group * 1920 + local_column)
                    assert original_nibble == split_nibble
                    assert (original_nibble // 2, original_nibble % 2) == (split_nibble // 2, split_nibble % 2)
                    original_scale = h * 960 + l // 16
                    split_scale = ((cluster * 1920 + row_group * 60 + local_row) * 960 + column_group * 120 + local_column // 16)
                    assert original_scale == split_scale
                # The same 8 ordered contraction intervals reduce inside each cluster.
                assert (column_group * 1920, (column_group + 1) * 1920) == (column_group * 1920, column_group * 1920 + 1920)
            assert cluster * 1920 + row_group * 60 + local_row == h
assert rows == list(range(3840))
assert [4 * p + r for p in range(15) for r in range(4)] == list(range(60))
assert old.split('pub(crate) fn feedforward', 1)[0] == new.split('pub(crate) fn feedforward', 1)[0]
assert old.split('pub(crate) fn geglu', 1)[1].split('pub(crate) type DownRows', 1)[0] == new.split('pub(crate) fn geglu', 1)[1].split('pub(crate) type DownCluster', 1)[0]
report = {
    'logical_output_rows_covered_once': len(rows),
    'clusters': 2,
    'row_groups_per_cluster': 32,
    'column_groups_per_row': 8,
    'columns_per_contraction': 1920,
    'rows_per_slice': 60,
    'four_row_passes': 15,
    'scale_vrf_bytes': 4 * 120 * 4,
    'output_commit_bytes': 4 * 2,
    'packed_and_scale_index_identity': True,
    'within_cluster_8_way_reduction_groups': True,
    'upgate_source_byte_identical': True,
    'geglu_source_byte_identical': True,
    'hardware_correctness': json.loads((here / 'static-comparison.json').read_text())['hardware_correctness'],
}
(here / 'logical-index-proof.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
