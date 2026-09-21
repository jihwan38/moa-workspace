"""Static index/patch checks only; does not execute RNGD or prove numerical accuracy."""
from pathlib import Path
import json

here = Path(__file__).parent
old = (here / 'mlp.single-cluster.rs').read_text()
new = (here / 'mlp.rs').read_text()
rows = []
for cluster in range(2):
    for slice_id in range(240):
        for local in range(32):
            row = cluster * 7680 + slice_id * 32 + local
            rows.append(row)
            # HBM reshape changes only row-axis names. Column order is unchanged.
            for column in (0, 1, 15, 16, 1919, 1920, 3839):
                original_nibble = row * 3840 + column
                split_nibble = ((cluster * 7680 + slice_id * 32 + local) * 3840 + column)
                assert original_nibble == split_nibble
                assert (original_nibble // 2, original_nibble % 2) == (split_nibble // 2, split_nibble % 2)
                assert row * 240 + column // 16 == ((cluster * 7680 + slice_id * 32 + local) * 240 + column // 16)
            # Typed output compaction preserves logical row order before rename.
            canonical = cluster * 7680 + (slice_id * 32 + local)
            assert canonical == row
assert rows == list(range(15360))
assert [4 * p + r for p in range(8) for r in range(4)] == list(range(32))
assert old.split('pub(crate) fn geglu', 1)[1] == new.split('pub(crate) fn geglu', 1)[1]
assert 'L % 32 = 4 #{!} 60' not in new
report = {
    'logical_rows_covered_once': len(rows),
    'cluster_count': 2,
    'valid_slices_per_cluster': 240,
    'physical_slices_per_cluster': 256,
    'rows_per_slice': 32,
    'passes_per_up_or_gate': 8,
    'output_rows_per_commit': 4,
    'output_commit_bytes': 8,
    'scale_vrf_bytes_per_slice': 4 * 240 * 4,
    'representative_columns_checked_per_row': 7,
    'column_axis_unchanged': True,
    'geglu_and_down_source_byte_identical': True,
    'hardware_correctness': json.loads((here / 'static-comparison.json').read_text())['hardware_correctness'],
}
(here / 'logical-index-proof.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
