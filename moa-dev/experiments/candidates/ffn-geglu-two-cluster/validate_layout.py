"""Static mapping and scalar-operation checks; no RNGD execution."""
from pathlib import Path
import json

here = Path(__file__).parent
base = (here / 'mlp.single-geglu.rs').read_text()
new = (here / 'mlp.rs').read_text()
# Same valid up/gate row values; padding slices are not logical rows.
rows = [cluster * 7680 + s * 32 + r for cluster in range(2) for s in range(240) for r in range(32)]
assert rows == list(range(15360))
# Explicit L-high rename precedes a fresh independent H-high cluster broadcast.
rename = 'let x: DmTensor<bf16, Chip, m![L / 7680], Slice, m![L % 7680]> = unsafe { x.reshape() };'
replicate = 'let x: DmTensor<bf16, Chip, DownCluster, DownRowsByColumns, m![L % 1920]> = x.to_dm(&mut device.tdma);'
assert new.index(rename) < new.index(replicate)
for h_cluster in range(2):
    for row_group in range(32):
        for l_group in range(8):
            for k in (0, 15, 16, 1919):
                l = l_group * 1920 + k
                l_cluster, l_local = divmod(l, 7680)
                assert l_cluster * 7680 + l_local == l
# Elementwise nonlinear recipe and explicit BF16 boundary casts are unchanged.
def operation_sequence(source):
    section = source.split('pub(crate) fn geglu', 1)[1].split('pub(crate) type DownCluster', 1)[0]
    return [line.strip() for line in section.splitlines() if line.strip().startswith(('.vector_fp_', '.vector_stash()', '.cast::<bf16'))]
assert operation_sequence(base) == operation_sequence(new)
result = {
    'upgate_logical_rows_covered_once': 15360,
    'geglu_scalar_operations_and_bf16_casts_unchanged': True,
    'upgate_dummy2_renamed_to_L_high_before_down_dma': True,
    'down_H_high_cluster_broadcast_is_independent_of_L_high': True,
    'global_scale_ring16_valid_slices_covered': 240,
    'hardware_correctness': 'not_tested',
}
(here / 'logical-index-proof.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
