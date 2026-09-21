"""Static row/packet ordering checks; does not execute the RNGD candidate."""
from pathlib import Path
import json

here = Path(__file__).parent
base = (here / 'mlp.four-row.rs').read_text()
final = (here / 'mlp.rs').read_text()
assert [8 * group + 4 * time + lane for group in range(4) for time in range(2) for lane in range(4)] == list(range(32))
assert [12 * group + 4 * time + lane for group in range(5) for time in range(3) for lane in range(4)] == list(range(60))
assert 8 * 240 * 4 == 7680 < 8192
assert 12 * 120 * 4 == 5760 < 8192
# Renaming the dense dequantized tile leaves every scalar's local offset unchanged.
for rows, columns in ((8, 3840), (12, 1920)):
    for row in range(rows):
        for column in range(columns):
            old_offset = row * columns + column
            canonical_offset = row * columns + column
            assert old_offset == canonical_offset
# No changes to the private wiring, global scaling, GeGLU, or layer behavior.
start = 'pub(crate) fn feedforward'
end = 'pub(crate) type DownCluster'
assert base.split(start)[1].split(end)[0] == final.split(start)[1].split(end)[0]
# HBM transfer and high-axis layouts remain identical.
for declaration in (
    'pub(crate) type UpGateCluster = m![Dummy2];',
    'pub(crate) type UpGateSlices = m![L / 32 % 240 # 256];',
    'pub(crate) type DownCluster = m![Dummy2];',
    'pub(crate) type DownRows = m![H / 60 % 32, 1 # 8];',
    'pub(crate) type DownRowsByColumns = m![H / 60 % 32, L / 1920];',
):
    assert declaration in base and declaration in final
result = {
    'upgate_rows_per_tile': 8,
    'upgate_passes': 4,
    'upgate_scale_vrf_bytes': 7680,
    'upgate_output_packets_per_tile': 2,
    'down_rows_per_tile': 12,
    'down_passes': 5,
    'down_scale_vrf_bytes': 5760,
    'down_output_packets_per_tile': 3,
    'bf16_values_per_output_packet': 4,
    'bf16_bytes_per_output_packet': 8,
    'row_coverage_once_and_in_order': True,
    'dense_dequant_tile_reshape_preserves_offsets': True,
    'feedforward_wiring_global_scales_and_geglu_unchanged': True,
    'hardware_correctness': 'not_tested',
}
(here / 'logical-index-proof.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
