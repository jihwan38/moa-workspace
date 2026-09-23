#!/usr/bin/env python3
"""Check logical ownership and HBM pitches, independently of SDK lowering."""
import hashlib
import json
from pathlib import Path
import re
import argparse

HERE = Path(__file__).resolve().parent
H, L, TILE = 3840, 15360, 7680


def verify(variant):
    produced = {}
    for stage in range(2):
        for cluster in range(2):
            for slot in range(240):
                for row in range(16):
                    logical = stage * TILE + cluster * 3840 + slot * 16 + row
                    assert logical not in produced
                    produced[logical] = (stage, cluster, slot, row)
    assert set(produced) == set(range(L))

    # Each H row receives exactly eight disjoint L intervals: two stages,
    # two L-owned clusters, and two slice columns. Check producer ownership
    # for every L coordinate and all row-group/column correspondences.
    consumers = 0
    for stage in range(2):
        for cluster in range(2):
            for column in range(2):
                for local_l in range(1920):
                    logical = stage * TILE + cluster * 3840 + column * 1920 + local_l
                    assert produced[logical][:2] == (stage, cluster)
                    assert produced[logical][2:] == (column * 120 + local_l // 16, local_l % 16)
                    consumers += 1
    assert consumers == L

    for h in range(H):
        row_group, local_h = divmod(h, 30)
        assert row_group < 128 and local_h < 30
        intervals = []
        for stage in range(2):
            for cluster in range(2):
                for column in range(2):
                    logical = stage * TILE + cluster * 3840 + column * 1920
                    slot = row_group * 2 + column
                    leader = (slot // 32) * 32 + column
                    assert slot < 256 and leader % 32 in (0, 1)
                    assert leader % 2 == column
                    intervals.append((logical, logical + 1920))
                    # Keep the original HBM pitch across H, even for stage 1.
                    packed_address = h * (L // 2) + logical // 2
                    scale_address = h * (L // 16) + logical // 16
                    assert packed_address % 8 == scale_address % 8 == 0
                    assert packed_address + 960 <= (h + 1) * (L // 2)
                    assert scale_address + 120 <= (h + 1) * (L // 16)
        assert sorted(intervals) == [(i, i + 1920) for i in range(0, L, 1920)]

    for global_l in range(L):
        assert (global_l * (H // 2)) % 8 == 0
        assert (global_l * (H // 16)) % 8 == 0
    assert 8 * (H // 16) * 4 == 7680 < 8192  # Up/gate scale VRF.
    assert 6 * 120 * 4 == 2880 < 8192       # Down scale VRF.
    assert 2 * 480 * 4 == 3840 < 8192      # Final stage partial VRF.

    source = (HERE / variant / 'mlp.rs').read_text()
    old = (HERE.parent / 'ffn-geglu-two-cluster/mlp.rs').read_text()
    def scalar_ops(text):
        body = text.split('pub(crate) fn geglu(', 1)[1].split('\n}\n', 1)[0]
        return re.findall(r'\.vector_fp_[^\n]+|\.cast::<bf16[^\n]+', body)
    # Compare the actual scalar operation and cast sequence after normalizing
    # coordinate names. This does not establish physical padding or NPU accuracy.
    new_ops = [x.replace('TileL', 'L') for x in scalar_ops(source)]
    assert new_ops == scalar_ops(old)
    return {
        'scope': 'Logical coordinate, pitch and GeGLU scalar-sequence checks only; not NPU or compiler-layout validation.',
        'producer_coordinates_checked': len(produced),
        'consumer_l_coordinates_checked': consumers,
        'down_rows_checked': H,
        'down_l_partition': '8 disjoint intervals per H row; every weight and scale selected once',
        'down_packed_hbm_row_pitch_bytes': L // 2,
        'down_scale_hbm_row_pitch_bytes': L // 16,
        'geglu_scalar_and_bf16_cast_sequence_matches_prior': True,
        'f32_down_reduction_order_changed': True,
        'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'variant': variant,
        'result': 'PASS',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variant')
    args = parser.parse_args()
    result = verify(args.variant)
    (HERE / f'logical-layout-{args.variant}.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
