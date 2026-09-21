"""Finite wire-order and transport-padding audit, not a hardware oracle."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_SHA = 'bf5e8c153ec6b226136cf0a9887267fedefd562a67b74ef34d47429960dbf602'
CANDIDATE_SHA = 'dff506dea523b17c7e32c4c200cda8c5138a800ea61e40ed805da1b1e3df3129'


def section(s, start, end):
    return s[s.index(start):s.index(end)]


def main():
    base = (HERE.parent/'ffn-native-fp8-full256-row-two/mlp.rs').read_text()
    candidate = (HERE/'mlp.rs').read_text()
    assert hashlib.sha256(base.encode()).hexdigest() == BASE_SHA
    assert hashlib.sha256(candidate.encode()).hexdigest() == CANDIDATE_SHA
    assert base[base.index('pub(crate) fn feedforward('):] == candidate[candidate.index('pub(crate) fn feedforward('):]
    before = section(base, 'fn project_rows_with_scale_contraction(', 'pub(crate) fn project_up_and_gate(')
    after = section(candidate, 'fn project_rows_with_scale_contraction(', 'pub(crate) fn project_up_and_gate(')
    # The declared axis extent changes32->30. All math/casts in this helper are identical;
    # only scalar storage changes from four-row transpose to 8-byte scalar slots.
    expected = before.replace('output: &mut DmTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![FfnRow]>',
                              'output: &mut DmTensor<bf16, Chip, UpGateCluster, UpGateSlices, m![FfnRow, 1 # 4]>')
    expected = expected.replace('        .transpose::<m![FfnRow / 4], m![FfnRow % 4 # 16]>()\n        .commit_trim::<m![FfnRow % 4]>()',
                                '        .commit_trim::<m![1 # 4]>()')
    assert expected == after
    prefix = candidate[:candidate.index('pub(crate) fn feedforward(')]
    assert '.memset(' not in prefix
    assert '.tile::' not in prefix
    assert 'FfnRow = 32' not in prefix and '#{!}' not in prefix
    assert 'axes![FfnTerm = 2, FfnSlice = 256, FfnRow = 30];' in prefix
    assert prefix.count('.commit_view(output.view_mut())') == 1
    assert prefix.count('FfnSlice / 2, FfnRow, FfnSlice % 2, H') == 4
    assert after.count('.cast::<bf16,') == before.count('.cast::<bf16,') == 2

    rows = set()
    poison = object()
    out_packets = 0
    for cluster in range(2):
        for group in range(128):
            sources = {}
            for side in range(2):
                physical_slice = 2*group+side
                # Four BF16 physical slots per output, only slot0 is logical data.
                slots = [poison]*(30*4)
                for row in range(30):
                    logical = 7680*cluster+60*group+2*row+side
                    assert logical not in rows
                    rows.add(logical)
                    assert (((cluster*128+group)*30+row)*2+side) == logical
                    for width in (3840,240):
                        hbm_first = (((cluster*128+group)*30+row)*2+side)*width
                        assert hbm_first == logical*width
                        # Local row occupies an exact contiguous interval; no tail row exists.
                        assert row*width+width <= 30*width
                    # 8-byte scalar commit retains the value at byte8*row.
                    slots[4*row] = logical
                    assert (4*row)*2 % 8 == 0
                assert sum(v is not poison for v in slots) == 30
                # Fetch Packet1#16 retains only the logical scalar at each time-row.
                values = [slots[4*row] for row in range(30)]
                assert all(v is not poison for v in values)
                sources[side] = values
                assert 0 <= physical_slice < 256
            # Switch puts side0,side1 next to each row, and transpose packs two rows.
            stream = [sources[side][row] for row in range(30) for side in range(2)]
            packets = [[sources[side][2*p+dr] for dr in range(2) for side in range(2)] for p in range(15)]
            assert [v for packet in packets for v in packet] == stream
            assert all(len(packet)*2 == 8 for packet in packets)
            assert stream == list(range((128*cluster+group)*60,(128*cluster+group+1)*60))
            out_packets += len(packets)
    assert rows == set(range(15360))
    print(json.dumps({
        'status':'PASS','candidate_sha256':CANDIDATE_SHA,
        'logical_rows_bijective':len(rows),'live_slices_per_cluster':256,
        'actual_rows_per_slice':30,'dummy_rows_per_slice':0,
        'weight_fp8_bytes_per_slice':30*3840,'scale_fp8_bytes_per_slice':30*240,
        'output_scalar_logical_values_per_slice':30,'output_scalar_physical_bytes_per_slice':30*8,
        'poison_padding_never_selected':True,'output_8byte_packets':out_packets,
        'native_helper_arithmetic_and_casts_unchanged':True,
        'feedforward_and_suffix_byte_identical':True,
        'limitations':'Finite coordinate/source proof only. Backend execution and numerical tolerance require NPU validation.'
    },indent=2))


if __name__ == '__main__':
    main()
