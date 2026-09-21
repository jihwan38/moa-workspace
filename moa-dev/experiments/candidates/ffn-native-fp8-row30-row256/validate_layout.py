"""Read-only finite coordinate/source check for the row256 diagnostic."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
CANDIDATE_SHA='b82b141e33f1afc14bd261b3505c4fff11d8431a7f43e473c9d1a8aeb2b085f9'
BASE_SHA='dff506dea523b17c7e32c4c200cda8c5138a800ea61e40ed805da1b1e3df3129'


def section(s,start,end):
    return s[s.index(start):s.index(end)]


def main():
    base=(HERE.parent/'ffn-native-fp8-row30/mlp.rs').read_text()
    candidate=(HERE/'mlp.rs').read_text()
    verified=(HERE.parent/'ffn-native-fp8-tail-zero/mlp.rs').read_text()
    assert hashlib.sha256(base.encode()).hexdigest()==BASE_SHA
    assert hashlib.sha256(candidate.encode()).hexdigest()==CANDIDATE_SHA
    assert section(base,'fn project_rows_with_scale_contraction(', 'pub(crate) fn project_up_and_gate(')==section(candidate,'fn project_rows_with_scale_contraction(', 'pub(crate) fn project_up_and_gate(')
    assert base[base.index('pub(crate) fn feedforward('):]==candidate[candidate.index('pub(crate) fn feedforward('):]
    old_project=section(base,'pub(crate) fn project_up_and_gate(', '    // L = 7680*cluster')
    new_project=section(candidate,'pub(crate) fn project_up_and_gate(', '    // L = 7680*cluster')
    assert old_project.replace('m![Dummy2, FfnSlice / 2, FfnRow, FfnSlice % 2, H', 'm![Dummy2, FfnRow, FfnSlice, H')==new_project
    old_gather=section(verified,'fn compact_full256_rows(', 'pub(crate) fn feedforward(')
    new_gather=section(candidate,'fn compact_full256_rows(', 'pub(crate) fn feedforward(')
    old_gather=old_gather.replace('UpGateSlices, m![FfnRow]>', 'UpGateSlices, m![FfnRow, 1 # 4]>')
    old_gather=old_gather.replace('.begin(x.view().tile::<m![FfnRow], 30, m![FfnRow = 30 # 32]>(0))', '.begin(x.view())')
    old_gather=old_gather.replace('FfnRow = 30','FfnRow')
    assert old_gather==new_gather
    prefix=candidate[:candidate.index('pub(crate) fn feedforward(')]
    assert '.memset(' not in prefix and '.tile::' not in prefix
    assert 'FfnRow = 32' not in prefix and 'FfnSlice / 2' not in prefix
    rows=set();poison=object();packets=0
    for cluster in range(2):
        storage={s:[poison]*120 for s in range(256)}
        for row in range(30):
            for slc in range(256):
                logical=7680*cluster+256*row+slc
                assert logical not in rows
                rows.add(logical)
                assert ((cluster*30+row)*256+slc)==logical
                for width in [3840,240]:
                    assert ((cluster*30+row)*256+slc)*width==logical*width
                    assert (row+1)*width<=30*width
                storage[slc][row*4]=logical
        gathered=[]
        for row in range(30):
            for group in range(64):
                packet=[storage[4*group+k][4*row] for k in range(4)]
                assert all(v is not poison for v in packet)
                assert len(packet)*2==8
                gathered.extend(packet);packets+=1
        assert gathered==list(range(7680*cluster,7680*(cluster+1)))
        for slc in range(256):
            assert sum(v is not poison for v in storage[slc])==30
    assert rows==set(range(15360))
    print(json.dumps({'status':'PASS','candidate_sha256':CANDIDATE_SHA,
                      'logical_rows_bijective':len(rows),'live_slices_per_cluster':256,
                      'actual_rows_per_slice':30,'zero_or_dummy_rows':0,
                      'scalar_storage_bytes_per_slice':240,'packed_output_8byte_packets':packets,
                      'poison_padding_never_selected':True,'native_helper_byte_identical_to_row30':True,
                      'feedforward_suffix_byte_identical':True,'row256_gather_matches_e39_after_input_storage_adjustment':True,
                      'limitation':'Coordinate/source audit only, not a hardware oracle.'},indent=2))


if __name__=='__main__':main()
