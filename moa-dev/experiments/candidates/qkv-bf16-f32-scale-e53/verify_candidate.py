#!/usr/bin/env python3
"""Source scope and exhaustive row-coordinate/8-byte-commit checks; no NPU proof."""
from pathlib import Path
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT/'moa-dev/experiments/artifacts/e53-bf16-output-clean-source'
CRATE = ROOT/'moa-dev/experiments/artifacts/qkv-bf16-f32-scale-e53/crate'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
old = (HERE/'projection.rs.original').read_text()
new = (HERE/'projection.rs').read_text()
prefix, suffix = new.split('pub(crate) fn project_output(', 1)
assert suffix == old.split('pub(crate) fn project_output(', 1)[1]
assert prefix.count('let contraction: DmTensor<f32,') == 4
assert 'let contraction: DmTensor<bf16,' not in prefix
assert prefix.count('.cast::<bf16,') == 2  # Exactly one after each channel scale helper.
for function in prefix.split('    let scaled: DmTensor<bf16,')[1:]:
    scale = function.split('    let output:', 1)[0]
    assert scale.index('FpBinaryOp::MulF') < scale.index('.cast::<bf16,')
assert 'TrfTensor<f8e4m3' not in prefix and 'prepare_fp8_split' not in prefix

ops_old = (HERE/'ops.rs.original').read_text()
ops_new = (HERE/'ops.rs').read_text()
signature = r'#\[device\(chip = 1\)\]\s+pub fn \w+\([\s\S]*?\) \{'
assert re.findall(signature, ops_old) == re.findall(signature, ops_new)
begin = '#[device(chip = 1)]\npub fn sliding_project_qkv('
end = '#[device(chip = 1)]\npub fn full_project_qkv('
assert ops_old[:ops_old.index(begin)] == ops_new[:ops_new.index(begin)]
assert ops_old[ops_old.index(end):] == ops_new[ops_new.index(end):]
head_start = ('    let q: DmTensor<bf16, Chip, Cluster, m![1 # 32, Ns], m![Gs, Ds]> =\n'
              '        sliding::rmsnorm::normalize_query')
assert ops_old[ops_old.index(head_start):] == ops_new[ops_new.index(head_start):]

mapping_reports = []
for rows, cluster_rows, local_rows in [(4096, 2048, 8), (2048, 1024, 4)]:
    before, after = set(), set()
    for row in range(rows):
        cluster = row // cluster_rows
        group = row // (local_rows*8) % 32
        slc = group*8 + row%8
        local = row//8 % local_rows
        assert row == cluster*cluster_rows + group*(local_rows*8) + 8*local + slc%8
        assert (cluster, slc, local) not in before
        before.add((cluster, slc, local))
        # DPE scalar stream is transposed in groups of two F32 rows -> 8-byte commit.
        raw_time, raw_packet = divmod(local, 2)
        assert raw_time*(2*4) + raw_packet*4 == local*4
        switched_slice, switched_local = group*8+local, slc%8
        assert (cluster, switched_slice, switched_local) not in after
        after.add((cluster, switched_slice, switched_local))
        # The canonical eight-row stream has four 8-byte commits.
        out_time, out_packet = divmod(switched_local, 2)
        assert out_time*8 + out_packet*4 == switched_local*4
        if rows == 4096:
            n, gd = divmod(row, 512)
            g, d = divmod(gd, 256)
            assert (cluster, switched_slice, switched_local) == (n//4, (n%4)*64+g*32+d//8, d%8)
        else:
            n, d = divmod(row, 256)
            assert switched_slice%8 < 4
            assert (cluster, switched_slice, switched_local) == (n//4, (n%4)*64+(d//32)*8+d//8%4, d%8)
    assert len(before) == len(after) == rows
    mapping_reports.append({'rows':rows, 'coordinates_verified':rows, 'local_rows_before_switch':local_rows,
                            'f32_commit_bytes':8, 'canonical_local_rows_after_switch':8,
                            'kv_padding_never_promoted':True if rows == 2048 else None})

changes = []
for p in sorted((BASE/'src').rglob('*')):
    if p.is_file() and sha(p) != sha(CRATE/p.relative_to(BASE)):
        changes.append(str(p.relative_to(BASE)))
assert changes == ['src/device/sliding/projection.rs', 'src/ops.rs']
assert (CRATE/'src/ops.rs').read_bytes() == (HERE/'ops.rs').read_bytes()
assert (CRATE/'src/device/sliding/projection.rs').read_bytes() == (HERE/'projection.rs').read_bytes()
print(json.dumps({'source_changes':changes, 'original_src_files':43,
                  'all_public_device_signatures_unchanged':True, 'headnorm_rope_cache_tail_unchanged':True,
                  'o_projection_suffix_byte_identical_to_e53':True, 'all_other_sources_including_ffn_unchanged':True,
                  'qkv_bf16_cast_only_after_channel_scale':True, 'mapping_checks':mapping_reports,
                  'npu_tested':False}, indent=2))
