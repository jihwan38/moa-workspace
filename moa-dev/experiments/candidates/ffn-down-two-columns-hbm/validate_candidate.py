"""Verify HBM relay is the only edit versus the two-column candidate."""
from pathlib import Path
import json

here = Path(__file__).parent
old = (here / 'mlp.dm-relay.rs').read_text()
new = (here / 'mlp.rs').read_text()
addition = '''    // Use a canonical HBM vector as the source of the replicated column segments.
    let x: HbmTensor<bf16, Chip, m![L]> = x.to_hbm(&mut device.tdma);
'''
assert new.count(addition) == 1
assert new.replace(addition, '') == old
assert old.split('pub(crate) type DownCluster')[0] == new.split('pub(crate) type DownCluster')[0]
result = {
    'only_change': 'BF16 GeGLU vector copied through canonical HBM L before existing down TDMA',
    'temporary_hbm_bytes': 15360 * 2,
    'new_arithmetic_or_rounding': False,
    'safe_two_pass_decode_unchanged': True,
    'upgate_geglu_and_down_dot_source_unchanged': True,
    'inherited_numerical_risk': 'Two-column FP32 sum grouping differs from eight-column reference.',
    'hardware_correctness': 'not_tested',
}
(here / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
