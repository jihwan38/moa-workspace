#!/usr/bin/env python3
"""Aggregate validated fixed-E4 numerical routes, excluding stale-OFF runs."""
from pathlib import Path
import hashlib
import json
import re
import compare_reports

root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent
pairs = [('decode512','r1334-fixed-e4-off-model512-corrected','r1335-fixed-e4-on-model512-corrected'),
         ('r32','r1336-fixed-e4-off-r32512-validated-images','r1315-fixed-e4-on-r32512'),
         ('partial896','r1337-fixed-e4-off-partial896-validated-images','r1317-fixed-e4-on-partial896')]
pairs += [(f'packed{case}', f'r{1338+2*case}-fixed-e4-off-packed{case}-validated-images',f'r{1339+2*case}-fixed-e4-on-packed{case}-validated-images') for case in range(11)]
results=[]
for mode,a,b in pairs:
    paths=[root/'runs'/name/'arena.log' for name in [a,b]]
    report=compare_reports.compare(*paths)
    gates=[]
    for path,enabled in zip(paths,[False,True]):
        text=path.read_text();assert f'enabled={str(enabled).lower()}' in text
        assert any(line.startswith('PASS FIXED_E4_') for line in text.splitlines())
        exact=list(re.finditer(r'^FIXED_E4_REPLAY .* values=(\d+) bitdiff=(\d+)$',text,re.M))
        assert len(exact)==report['row_count'] and all(m[2]=='0' for m in exact)
        cache=0
        for line in text.splitlines():
            if line.startswith(('FIXED_E4_PACKED_CACHE ','FIXED_E4_PACKED_SENTINEL ')):
                cache+=int(re.search(r'bf16_values=(\d+)',line)[1])
            elif line.startswith(('PASS FIXED_E4_MODEL ','PASS FIXED_E4_PARTIAL ')):
                cache+=int(re.search(r'replay_cache_bf16_values=(\d+)',line)[1])
        gates.append({'run':path.parent.name,'log_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'enabled':enabled,'exact_logit_values':sum(int(m[1]) for m in exact),'compared_cache_bf16_values_one_side':cache,'compared_cache_bytes_one_side':2*cache})
    assert gates[0]['exact_logit_values']==gates[1]['exact_logit_values']
    assert gates[0]['compared_cache_bf16_values_one_side']==gates[1]['compared_cache_bf16_values_one_side']
    results.append({'route':mode,'gates':gates,'comparison':report})
rows=[r for x in results for r in x['comparison']['rows']]
summary={'routes':len(results),'packed_routes':11,'logged_rows_per_precision':len(rows),
         'exact_replay_logit_values_per_precision':sum(x['gates'][0]['exact_logit_values'] for x in results),
         'compared_cache_bf16_values_per_precision_one_side':sum(x['gates'][0]['compared_cache_bf16_values_one_side'] for x in results),
         'top1_changes':sum(not r['same_top1'] for r in rows),
         'seeded_stochastic_changes':sum(not r['same_seeded_stochastic'] for r in rows),
         'mean_128_sample_relative_l2':sum(r['sample_relative_l2'] for r in rows)/len(rows),
         'max_128_sample_relative_l2':max(r['sample_relative_l2'] for r in rows),
         'max_128_sample_abs':max(r['sample_max_abs'] for r in rows),
         'scope':'Repeated/reset/interleaved diagnostic trajectories, not independent task cases. Error over128 fixed samples only; no all-element error norm. Same-program replay exactness does not establish public quality. R32 mode has no separate full-cache byte snapshot.'}
(out/'validated-route-suite.json').write_text(json.dumps({'summary':summary,'results':results},indent=2)+'\n')
print(json.dumps(summary,indent=2))
