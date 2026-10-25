import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).parent
SOURCE = ROOT.parents[2] / 'runs/r1245-f4-mse-supplied-weights/arena.log'
raw = SOURCE.read_bytes()
text = raw.decode()
rows = []
for line in text.splitlines():
    if not line.startswith('F4_MSE_WEIGHT '): continue
    row = {}
    for key, value in re.findall(r'(\w+)=([^\s]+)', line):
        if key == 'host_memory': continue
        try: value = float(value) if any(x in value for x in '.eE') else int(value)
        except ValueError: pass
        row[key] = value
    row['process_hwm_kib'] = int(re.search(r'VmHWM:\s*(\d+)',line).group(1))
    row['current_rss_kib'] = int(re.search(r'VmRSS:\s*(\d+)',line).group(1))
    rows.append(row)
assert {(r['layer'],r['projection']) for r in rows} == {(l,p) for l in [0,5,6,47] for p in ['up','gate','down']}
assert 'PASS F4_MSE_WEIGHT_ONLY projections=12 all_groups_SSE_not_worse=true independent_F64_values=737280' in text
def bounds(key): return dict(min=min(r[key] for r in rows),max=max(r[key] for r in rows),mean=statistics.mean(r[key] for r in rows))
result = dict(status='PASS_WEIGHT_ONLY', job=321289, duration_seconds=6.561039577,
              source_commit='8e7ce99e50c2da081c707cd053e46ce12887a8bc',
              source_log=str(SOURCE),log_sha256=hashlib.sha256(raw).hexdigest(),
              projections=rows, total_values=sum(r['values'] for r in rows),
              total_groups=sum(r['groups'] for r in rows),
              strictly_improved_groups=sum(r['strictly_improved_groups'] for r in rows),
              independent_F64_values=sum(r['independent_F64_values'] for r in rows),
              summaries={k:bounds(k) for k in ['baseline_relative_l2','fitted_relative_l2','baseline_row_dequant_l2','fitted_row_dequant_l2','D2H_ms','baseline_encode_ms','fitted_encode_and_histogram_report_ms']},
              memory_note='Process VmHWM includes initial current model loading;141140KiB live RSS after each CPU projection. Not pure encoder peak.',
              scope='Errors against current native E4 and its unchanged row dequantizer. No uploaded F4 shadows, F4 device consumer, model forward, selection or task-quality result.')
(ROOT/'actual-r1245.log').write_bytes(raw)
(ROOT/'actual-r1245.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['projections','source_log','summaries']},indent=2))
