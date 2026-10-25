"""Read-only fixed-E4 trace attribution. No compiler or NPU invocation."""
import hashlib
import json
from pathlib import Path
import statistics


def overlap(a, b):
    return max(0, min(a['end_cycle'], b['end_cycle']) -
               max(a['begin_cycle'], b['begin_cycle']))


root = Path(__file__).resolve().parents[3]
here = Path(__file__).resolve().parent
resource = root / 'b16-fixed-e4-kv/experiments/b16-fixed-e4-kv/candidate-schedule.json'
schedule = json.loads(resource.read_text())
operators = schedule['description']['operators']
selected = {}
for key in ['@40.44', '@40.59', '@40.88']:
    op = operators[key]
    selected[key] = {'kind': op['kind'], 'source': op['description'],
                     'timeline': schedule['timeline']['operators'][key]}

profiles = []
logs = []
for name in ['r1289-arena.log', 'r1290-arena.log']:
    path = root / 'b16-fixed-e4-kv-micro/experiments/b16-fixed-e4-kv-micro' / name
    logs.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    current = None
    groups = []
    for line in path.read_text().splitlines():
        if line.startswith('PROFILE_VARIANT '):
            current = {'label': line, 'spans': []}
            groups.append(current)
        elif line.startswith('PROFILE_END'):
            current = None
        elif line.startswith('NPU_SPAN ') and current is not None:
            current['spans'].append(json.loads(line[9:]))
    for group in groups:
        if 'candidate=true' not in group['label']:
            continue
        for tid in [0, 1]:
            spans = sorted([s for s in group['spans'] if s['tid'] == tid],
                           key=lambda s: s['begin_cycle'])
            tu = [s for s in spans if s['name'] == 'Renegade::TuExec']
            tasks = [s for s in spans if s['name'] == 'Task']
            # These signature checks are specific to this frozen graph, not a
            # generic DMA/TU ordinal mapper. Static source has four equal codec
            # passes, two large K->TRF loads, then QK immediately before V transpose.
            codec = [s for s in tu if 11589 <= s['dur'] <= 11591]
            assert len(codec) == 4 and len(tu) == 46 and len(tasks) == 1
            task = tasks[0]
            rows = []
            for k, v in zip(codec[0::2], codec[1::2]):
                kt = [s for s in tu if s['begin_cycle'] >= k['end_cycle'] and
                      s['end_cycle'] <= v['begin_cycle'] and 16000 < s['dur'] < 18000]
                vg = [s for s in spans if s['name'] == 'DMA' and
                      s['begin_cycle'] < k['end_cycle'] < s['end_cycle'] and
                      25000 < s['dur'] < 35000]
                assert len(kt) == 1 and len(vg) == 1
                kt, vg = kt[0], vg[0]
                kg = [s for s in spans if s['name'] == 'DMA' and
                      s['end_cycle'] <= k['begin_cycle'] and 25000 < s['dur'] < 35000][-1]
                table = [s for s in spans if s['name'] == 'Renegade::StoTab' and
                         s['end_cycle'] < k['begin_cycle']][-1]
                table_dma = [s for s in spans if s['name'] == 'DMA' and
                             s['end_cycle'] <= table['begin_cycle']][-1]
                qk = [s for s in tu if s['begin_cycle'] >= v['end_cycle']][0]
                transpose = [s for s in tu if s['begin_cycle'] >= qk['end_cycle']][0]
                assert qk['dur'] < 2000 and 17000 < transpose['dur'] < 18000
                assert kt['end_cycle'] < vg['end_cycle']
                rows.append({'K_decode': k, 'K_TRF': kt, 'K_gather': kg,
                             'V_gather': vg, 'V_decode': v, 'QK': qk,
                             'K_table': table, 'K_table_DMA': table_dma,
                             'K_TRF_overlap_V_gather': overlap(kt, vg),
                             'K_table_overlap_K_gather': overlap(table, kg),
                             'K_TRF_ready_before_V_gather_end': vg['end_cycle'] - kt['end_cycle']})
            decode = sum(g['K_decode']['dur'] for g in rows)
            qk = sum(g['QK']['dur'] for g in rows)
            table_dma = sum(g['K_table_DMA']['dur'] for g in rows)
            table_exposed = sum(g['K_table']['dur'] - g['K_table_overlap_K_gather'] for g in rows)
            profiles.append({'log': name, 'label': group['label'], 'cluster': tid,
                             'Task_cycles': task['dur'], 'groups': rows,
                             'K_decode_cycles': decode, 'QK_cycles': qk,
                             'K_table_DMA_cycles': table_dma,
                             'K_table_outside_K_gather_cycles': table_exposed,
                             'decode_plus_entire_QK_over_task': (decode + qk) / task['dur'],
                             'plus_table_DMA_over_task': (decode + qk + table_dma) / task['dur'],
                             'generous_current_schedule_credit_over_task':
                                 (decode + qk + table_dma + table_exposed) / task['dur']})

report = {'decision': 'HOLD without resource build: no defensible >=10% complete gain from this bounded change',
          'scope': 'Same B16 geometry, fixed E4 K cache; new query quantization is approximate. Actual span accounting is not an Amdahl proof or wall-time prediction.',
          'attribution': 'Source/resource topology plus repeated unique span signatures; actual spans contain no source/profile IDs. No cross-graph ordinal-only matching.',
          'logs': logs, 'resource_path': str(resource),
          'resource_sha256': hashlib.sha256(resource.read_bytes()).hexdigest(),
          'resource_operations': selected, 'profiles': profiles,
          'means': {key: statistics.mean(p[key] for p in profiles) for key in
                    ['Task_cycles', 'K_decode_cycles', 'QK_cycles', 'K_table_DMA_cycles',
                     'K_table_outside_K_gather_cycles', 'decode_plus_entire_QK_over_task',
                     'plus_table_DMA_over_task', 'generous_current_schedule_credit_over_task']}}
print(json.dumps(report, indent=2))
