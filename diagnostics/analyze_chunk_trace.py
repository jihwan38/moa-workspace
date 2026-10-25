#!/usr/bin/env python3
"""Read optional SCHED_CHUNK records without changing or inferring serving policy."""
import argparse
import collections
import json
import re
from pathlib import Path


def tuples(value, names):
    if not value:
        return []
    return [dict(zip(names, map(int, row.split(':')), strict=True)) for row in value.split(',')]


def parse(path):
    chunks = []
    for line in path.read_text().splitlines():
        if 'SCHED_CHUNK ' not in line:
            continue
        fields = dict(re.findall(r'(\w+)=([^\s]+)', line.split('SCHED_CHUNK ', 1)[1]))
        event = {k: int(fields[k]) for k in ['seq', 'cohort', 'ready1', 'visible', 'queued',
                                            'batch', 'owners', 'tokens', 'quanta']}
        event.update({k: float(fields[k]) for k in ['work_ms', 'span_ms']})
        event['rows'] = tuples(fields.get('rows', ''), ['id', 'ring', 'position', 'count', 'remaining'])
        event['pending'] = tuples(fields.get('pending', ''), ['id', 'ring', 'position', 'remaining', 'ready_count'])
        assert len(event['rows']) == event['owners']
        assert sum(r['count'] for r in event['rows']) == event['tokens']
        assert all(r['count'] <= r['remaining'] for r in event['rows'])
        # Only full, currently eligible, page-local128 rows can use existing
        # independent grouping. Zero-ready late arrivals cannot count as work.
        eligible = [r for r in event['pending'] if r['ready_count'] == 128
                    and r['position'] // 512 == (r['position'] + 127) // 512]
        event['eligible_full128_owners'] = len(eligible)
        # Hypothetical chronological quarters before any partial page tail.
        # Repeated owners require a different explicit Model contract; this is
        # an availability count, not a claim that the current scheduler can pack it.
        event['eligible_full128_quarters'] = sum(min(4, r['remaining'] // 128,
                                                   (512 - r['position'] % 512) // 128) for r in eligible)
        chunks.append(event)
    shapes = collections.Counter(tuple(r['count'] for r in e['rows']) for e in chunks)
    singles = [e for e in chunks if e['owners'] == 1 and e['tokens'] == 128]
    return {'log': str(path.resolve()), 'chunks': chunks,
            'shapes': {','.join(map(str, k)): v for k, v in sorted(shapes.items())},
            'completed_prompt_tokens': sum(e['tokens'] for e in chunks),
            'single128_with_other_full_owner': [e['seq'] for e in singles if e['eligible_full128_owners'] > 1],
            'single128_with_four_available_quarters': [e['seq'] for e in singles if e['eligible_full128_quarters'] >= 4],
            'limits': ['Each snapshot describes scheduling time, not the membership at completion.',
                       'Discard incomplete/failed chunks; worker-local sequence IDs restart with the worker.',
                       'Mixed singleton work_ms includes all rows in that dispatch.',
                       'No request/client ID join or feasible repacking is asserted.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = parse(args.log)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'chunks'}, indent=2))


if __name__ == '__main__':
    main()
