#!/usr/bin/env python3
"""CPU-only encoding benchmark; each ABBA arm has a fresh process/RSS record."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import resource
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--child', choices=['current', 'packed'])
parser.add_argument('--source-commit')
args = parser.parse_args()
if args.child:
    result = subprocess.run([str(ROOT / 'cpu_bench'), '5', args.child], check=True, text=True, capture_output=True)
    print(json.dumps(dict(variant=args.child, peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                         output=result.stdout)))
    raise SystemExit

arms = []
for i, mode in enumerate(['current', 'packed', 'packed', 'current']):
    child = subprocess.check_output([sys.executable, __file__, '--child', mode], text=True)
    arm = json.loads(child)
    (ROOT / f'cpu-benchmark-arm{i}-{mode}.log').write_text(arm['output'])
    arm['seconds'] = [float(v) for v in re.findall(r'\bseconds=([0-9.]+)', arm.pop('output'))]
    assert len(arm['seconds']) == 5
    arms.append(arm)
summary = {}
for mode in ['current', 'packed']:
    samples = [x for a in arms if a['variant'] == mode for x in a['seconds']]
    summary[mode] = dict(samples=len(samples), median_seconds=statistics.median(samples), min_seconds=min(samples),
                         max_seconds=max(samples), median_source_MB_per_s=58_982_400 / statistics.median(samples) / 1e6,
                         peak_rss_kib=max(a['peak_rss_kib'] for a in arms if a['variant'] == mode))
cpu = next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name'))
result = dict(source_commit=args.source_commit,
              helper_sha256=hashlib.sha256((ROOT / '../../src/host/f4_group960.rs').read_bytes()).hexdigest(),
              binary_sha256=hashlib.sha256((ROOT / 'cpu_bench').read_bytes()).hexdigest(),
              rustc=subprocess.check_output(['rustc', '--version'], text=True).strip(), cpu_model=cpu,
              raw_source_bytes=58_982_400, rows=15360, cols=3840, packed_bytes=29_491_200, group_scale_bytes=245_760,
              arms=arms, summary=summary,
              scope='One CPU thread, deterministic finite E4 fixture; encode and output allocations timed; fixture generation/hash/drop excluded. No NPU transfer, model startup or quality claim. Peak RSS from wait4/getrusage includes the source and returned outputs.')
(ROOT / 'cpu-benchmark.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
