#!/usr/bin/env python3
"""Copy a frozen server and public harness with a recorded repetition count."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--candidate', required=True)
parser.add_argument('--name', required=True)
parser.add_argument('--rounds', type=int, required=True)
parser.add_argument('--log-tail-bytes', type=int, default=8000)
parser.add_argument('--warmup', action='store_true',
                    help='Complete one separate concurrency-sized performance warmup before measurement')
args = parser.parse_args()
assert 1 <= args.rounds <= 1000
assert 8000 <= args.log_tail_bytes <= 1000000
assert Path(args.candidate).name == args.candidate
assert Path(args.name).name == args.name
source = root / 'candidates' / args.candidate
manifest = json.loads((source / 'candidate.json').read_text())
binary = source / 'target/release/server'
assert hashlib.sha256(binary.read_bytes()).hexdigest() == manifest['binary_sha256']
original = (source / 'scripts/stage2/accuracy.py').read_text()
assert original.count('ROUNDS = 2') == 1
assert original.count('[-8000:]') == 1
destination = root / 'evaluations' / args.name
destination.mkdir(exist_ok=False)
for folder in ('scripts', 'public-test'):
    shutil.copytree(source / folder, destination / folder)
(destination / 'target/release').mkdir(parents=True)
shutil.copy2(binary, destination / 'target/release/server')
shutil.copy2(source / 'candidate.json', destination / 'candidate.json')
helper = destination / 'scripts/stage2/accuracy.py'
modified = original.replace('ROUNDS = 2', f'ROUNDS = {args.rounds}').replace(
    '[-8000:]', f'[-{args.log_tail_bytes}:]')
if args.warmup:
    marker = '        print(f"Sending {total} requests for {config[\'test\']}, {concurrency} at a time", flush=True)'
    assert modified.count(marker) == 1
    warmup = '''        # Actual completed requests in the same server; never relabel measured records.
        if prefix == "STAGE2_PERFORMANCE":
            warm_records = []
            warm_start = threading.Event()
            with ThreadPoolExecutor(max_workers=concurrency) as warm_pool:
                try:
                    warm_futures = [warm_pool.submit(
                        run_one, rows[i % len(rows)], bodies[i % len(rows)], i + 1,
                        config["request_timeout"], warm_start) for i in range(concurrency)]
                finally:
                    warm_start.epoch = time.monotonic()
                    warm_start.set()
                for future in as_completed(warm_futures):
                    record = future.result()
                    warm_records.append(record)
                    emit(prefix + "_WARMUP_RESULT ", record)
            warm_summary = {"total": len(warm_records),
                            "passed": sum(r["passed"] for r in warm_records)}
            if summarize is not None:
                warm_summary.update(summarize(warm_records))
            emit(prefix + "_WARMUP_SUMMARY ", warm_summary)
            if not all(r["passed"] for r in warm_records):
                raise RuntimeError("A separate warmup request failed")

'''
    modified = modified.replace(marker, warmup + marker)
compile(modified, str(helper), 'exec')
helper.write_text(modified)
(destination / 'harness.json').write_text(json.dumps({
    'base_candidate': args.candidate,
    'change': f'Public harness ROUNDS 2 to {args.rounds}; server unchanged',
    'server_log_tail_bytes': args.log_tail_bytes,
    'separate_performance_warmup': args.warmup,
    'warmup_requests': 'concurrency, completed before measurement' if args.warmup else 0,
    'server_sha256': manifest['binary_sha256'],
    'original_accuracy_helper_sha256': hashlib.sha256(original.encode()).hexdigest(),
    'accuracy_helper_sha256': hashlib.sha256(helper.read_bytes()).hexdigest(),
    'source_unchanged': True,
}, indent=2) + '\n')
print(destination)
