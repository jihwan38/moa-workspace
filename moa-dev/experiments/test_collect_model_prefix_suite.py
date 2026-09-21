#!/usr/bin/env python3
"""Mutation checks of actual job71239 evidence; no model/NPU execution."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('collector', HERE / 'collect_model_prefix_suite.py')
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)
ROOT = HERE.parents[1]
ART = ROOT / 'moa-dev/experiments/artifacts/model-prefix-suite-pair-e55'


def main():
    original_status = json.loads((ART / 'status-71239.json').read_text())
    original_lines = (ART / 'log-71239.txt').read_text().splitlines()
    original_archive = ART / 'job-71239.tar'
    passed = []
    with tempfile.TemporaryDirectory(prefix='moa-suite-collector-') as temporary:
        temp = Path(temporary)
        status, log = temp / 'status.json', temp / 'log.txt'

        def run(label, mutate=None, bad_status=False, archive=None):
            lines = list(original_lines)
            data = copy.deepcopy(original_status)
            if bad_status:
                data['status'] = 'RUNNING'
                data['exit_code'] = None
            if mutate:
                lines = mutate(lines)
            status.write_text(json.dumps(data))
            log.write_text('\n'.join(lines) + '\n')
            try:
                result = collector.collect(status, log, archive or original_archive)
            except (ValueError, KeyError, IndexError, OverflowError):
                if label == 'actual_positive':
                    raise
                passed.append(label)
            else:
                if label != 'actual_positive':
                    raise AssertionError('incorrectly accepted ' + label)
                if result['total_logits_compared'] != 1048576 or result['accuracy_pass'] is not None:
                    raise AssertionError('positive result scope')
                passed.append(label)

        def alter(predicate, mutation):
            def change(lines):
                output = []
                for line in lines:
                    item = json.loads(line) if line.startswith('{') else None
                    if item is not None and predicate(item):
                        mutation(item)
                        line = json.dumps(item, ensure_ascii=False)
                    output.append(line)
                return output
            return change

        case = lambda x: x.get('event') == 'case_logits_compared' and x['case_index'] == 0
        before = lambda x: x.get('schema') == collector.ASSET_SCHEMA and not x['header_only']
        after = lambda x: x.get('schema') == collector.ASSET_SCHEMA and x['header_only']
        run('actual_positive')
        run('nonterminal', bad_status=True)
        run('missing_final_marker', lambda lines: [x for x in lines if x != 'MODEL_PREFIX_SUITE_PAIR_EXIT=0'])
        run('missing_final_event', lambda lines: [x for x in lines if '"event":"suite_comparison_complete"' not in x])
        run('missing_case', lambda lines: [x for x in lines if not (x.startswith('{') and case(json.loads(x)))])
        run('wrong_case_index', alter(case, lambda x: x.update(case_index=3)))
        run('wrong_token_ids', alter(case, lambda x: x['token_ids'].__setitem__(0, 1)))
        run('false_input_accuracy_pass', alter(lambda x: x.get('event') == 'suite_inputs_validated',
                                             lambda x: x.update(accuracy_pass=True)))
        run('false_reference_accuracy_pass', alter(lambda x: x.get('event') == 'reference_case_ready',
                                                 lambda x: x.update(accuracy_pass=True)))
        run('negative_top_id', alter(case, lambda x: x['comparison']['actual_top'].update(top1_id=-1)))
        run('wrong_top_margin', alter(case, lambda x: x['comparison']['actual_top'].update(top1_margin=0)))
        run('max_index_out_of_range', alter(case, lambda x: x['comparison'].update(max_abs_index=262144)))
        run('wrong_max_values', alter(case, lambda x: x['comparison'].update(actual_at_max_abs=0)))
        run('wrong_bit_equal_count', alter(case, lambda x: x['comparison'].update(bit_equal_count=262144)))
        run('partial_weight_hash', alter(before, lambda x: x['files'][0]['full_file'].update(status='incomplete', sha256=None)))
        run('changed_metadata_hash', alter(after, lambda x: x['files'][1]['full_file'].update(sha256='0' * 64)))
        run('short_metadata_hash', alter(after, lambda x: x['files'][1]['full_file'].update(bytes_hashed=1)))
        run('changed_weight_stat', alter(after, lambda x: x['files'][0]['stat_after'].update(mtime_ns=0)))
        run('wrong_contract', alter(lambda x: x.get('schema') == collector.ASSET_SCHEMA,
                                    lambda x: x.update(embedded_contract_sha256='0' * 64)))
        for label, target in [('tampered_binary', 'probe_model_e55'), ('tampered_job_config', 'job.yaml')]:
            archive = temp / (label + '.tar')
            with tarfile.open(original_archive) as source, tarfile.open(archive, 'w') as dest:
                for member in source.getmembers():
                    payload = source.extractfile(member).read()
                    if member.name == target:
                        payload = (b'X' + payload[1:]) if target != 'job.yaml' else payload.replace(b'timeout_sec: 70', b'timeout_sec: 69')
                    entry = copy.copy(member)
                    entry.size = len(payload)
                    dest.addfile(entry, io.BytesIO(payload))
            run(label, archive=archive)
    print(json.dumps({'status': 'PASS', 'positive_count': 1, 'negative_count': len(passed) - 1,
                      'tests': passed, 'source_job': 71239, 'npu_run': False}, indent=2))


if __name__ == '__main__':
    main()
