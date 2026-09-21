#!/usr/bin/env python3
"""Check actual control archives and reject mislabeled/replaced kernel evidence."""
import copy
import io
import json
from pathlib import Path
import tarfile
import tempfile

import collect_model_kernel_control as collector

R = collector.R
C = R/'moa-dev/experiments/candidates/model-single-kernel-controls-e55'

def main():
    report_paths = sorted((R/'docs/experiments').glob('e55-model-control-*.json'))
    passed = []
    for path in report_paths:
        saved = json.loads(path.read_text())
        result = collector.collect(Path(saved['log']['path']).with_name('status-'+str(saved['job']['id'])+'.json'),
                                   Path(saved['log']['path']), Path(saved['submitted_archive']['path']),
                                   R/saved['package_manifest'])
        if result['total_logits_compared'] != 1048576 or result['accuracy_pass'] is not None:
            raise AssertionError('Invalid positive scope')
        passed.append('actual_' + saved['control'])
    if not report_paths:
        raise RuntimeError('No actual control evidence to test')
    base = json.loads((R/'docs/experiments/e55-model-control-sliding-o.json').read_text())
    status = Path(base['log']['path']).with_name('status-'+str(base['job']['id'])+'.json')
    log = Path(base['log']['path'])
    archive = Path(base['submitted_archive']['path'])
    original_manifest = json.loads((R/base['package_manifest']).read_text())
    negatives = []
    with tempfile.TemporaryDirectory(prefix='moa-controls-negative-', dir=C) as tmp:
        tmp = Path(tmp)
        manifest_path = tmp/'manifest.json'
        def rejected(label, mutate, input_archive=archive, input_log=log, expected_error=None):
            manifest = copy.deepcopy(original_manifest)
            mutate(manifest)
            manifest_path.write_text(json.dumps(manifest))
            try:
                collector.collect(status, input_log, input_archive, manifest_path)
            except (ValueError, AssertionError, KeyError, IndexError) as error:
                if expected_error is not None and str(error) != expected_error:
                    raise AssertionError(f'{label}: unexpected rejection: {error}') from error
                negatives.append(label)
            else:
                raise AssertionError('Incorrectly accepted '+label)
        rejected('wrong_control_label', lambda m:m.update(control='qkv'))
        rejected('wrong_target', lambda m:m.update(target_image='furiosa_opt_gemma4::ops::embed_token'))
        rejected('wrong_candidate_label', lambda m:m.update(candidate_label='e55'))
        rejected('wrong_reference_label', lambda m:m.update(reference_label='e55'))
        rejected('modified_package_digest', lambda m:m['files']['probe_model_control'].update(sha256='0'*64))
        rejected('invalid_reference_label', lambda m:m.update(reference_label='unknown'))
        # Replace a host with a real valid ELF and update its declared hash too.
        # Rejection must come from exact embedded-image isolation, not ELF syntax.
        for label, replacement in [('target_not_restored','e55'), ('other_three_changed','original')]:
            binary = R/'moa-dev/experiments/artifacts/model-prefix-suite-e55'/replacement/'probe_model_prefix_suite'
            payload = binary.read_bytes()
            bad_archive = tmp/(label+'.tar')
            with tarfile.open(archive) as source, tarfile.open(bad_archive,'w') as dest:
                for member in source.getmembers():
                    data = payload if member.name == 'probe_model_control' else source.extractfile(member).read()
                    entry = copy.copy(member)
                    entry.size = len(data)
                    dest.addfile(entry,io.BytesIO(data))
            digest = collector.suite.fingerprint(binary)
            original_digest = original_manifest['files']['probe_model_control']['sha256']
            original_line = original_digest + '  ./probe_model_control'
            log_text = log.read_text()
            if log_text.count(original_line) != 1:
                raise AssertionError('Expected one actual executed control digest')
            bad_log = tmp/(label+'.txt')
            bad_log.write_text(log_text.replace(original_line, digest['sha256']+'  ./probe_model_control'))
            expected_error = ('Uploaded target image differs from original' if replacement == 'e55'
                              else 'Uploaded non-target image differs from E55')
            rejected(label, lambda m:m['files']['probe_model_control'].update(sha256=digest['sha256'],bytes=digest['bytes']),
                     bad_archive, bad_log, expected_error)
    print(json.dumps({'status':'PASS','positive_count':len(passed),'positive_tests':passed,
                      'negative_count':len(negatives),'negative_tests':negatives,
                      'npu_run':False},indent=2))

if __name__ == '__main__':
    main()
