#!/usr/bin/env python3
"""Validate the Python-only 40-checkpoint continuation and link prior observations.

Core validation is shared with the four-prompt collector via immutable config.
The prior-prefix comparison is an observation, never an accuracy verdict.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import collect_model_continuation as core

CONFIG = core.CollectorConfig(
    schema='moa-model-continuation-python40-v1',
    prompts=('Complete the Python expression: len([1, 2, 3])',),
    max_new=40,
    max_total=128,
    marker_prefix='MODEL_CONTINUATION_PYTHON40',
    package='moa-dev/experiments/candidates/model-continuation-python40-pair-e55/package-manifest.json',
    scope_intro='One Python prompt with 1..40 reference-defined checkpoints',
)
PRIOR_REPORT = core.ROOT / 'docs/experiments/e55-model-continuation.json'
PRIOR_SHA256 = '459a1e99ad72fc7dde7c54020ba3e56ef9b55420d5b2701bde11e9702f72f6f7'
PRIOR_BYTES = 387822
LINK_FIELDS = (
    'step', 'position', 'consumed_token_ids', 'element_count', 'nonfinite_count',
    'reference_selected_token', 'reference_selected_is_eos', 'reference_legal_choice',
    'candidate_selected_token', 'candidate_selected_is_eos', 'candidate_legal_choice',
    'selected_token_match', 'reference_choice_candidate_rank',
    'first_legal_greedy_disagreement_step', 'replay_policy', 'comparison',
)


def differing_fields(before, after, path=''):
    """Report exact parsed-JSON value differences without implying array identity."""
    if isinstance(before, dict) and isinstance(after, dict):
        differences = []
        for key in sorted(before.keys() | after.keys()):
            child = f'{path}.{key}' if path else key
            if key not in before or key not in after:
                differences.append(child)
            else:
                differences.extend(differing_fields(before[key], after[key], child))
        return differences
    if isinstance(before, list) and isinstance(after, list):
        differences = [path + '.length'] if len(before) != len(after) else []
        for index, (old, new) in enumerate(zip(before, after)):
            differences.extend(differing_fields(old, new, f'{path}[{index}]'))
        return differences
    return [] if before == after else [path]


def asset_content_identity(identities):
    # mtime/inode need not agree between jobs; content hashes and lengths must.
    return {name: {'sha256': item['sha256'], 'size_bytes': item['size_bytes']}
            for name, item in identities.items()}


def link_prior_observations(result, prior_path=PRIOR_REPORT):
    fingerprint = core.common.fingerprint(prior_path)
    core.require(fingerprint['sha256'] == PRIOR_SHA256 and fingerprint['bytes'] == PRIOR_BYTES,
                 'prior continuation report differs from frozen 71329 evidence')
    prior = core.parse_json(prior_path.read_text())
    core.require(prior['job']['id'] == 71329, 'wrong prior continuation job')
    previous = core.validate_events(prior['continuation_events'])
    old = previous['case_summaries'][3]
    current = result['case_summaries'][0]
    core.require(old['prompt'] == current['prompt'] == CONFIG.prompts[0], 'prior/new Python prompt mismatch')
    core.require(len(old['checkpoint_comparisons']) == 8, 'prior Python prefix is incomplete')
    assets_before = asset_content_identity(prior['asset_identities_before'])
    assets_after = asset_content_identity(result['asset_identities_before'])
    asset_differences = differing_fields(assets_before, assets_after)
    count = min(8, len(current['checkpoint_comparisons']))
    comparisons = []
    for step in range(count):
        before = {field: old['checkpoint_comparisons'][step][field] for field in LINK_FIELDS}
        after = {field: current['checkpoint_comparisons'][step][field] for field in LINK_FIELDS}
        differences = differing_fields(before, after)
        comparisons.append({'step': step, 'reported_fields_equal': not differences, 'different_fields': differences})
    all_eight_equal = count == 8 and all(item['reported_fields_equal'] for item in comparisons)
    if asset_differences:
        status = 'DIFFERENT_ASSETS'
    elif count < 8:
        status = 'FEWER_THAN_EIGHT_NEW_CHECKPOINTS'
    elif all_eight_equal:
        status = 'SAME_ASSETS_FIRST_EIGHT_REPORTED_FIELDS_MATCH'
    else:
        status = 'FIRST_EIGHT_REPORTED_FIELDS_DIFFER'
    return {
        'status': status, 'prior_report': fingerprint, 'prior_job_id': 71329,
        'prior_case_index': 3, 'new_case_index': 0, 'requested_checkpoint_count': 8,
        'available_checkpoint_count': count, 'same_asset_content_hashes_and_lengths': not asset_differences,
        'asset_different_fields': asset_differences, 'prior_asset_content_identity': assets_before,
        'new_asset_content_identity': assets_after, 'checked_checkpoint_fields': list(LINK_FIELDS),
        'checkpoints': comparisons, 'all_first_eight_reported_fields_equal': all_eight_equal,
        'comparison_is_observational': True, 'accuracy_acceptance_threshold': None, 'accuracy_pass': None,
        'scope': 'Exact equality of parsed host-reported histories, legal-choice summaries and full-vocabulary aggregate statistics. Different assets or observations do not invalidate the new run structurally. Case index, checkpoint filename, timings and reference-file SHA are excluded; JSON case coordinates differ. No downloaded-logit or whole-array equivalence claim.',
    }


def collect(status_path, log_path, archive_path, package_path=None):
    result = core.collect(status_path, log_path, archive_path, package_path, CONFIG)
    result['collector_core'] = result['collector']
    result['collector'] = core.common.fingerprint(Path(__file__).resolve())
    result['collector_config'] = asdict(CONFIG)
    result['prior_python_first_eight_observations'] = link_prior_observations(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('status', 'log', 'archive'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.status, args.log, args.archive), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
