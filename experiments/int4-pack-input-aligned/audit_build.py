#!/usr/bin/env python3
"""Verify frozen source inputs and six embedded images; no build or execution."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREE = HERE.parent.parent
ROOT = TREE.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = ROOT/'b16-f4-mse-lm-fp8/experiments/sdk09-cache-audit/audit_registry.py'
    spec = importlib.util.spec_from_file_location('registry_audit', parser)
    registry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(registry)
    old = ROOT/'runs/a005-int4-pack-staged-device-build/built/target/release/test_int4_pack_staged_store'
    new = ROOT/'runs/a004-int4-pack-input-aligned-build/built/target/release/test_int4_pack_input_aligned'
    comparison = registry.compare(registry.read_artifact(old), registry.read_artifact(new))
    assert comparison['left']['record_count'] == 5
    assert comparison['right']['record_count'] == 6
    assert comparison['equal_common_count'] == 5 and comparison['different_common_count'] == 0
    assert comparison['left_only'] == []
    assert comparison['right_only'] == [{'path': 'furiosa_opt_gemma4::ops_int4_pack_input_aligned::pack_input', 'key': []}]
    fresh = HERE/'kernel-out-a004/furiosa-opt-gemma4'
    for row in comparison['right']['records']:
        assert sha(fresh/(row['path']+'.bin')) == row['sha256']
    candidate = next(r for r in comparison['right']['records'] if 'input_aligned' in r['path'])
    assert sha(HERE/'kernel-out-a002/furiosa-opt-gemma4'/f"{candidate['path']}.bin") == candidate['sha256']
    records = []
    for name in ['a003-int4-pack-input-aligned-probecheck', 'a004-int4-pack-input-aligned-build']:
        directory = HERE/'runs'/name
        record = json.loads((directory/'run.json').read_text())
        assert record['commit'] == '58dadd8eb1ff4e30d7075bba73b0c961ef2e9be0'
        assert record['exit_code'] == 0
        hashes = json.loads((directory/'source-hashes.json').read_text())
        for relative, expected in hashes.items():
            assert sha(TREE/relative) == expected, relative
        records.append(dict(name=name, input_count=len(hashes),
                            run=record, files={p.name: sha(p) for p in directory.iterdir()}))
    build = records[-1]['run']
    artifact = build['built_artifacts']['target/release/test_int4_pack_input_aligned']
    assert build['build_inputs_unchanged']
    assert sha(new) == artifact['sha256'] == '9900b5af49813e0e3a8ad167ccea3f8d9f3ba540ce78153999a378ff4ef23715'
    assert new.stat().st_size == artifact['size'] == 3816864
    output = dict(
        source_commit=build['commit'], records=records, artifact=artifact,
        registry_parser=dict(path=str(parser), sha256=sha(parser)),
        registry_comparison=comparison, all_six_match_fresh_bins=True,
        new_candidate_matches_resource_image=True,
        independent_source_review={'agent': '/root/e2e_evidence', 'source': '58dadd8e', 'result': 'PASS',
                                   'scope': 'Narrow H call/markers/delayed20percentgate delta; full inherited oracles unchanged.'},
        actual_plan_sha256=sha(HERE/'actual-plan.json'),
        device_executed=False, preserved_H_resource_holds=True,
        dispatch=dict(owner='root', timeout_seconds=70, args=[],
                      executable='/round2/runs/a004-int4-pack-input-aligned-build/built/target/release/test_int4_pack_input_aligned',
                      optional_profile_env='FURIOSA_OPT_PROFILE=trace'),
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
