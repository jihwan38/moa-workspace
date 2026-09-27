#!/usr/bin/env python3
"""Freeze and exact-compile one axis-name-only v7b API diagnostic; never host-build."""
import argparse
import difflib
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ORIGIN = ROOT / 'moa-dev/experiments/candidates/fp4-native-boundary'
LABEL = 'interleave-name-v1'
REMOTE = '/tmp/moa-e131-interleave-name/' + LABEL

def digest(data):
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

def record(path):
    return {'path': str(path.relative_to(ROOT)), **digest(path.read_bytes())}

def write_json(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')

def inputs():
    return {str(p.relative_to(HERE/'source')): digest(p.read_bytes())
            for p in sorted((HERE/'source').rglob('*')) if p.is_file()}

def prepare():
    old = json.loads((ORIGIN/'build-boundary-v7b.json').read_text())
    frozen = ORIGIN/'evidence/boundary-v7b'
    source = HERE/'source'
    source.mkdir(exist_ok=False)
    changed = []
    for name, expected in old['source_inputs'].items():
        data = (frozen/name).read_bytes()
        assert digest(data) == expected, name
        if name == 'src/probe.rs':
            before = data.decode()
            after, occurrences = re.subn(r'\bProjection\b', 'interleave', before)
            assert occurrences > 0 and 'Projection' not in after
            data = after.encode()
            changed.append(name)
            (HERE/'source.diff').write_text(''.join(difflib.unified_diff(
                before.splitlines(keepends=True), after.splitlines(keepends=True),
                fromfile='frozen-v7b/src/probe.rs', tofile='axis-name-v1/src/probe.rs')))
        target = source/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    assert changed == ['src/probe.rs']
    recipe = {
        'id': 'E131-RECOVERY-API-INTERLEAVE', 'label': LABEL,
        'scope': 'Axis-name-only API feasibility diagnostic, not a new performance family',
        'actual_start_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        'actual_branch': subprocess.check_output(['git','branch','--show-current'], cwd=ROOT, text=True).strip(),
        'source_base_commit': old['base_commit'],
        'source_origin': record(ORIGIN/'build-boundary-v7b.json'),
        'prior_recipe': record(ORIGIN/'v7b/recipe.json'),
        'prior_failure': record(frozen/'build.log'),
        'performance_control': 'baseline/e128.json (context only; no performance comparison authorized)',
        'question': 'Does naming the public I axis interleave remove the declared Projection_1 versus pipeline interleave_1 exact-lowering mismatch?',
        'change': 'All complete Projection tokens in frozen src/probe.rs become interleave; declaration and mappings/index operations change together. No alias.',
        'replacement_occurrences': occurrences,
        'unchanged': 'All other source/build inputs, dimensions, arithmetic, compact HBM loads, scale/reduction order and outputs remain byte-identical.',
        'full_graph': 'BF16 preparation -> E5 3-term TRF/input scale -> two compact FP4 DM inputs -> begin_interleaved lookup/native -> F32 block partial -> separate block-scale/reductions -> F32 outputs',
        'first_stop': 'Preserve exact compile outcome. On success inspect descriptors and order; no host build or NPU. A different compiler error is not the original mismatch.',
        'decisions': {'adoption': 'not applicable; diagnostic only', 'further_tuning': 'no padding/shape sweep', 'research_priority': 'API feasibility for avoiding E131 packed/scale assembly, pending full-path design and NPU evidence'},
        'expected_source_inputs': inputs(), 'source_diff': record(HERE/'source.diff'),
        'sdk_evidence': 'Installed std0.8.1 context.rs:96-113 preserves generic I in Symbol<I>; mapping-macro lib.rs:105-106 stringifies axis name. NPU primitive implementation is not in public crates; literal-name cause remains a hypothesis.'
    }
    write_json(HERE/'recipe.json', recipe)
    (HERE/'README.md').write_text(
        '# E131 interleave axis-name diagnostic\n\n'
        'Frozen `fp4-native-boundary/v7b` source is reproduced from its recorded nine-file build input. '
        'Only complete `Projection` tokens in `src/probe.rs` become `interleave`; `source.diff` records every edit. '
        'The source recipe remains historical v7b, while the adopted performance reference is E128.\n\n'
        'The first gate is exact compilation of `probe::fused`. Success establishes only target lowering of this fixed probe. '
        'Two-source bounds/order still require emitted-descriptor review, and accuracy/performance require later authorized NPU tests. '
        'No host build, NPU run, SDK modification, prepacking, numerical change or layout sweep is performed.\n\n'
        'Reproduction: `python3 build.py prepare` creates a new immutable snapshot; `python3 build.py compile` compiles it once. '
        'Existing evidence is never overwritten.\n')
    print('Prepared', LABEL, 'replacement tokens', occurrences, flush=True)

def compile_probe():
    recipe = json.loads((HERE/'recipe.json').read_text())
    manifest = inputs()
    assert manifest == recipe['expected_source_inputs']
    evidence = HERE/'evidence'/LABEL
    evidence.mkdir(parents=True, exist_ok=False)
    subprocess.run(['docker','exec','moa2026-dev','mkdir','-p',REMOTE], check=True)
    subprocess.run(['docker','cp',str(HERE/'source')+'/.','moa2026-dev:'+REMOTE], check=True)
    archive = subprocess.check_output(['docker','exec','--workdir',REMOTE,'moa2026-dev','tar','-cf','-',*manifest])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        assert {m.name:digest(tar.extractfile(m).read()) for m in tar if m.isfile()} == manifest
    command = ['docker','exec','--workdir',REMOTE,'-e','CARGO_HOME=/home/moa/.cargo',
               '-e','CARGO_NET_OFFLINE=true','-e','CARGO_TARGET_DIR=/tmp/moa-c01-upgate/target',
               '-e','FURIOSA_OPT_OUT_DIR='+REMOTE+'/kernels','moa2026-dev',
               'cargo','furiosa-opt','compile','probe::fused','--exact',
               '--dump-graph','graph.json','--dump-schedule','schedule.json',
               '--dump-ir','kernel.ir','--dump-visa','kernel.visa']
    result = {'label': LABEL, 'recipe':record(HERE/'recipe.json'), 'builder':record(Path(__file__)),
              'source_inputs':manifest,'container_inputs_verified':True,'command':command,
              'scope':'exact compile only; no host build or NPU', 'dumps':{}}
    try:
        with (evidence/'compile.log').open('x') as log:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        result['exact_compile_exit_code'] = code
        if code == 0:
            for name in ('graph.json','schedule.json','kernel.ir','kernel.visa'):
                subprocess.run(['docker','cp','moa2026-dev:'+REMOTE+'/'+name,str(evidence/name)], check=True)
                result['dumps'][name] = record(evidence/name)
            subprocess.run(['docker','cp','moa2026-dev:'+REMOTE+'/kernels/furiosa-opt-gemma4/furiosa_opt_gemma4::probe::fused.bin',str(evidence/'exact-kernel.bin')], check=True)
            result['image'] = record(evidence/'exact-kernel.bin')
            result['compiler_makespan'] = max(x['lifetime']['end'] for x in json.loads((evidence/'schedule.json').read_text())['instructions'])
    finally:
        result['compile_log'] = record(evidence/'compile.log')
        write_json(HERE/('build-'+LABEL+'.json'), result)
    print(json.dumps({k:v for k,v in result.items() if k in ('label','exact_compile_exit_code','image','compiler_makespan')}), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','compile'])
    args = parser.parse_args()
    prepare() if args.action == 'prepare' else compile_probe()
