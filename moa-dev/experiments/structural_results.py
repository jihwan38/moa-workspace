#!/usr/bin/env python3
"""Extract a structural screen; only complete passing stages have performance aggregates."""
import argparse, json, statistics
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

def report(batch, stage):
    assert batch["stages"]["control_before"]["valid"]
    assert batch["stages"]["control_after"]["valid"]
    run = {k: batch[k] for k in ("job", "log", "log_sha256", "package_files")}
    run.update(batch["stages"][stage], batch_stage=stage,
               executed_host=stage + "_test_runtime",
               benchmark_files={n: stage + "_" + n for n in ("test_runtime", "fixtures.safetensors", "remote_entrypoint.sh")})
    value = dict(schema_version=1, valid=run["valid"], runs=[run], independent_arena_jobs=1,
                 method="One upload-verified Arena allocation; unchanged benchmark hosts with E104 before and after. A failed candidate has no performance aggregate.")
    if run["valid"]:
        value["aggregate"] = {k: {"median_cycles": v["median_cycles"], "job_medians": [v["median_cycles"]]}
                              for k, v in run["kernels"].items()}
    return value


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate');parser.add_argument('label');parser.add_argument('experiment');args=parser.parse_args()
    HERE=ROOT/'moa-dev/experiments/candidates'/args.candidate
    batch=json.loads((HERE/f'{args.label}-results.json').read_text())
    build=json.loads((HERE/f'{args.label}-build.json').read_text())
    before=report(batch,'control_before');after=report(batch,'control_after')
    candidate=report(batch,'candidate_0')
    reference=f'docs/experiments/{args.label}-e104-before.json'
    candidate['reference']=reference
    candidate['control_after_report']=f'docs/experiments/{args.label}-e104-after.json'
    stage=next(s for s in build['stages'] if s['name']=='candidate_0')
    candidate['candidate_build']=str((HERE/f"build-{stage['build']}-repro.json").relative_to(ROOT))
    target=json.loads((HERE/f"build-{stage['build']}-repro.json").read_text())['target_kernel']
    candidate['target_kernel']=target
    if candidate['valid']:
        c=candidate['aggregate'][target]['median_cycles']
        b=before['aggregate'][target]['median_cycles']
        a=after['aggregate'][target]['median_cycles']
        candidate['screen_change_percent']={'vs_before':100*(c/b-1),'vs_after':100*(c/a-1),'vs_bracketing_median':100*(c/statistics.median([b,a])-1)}
    for path,value in [(ROOT/reference,before),(ROOT/candidate['control_after_report'],after),(ROOT/f'docs/experiments/{args.experiment}.json',candidate)]:
        with path.open('x') as out:json.dump(value,out,indent=2);out.write('\n')
    print(json.dumps({'valid':candidate['valid'],'candidate':args.experiment,'change':candidate.get('screen_change_percent')},indent=2))
