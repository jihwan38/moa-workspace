#!/usr/bin/env python3
"""Validate frozen C09 performance with actual E122 bracketing standard hosts.

No compilation. Two clean allocations reuse the exact E126 three-file stage
payloads. Historical E126 is pooled only after normalizing the stage payloads;
its seven-process outer batch is recorded separately from these clean batches.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tarfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
VALIDATION=HERE/'validation'
ARTIFACTS=ROOT/'moa-dev/experiments/artifacts'/HERE.name
BUILD_LABEL='c09-first-npu-v1'
CLEAN_LABELS=('e127-c09-clean-a','e127-c09-clean-b')
STAGES=('control_before','candidate_whole','control_after')
KERNELS=('sliding_project_qkv','sliding_attention_output','decoder_feedforward')
O=KERNELS[1]
sys.path.insert(0,str(HERE))
from package import checked_control,digest,file_record
sys.path.insert(0,str(ROOT/'moa-dev'))
from measure import cli,parse_log


def read(path): return json.loads(path.read_text())
def write(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')
def record(name): return VALIDATION/f'performance-{name}.json'
def clean_label(label): assert label in CLEAN_LABELS,label

def checked_packages():
    _,control_build,control,control_record=checked_control()
    package_path=HERE/f'package-{BUILD_LABEL}.json';package=read(package_path)
    assert package['status']=='ready' and package['control']==control_record
    folder=ARTIFACTS/BUILD_LABEL/'package'
    assert {p.name:digest(p.read_bytes()) for p in folder.iterdir()}==package['package_files']
    host=(folder/'test_runtime').read_bytes()
    for kernel in KERNELS:
        raw=(ARTIFACTS/BUILD_LABEL/'standard-kernels'/f'furiosa_opt_gemma4::ops::{kernel}.bin').read_bytes()
        assert host.count(raw)==1 and digest(raw)=={k:package['images'][kernel][k] for k in ('bytes','sha256')}
        if kernel!=O: assert package['images'][kernel]==control_build['images'][kernel]
    original=read(HERE.parent/'c09-output-pipeline/build-c09-two-stage-v1-repro.json')
    assert {k:package['images'][O][k] for k in ('bytes','sha256')}==original['exact_image']
    for name in ('fixtures.safetensors','remote_entrypoint.sh'):
        assert (folder/name).read_bytes()==(control/name).read_bytes()
    return control,folder,package_path,package,control_record


def prepare():
    control,candidate,package_path,package,control_record=checked_packages()
    path=record('build');assert not path.exists()
    out=ARTIFACTS/'e127-clean-package';out.mkdir(parents=True,exist_ok=False)
    for side,source in [('control',control),('candidate',candidate)]:
        (out/f'{side}_test_runtime').write_bytes((source/'test_runtime').read_bytes())
    for name in ('fixtures.safetensors','remote_entrypoint.sh'):
        (out/name).write_bytes((control/name).read_bytes())
    runner=['#!/bin/sh','set -u','cd "$(dirname "$0")" || exit 1','any_failed=0']
    for stage in STAGES:
        side='candidate' if stage=='candidate_whole' else 'control'
        runner += [f"printf 'DIAG_STAGE_BEGIN {stage}\\n'",
                   f'mkdir {stage} || exit 1',
                   f'cp {side}_test_runtime {stage}/test_runtime || exit 1',
                   f'cp fixtures.safetensors remote_entrypoint.sh {stage}/ || exit 1',
                   f'if FURIOSA_OPT_PROFILE=info sh {stage}/remote_entrypoint.sh; then code=0; else code=$?; any_failed=1; fi',
                   f"printf 'DIAG_STAGE_EXIT {stage} %s\\n' \"$code\"",
                   f"printf 'DIAG_STAGE_END {stage}\\n'"]
    runner += ["printf 'DIAG_SCREEN_COMPLETE\\n'",'exit "$any_failed"']
    (out/'run.sh').write_text('\n'.join(runner)+'\n');(out/'run.sh').chmod(0o755)
    write(path,{'schema_version':1,'build_label':BUILD_LABEL,'candidate_package':file_record(package_path),
                'control':control_record,'stages':list(STAGES),'timeout_seconds':70,
                'expected_comparisons':45,'builder':file_record(Path(__file__).resolve()),
                'package_dir':str(out.relative_to(ROOT)),
                'package_files':{p.name:digest(p.read_bytes()) for p in sorted(out.iterdir())},
                'normalized_stage_files':{'candidate':package['package_files'],
                    'control':{p.name:digest(p.read_bytes()) for p in control.iterdir()}},
                'method':'Three standard-host processes: actual E122 before -> frozen C09 -> actual E122 after. Original three inputs and PROFILE=info. No diagnostic host or compile.'})
    print('prepared one immutable clean package for two independent jobs')


def submit(label):
    clean_label(label);saved=read(record('build'));checked_packages()
    path=record(label+'-submission');assert not path.exists()
    out=ROOT/saved['package_dir'];assert {p.name:digest(p.read_bytes()) for p in out.iterdir()}==saved['package_files']
    remote='/tmp/moa-c09-validation/'+label
    subprocess.run(['docker','exec','moa2026-dev','mkdir','-p',remote],check=True)
    subprocess.run(['docker','cp',str(out)+'/.','moa2026-dev:'+remote],check=True)
    for name,expected in saved['package_files'].items():
        actual=subprocess.check_output(['docker','exec','moa2026-dev','sha256sum',remote+'/'+name],text=True).split()[0]
        assert actual==expected['sha256'],name
    write(path,{'attempted_at_utc':datetime.now(timezone.utc).isoformat(),'build_record':file_record(record('build'))})
    response=cli('submit',*(remote+'/'+name for name in saved['package_files']),
                 '--entrypoint','run.sh','--timeout',saved['timeout_seconds'],'--name','beef-'+label)
    value=read(path);value['response']=response
    jobs=re.findall(r'submitted job (\d+)',response)
    if len(jobs)==1:value['job_id']=int(jobs[0])
    path.write_text(json.dumps(value,indent=2)+'\n');assert len(jobs)==1,response
    print(response,end='')


def status(label):
    clean_label(label);print(cli('status',read(record(label+'-submission'))['job_id']),end='')


def parse_stages(raw,expected):
    sections=re.findall(r'^DIAG_STAGE_BEGIN (\w+)\n(.*?)^DIAG_STAGE_END \1$',raw,re.M|re.S)
    names=[name for name,_ in sections];result={}
    for name in expected:
        parts=[part for observed,part in sections if observed==name]
        try:
            assert len(parts)==1,'missing/incomplete/duplicated stage'
            exits=re.findall(r'^DIAG_STAGE_EXIT '+name+r' (\d+)$',parts[0],re.M);assert exits==['0'],exits
            kernels=parse_log(parts[0]);assert sum(k['output_comparisons_passed'] for k in kernels.values())==15
            result[name]={'valid':True,'exit_code':0,'correctness_passes':15,'kernels':kernels}
        except (AssertionError,ValueError) as error:
            result[name]={'valid':False,'error':repr(error),'stage_started':f'DIAG_STAGE_BEGIN {name}\n' in raw}
    return result,names


def collect(label):
    clean_label(label);saved=read(record('build'));job=read(record(label+'-submission'))['job_id']
    path=record(label+'-results');assert not path.exists()
    job_status=json.loads(cli('status',job));assert job_status['status'] in {'SUCCEEDED','COMPLETED','FAILED','CANCELLED','CANCELED'},job_status
    raw=cli('logs',job);log=VALIDATION/f'performance-arena-{job}.log'
    with log.open('x') as stream:stream.write(raw)
    result={'job':job_status,'log':str(log.relative_to(ROOT)),'log_digest':digest(raw.encode()),
            'build_record':file_record(record('build')),'valid':False,'stages':{},'actual_upload_matches':False,
            'method':saved['method']}
    try:
        remote='/tmp/moa-c09-validation/'+label+f'/job-{job}.tar'
        cli('download',job,'--out',remote)
        archive=ARTIFACTS/f'job-{job}.tar'
        subprocess.run(['docker','cp','moa2026-dev:'+remote,str(archive)],check=True)
        with tarfile.open(archive) as tar:
            members=[m for m in tar if m.isfile()];assert len(members)==len({m.name for m in members})
            files={m.name:digest(tar.extractfile(m).read()) for m in members}
        assert {k:v for k,v in files.items() if k!='job.yaml'}==saved['package_files']
        result.update(actual_upload_matches=True,package_files=files,archive=file_record(archive))
        stages,names=parse_stages(raw,STAGES);result['stages']=stages
        result['stage_sequence_matches']=names==list(STAGES)
        result['screen_complete']=raw.count('DIAG_SCREEN_COMPLETE')==1
        result['valid']=(job_status['status'] in {'SUCCEEDED','COMPLETED'} and job_status['exit_code']==0
                         and result['stage_sequence_matches'] and result['screen_complete'] and all(s['valid'] for s in stages.values()))
        if result['valid']:
            result['correctness_passes']=sum(s['correctness_passes'] for s in stages.values());assert result['correctness_passes']==45
            before=stages['control_before']['kernels'][O]['median_cycles'];after=stages['control_after']['kernels'][O]['median_cycles']
            candidate=stages['candidate_whole']['kernels'][O]['median_cycles'];bracket=statistics.median([before,after])
            result['whole_o_comparison']={'control_before':before,'candidate':candidate,'control_after':after,
                'bracketing_median':bracket,'candidate_change_percent':100*(candidate/bracket-1)}
    except Exception as error:result.update(valid=False,error=repr(error))
    write(path,result);print(json.dumps({k:v for k,v in result.items() if k not in {'stages','package_files'}},indent=2))
    assert result['valid'],'Saved failed result; stop and inspect before further allocations'


def verified_batch(path):
    batch=read(path);assert batch['valid'] and batch['actual_upload_matches']
    raw=(ROOT/batch['log']).read_bytes();assert digest(raw)==batch['log_digest']
    assert file_record(ROOT/batch['archive']['path'])==batch['archive']
    with tarfile.open(ROOT/batch['archive']['path']) as tar:
        members=[m for m in tar if m.isfile()];assert len(members)==len({m.name for m in members})
        files={m.name:digest(tar.extractfile(m).read()) for m in members}
    assert files==batch['package_files']
    reparsed,_=parse_stages(raw.decode(),STAGES)
    for name in STAGES:
        assert reparsed[name]['valid'] and reparsed[name]['kernels']==batch['stages'][name]['kernels']
    return batch


def report(check=False):
    control,candidate,package_path,package,control_record=checked_packages()
    paths=[HERE/'e126-c09-first-npu-screen-job-105820-results.json']+[record(label+'-results') for label in CLEAN_LABELS]
    batches=[verified_batch(path) for path in paths];assert len({b['job']['id'] for b in batches})==3
    normalized={'candidate':package['package_files'],'control':{p.name:digest(p.read_bytes()) for p in control.iterdir()}}
    candidates=[];controls=[];paired=[]
    for path,batch in zip(paths,batches):
        files=batch['package_files'];common={'job':batch['job'],'log':batch['log'],'log_sha256':batch['log_digest']['sha256'],
            'valid':True,'source_batch_report':str(path.relative_to(ROOT)),'source_batch_report_digest':digest(path.read_bytes()),
            'uploaded_archive_path':batch['archive']['path'],'uploaded_archive_digest':{k:batch['archive'][k] for k in ('bytes','sha256')},
            'uploaded_archive_files':files,'uploaded_archive_verified':True,
            'package_files_scope':'Canonical three-file benchmark stage, not the full uploaded batch archive.'}
        for side in ('candidate','control'):
            mapping={'test_runtime':side+'_test_runtime','fixtures.safetensors':'fixtures.safetensors','remote_entrypoint.sh':'remote_entrypoint.sh'}
            assert {key:files[value] for key,value in mapping.items()}==normalized[side]
        cand=batch['stages']['candidate_whole'];before=batch['stages']['control_before'];after=batch['stages']['control_after']
        candidates.append({**common,'package_files':normalized['candidate'],'kernels':cand['kernels'],'comparisons_passed':15,
            'measurement_kind':'actual_benchmark_stage','original_batch_stage':'candidate_whole',
            'original_benchmark_files':{'test_runtime':'candidate_test_runtime','fixtures.safetensors':'fixtures.safetensors','remote_entrypoint.sh':'remote_entrypoint.sh'}})
        kernels={k:{'correctness':'passed','median_cycles':statistics.median([before['kernels'][k]['median_cycles'],after['kernels'][k]['median_cycles']]),
            'bracketing_process_medians':[before['kernels'][k]['median_cycles'],after['kernels'][k]['median_cycles']],
            'output_comparisons_passed':before['kernels'][k]['output_comparisons_passed']+after['kernels'][k]['output_comparisons_passed']} for k in KERNELS}
        controls.append({**common,'package_files':normalized['control'],'kernels':kernels,'comparisons_passed':30,
            'measurement_kind':'derived_bracketing_control_reference','control_before':before,'control_after':after,
            'original_benchmark_files':{name:{'test_runtime':'control_test_runtime','fixtures.safetensors':'fixtures.safetensors','remote_entrypoint.sh':'remote_entrypoint.sh'} for name in ('control_before','control_after')},
            'note':'Derived median of two measured process medians, not an extra process or a three-input sample.'})
        cy=cand['kernels'][O]['median_cycles'];ref=kernels[O]['median_cycles']
        paired.append({'job_id':batch['job']['id'],'outer_batch':'historical seven-stage diagnostic screen' if path==paths[0] else 'prospective three-stage standard-host clean comparison',
            'candidate_o_cycles':cy,'control_before_o_cycles':before['kernels'][O]['median_cycles'],'control_after_o_cycles':after['kernels'][O]['median_cycles'],
            'bracketing_control_o_cycles':ref,'ratio':cy/ref,'delta_percent':100*(cy/ref-1)})
    method='Three distinct Arena allocations of identical standard benchmark-stage host/fixture/runner: one historical E126 seven-stage outer batch and two prospective clean E122 -> C09 -> E122 batches. All uploaded archives and raw logs verified. Original three inputs reused; diagnostic timings excluded.'
    def base(label,runs):
        assert all(r['package_files']==runs[0]['package_files'] for r in runs)
        return {'schema_version':1,'label':label,'valid':True,'runs':runs,'independent_arena_jobs':3,'method':method,
            'aggregate':{k:{'median_cycles':statistics.median([r['kernels'][k]['median_cycles'] for r in runs]),
                'job_medians':[r['kernels'][k]['median_cycles'] for r in runs],
                'min_cycles':min(r['kernels'][k]['median_cycles'] for r in runs),
                'max_cycles':max(r['kernels'][k]['median_cycles'] for r in runs)} for k in KERNELS}}
    candidate_report=base('E127 frozen C09 adoption performance validation',candidates)
    control_report=base('E122 paired reference for E127',controls)
    candidate_report.update(reference='docs/experiments/e127-c09-e122-paired-control.json',
        candidate_build=str((HERE/f'build-{BUILD_LABEL}-repro.json').relative_to(ROOT)),candidate_package=str(package_path.relative_to(ROOT)),
        target_kernel=O,paired_observations=paired,median_paired_delta_percent=statistics.median(p['delta_percent'] for p in paired),
        aggregate_median_delta_percent=100*(candidate_report['aggregate'][O]['median_cycles']/control_report['aggregate'][O]['median_cycles']-1),
        all_paired_o_improved=all(p['delta_percent']<0 for p in paired),
        unchanged_kernel_timing_note='QKV/FFN images unchanged; observed timing differences are not optimizations.',
        adoption_budget_exception='User requested bounded adoption validation of the already implemented small gain. The 5% exploration-budget cutoff is not an adoption prohibition. No parameter tuning.',
        adopted=False,official_submission=False,accuracy_scope='Original three inputs only in this performance report. Expanded correctness is separate and required before adoption.')
    summary={'id':'E127','valid':True,'independent_allocations':3,'new_clean_allocations':2,'normalized_stage_payloads':normalized,
        'paired_observations':paired,'median_paired_delta_percent':candidate_report['median_paired_delta_percent'],
        'aggregate_median_delta_percent':candidate_report['aggregate_median_delta_percent'],
        'all_paired_o_improved':candidate_report['all_paired_o_improved'],
        'candidate_report':'docs/experiments/e127-c09-adoption-performance.json','control_report':'docs/experiments/e127-c09-e122-paired-control.json',
        'builder':file_record(Path(__file__).resolve()),'method':method,'adopted':False,
        'status':'performance_reproduced_pending_expanded_correctness' if candidate_report['all_paired_o_improved'] else 'performance_not_consistent_requires_budget_decision'}
    outputs={ROOT/summary['candidate_report']:candidate_report,ROOT/summary['control_report']:control_report,record('summary'):summary}
    for path,value in outputs.items():
        if check:assert read(path)==value,path
        else:write(path,value)
    md=['# E127 동결 C09 성능 채택 검증','',method,'','| Job | 전체 O candidate | E122 before / after | paired 변화율 |','|---|---:|---|---:|']
    for row in paired:md.append(f"| {row['job_id']} | {row['candidate_o_cycles']:,} | {row['control_before_o_cycles']:,} / {row['control_after_o_cycles']:,} | {row['delta_percent']:+.6f}% |")
    md += ['',f"Paired 변화율 중앙값 **{summary['median_paired_delta_percent']:+.6f}%**; job 중앙값 aggregate 비율 **{summary['aggregate_median_delta_percent']:+.6f}%**.",
        '', '각 stage의 실제 host/fixture/runner bytes가 E126과 동일함을 업로드 tar에서 검증했다. 한 job의3개 입력 또는 before/after를 독립 allocation으로 세지 않았다. QKV/FFN timing 흔들림은 최적화로 세지 않는다.',
        '', '5%는 탐색 예산 기준이며 채택 검증 금지 기준이 아니다. 이번 검증은 기존 구현의 재현성 확인으로 한정했다. 원본3입력 성능 반복만으로 채택을 선언하지 않으며 확장 정확도 결과와 종합 검토가 별도로 필요하다.',
        '',f"검증 상태: `{summary['status']}`. 이 기록 작성 시점 adopted=false, 공식 제출 없음."]
    text='\n'.join(md)+'\n';mdpath=VALIDATION/'performance-summary.md'
    if check:assert mdpath.read_text()==text
    else:
        with mdpath.open('x') as stream:stream.write(text)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','submit','status','collect','report'])
    parser.add_argument('label',nargs='?');parser.add_argument('--check',action='store_true');args=parser.parse_args()
    VALIDATION.mkdir(exist_ok=True)
    if args.action in {'submit','status','collect'}:
        assert args.label and not args.check
        {'submit':submit,'status':status,'collect':collect}[args.action](args.label)
    elif args.action=='prepare':assert args.label is None and not args.check;prepare()
    else:assert args.label is None;report(args.check)
