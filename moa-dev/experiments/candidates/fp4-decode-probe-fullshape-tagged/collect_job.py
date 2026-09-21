#!/usr/bin/env python3
"""Read Arena70658 only. No submit/rerun and no source/binary mutation."""
import ast,datetime,hashlib,json,re,subprocess,sys,tarfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
ART=ROOT/'moa-dev/experiments/artifacts/fp4-decode-probe-fullshape-tagged';MOA=ROOT/'moa-dev/moa';JOB=70658
OUT=ROOT/'docs/experiments/fp4-decode-probe-fullshape-tagged.json';LOG=ROOT/f'moa-dev/logs/fp4-decode-probe-fullshape-tagged-job-{JOB}.log'
EXPECTED_CHECKED=2*240*32*3840

def cli(*args):return subprocess.check_output([str(MOA),'furiosa-arena',*map(str,args)],text=True,stderr=subprocess.STDOUT)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert not OUT.exists(), 'refuse to overwrite a result'
    previous=None;deadline=time.monotonic()+1800
    while True:
        status=json.loads(cli('status',JOB));state=status['status'].upper();current=(state,status.get('queue_ahead'))
        if previous!=current:print(f'job{JOB}: {state}, queue_ahead={current[1]}',flush=True);previous=current
        if state in {'SUCCEEDED','COMPLETED','FAILED','CANCELED','CANCELLED'} or status.get('exit_code') is not None:break
        if time.monotonic()>deadline:raise TimeoutError('job still pending; no new job created')
        time.sleep(5)
    log=cli('logs',JOB);LOG.write_text(log)
    cp=f'/tmp/moa-fp4-decode-probe-fullshape-tagged/job-{JOB}.tar';hp=ART/f'job-{JOB}.tar'
    cli('download',JOB,'--out',cp,'--create-dirs')
    subprocess.run(['docker','cp',f'moa2026-dev:{cp}',str(hp)],check=True)
    package={}
    with tarfile.open(hp) as archive:
        for m in archive.getmembers():
            if not m.isfile():continue
            assert m.name not in package
            b=archive.extractfile(m).read();package[m.name]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
    assert set(package)=={'test_runtime','remote_entrypoint.sh','job.yaml'}
    for name,expected in json.loads((HERE/'package-manifest.json').read_text())['files'].items():assert package[name]==expected,f'uploaded {name} changed'
    results={}
    pattern=r'^DECODE_RESULT mode=(legacy|fused) case=(\d+) checked=(\d+) mismatches=(\d+) nonfinite=(\d+) signed_zero_only=(\d+) status=(PASS|FAIL)$'
    for mode,case,n,bad,nf,sz,st in re.findall(pattern,log,re.M):
        key=(mode,int(case));assert key not in results and int(case)==1 and int(n)==EXPECTED_CHECKED
        assert (int(bad)==0)==(st=='PASS') and int(nf)<=int(bad) and int(sz)<=int(bad)
        results[key]={'mode':mode,'case':int(case),'checked':int(n),'mismatches':int(bad),'nonfinite':int(nf),'signed_zero_only':int(sz),'status':st}
    for mode,case,by_cluster,by_code,by_nibble,by_lane in re.findall(r'^DECODE_COUNTS mode=(legacy|fused) case=(\d+) by_cluster=(\[[^\]]*\]) by_code=(\[[^\]]*\]) by_nibble=(\[[^\]]*\]) by_mod64=(\[[^\]]*\])$',log,re.M):
        r=results[(mode,int(case))]
        for name,text,size in [('by_cluster',by_cluster,2),('by_code',by_code,16),('by_nibble',by_nibble,2),('by_mod64',by_lane,64)]:
            values=ast.literal_eval(text);assert len(values)==size and sum(values)==r['mismatches'];r[name]=values
    for mode,case,phase,values in re.findall(r'^DECODE_SAMPLES mode=(legacy|fused) case=(\d+) nibble=(\d+) by_code=(\[[^\]]*\])$',log,re.M):
        values=ast.literal_eval(values);assert len(values)==16
        results[(mode,int(case))].setdefault('sample_bits_by_nibble_and_code',{})[phase]=values
    samples=[]
    for line in log.splitlines():
        if not line.startswith('DECODE_MISMATCH '):continue
        parsed=dict(re.findall(r'(\w+)=([^ ]+)',line))
        samples.append({k:(v if k=='mode' else int(v,0)) for k,v in parsed.items()})
    for r in results.values():
        r['coordinate_samples']=[s for s in samples if s['mode']==r['mode'] and s['case']==r['case']]
        assert len(r['coordinate_samples'])==min(32,r['mismatches'])
    launches=[{'mode':m,'case':int(c),'stage':s} for m,c,s in re.findall(r'^DECODE_PROBE mode=(legacy|fused) case=(\d+) (LAUNCH_BEGIN|LAUNCH_RETURNED)$',log,re.M)]
    roundtrips=[int(c) for c in re.findall(r'^INPUT_ROUNDTRIP case=(\d+) elements=58982400 status=PASS$',log,re.M)]
    expected={(mode,case) for mode in ['legacy','fused'] for case in [1]}
    complete=set(results)==expected
    final=re.findall(r'^FP4_DECODE_FULLSHAPE_TAGGED variants=2 cases=1 clusters=2 live_slices=240 rows=32 columns=3840 total_mismatches=(\d+) status=(PASS|FAIL)$',log,re.M)
    all_exact=complete and all(r['mismatches']==0 for r in results.values())
    if complete:
        assert len(final)==1 and int(final[0][0])==sum(r['mismatches'] for r in results.values())
        assert (final[0][1]=='PASS')==all_exact
        assert roundtrips==[1]
        assert status['exit_code']==(0 if all_exact else 1)
    scope='Standalone32x3840 decode,2active clusters and240live/256physical slices each, independently tagged cluster/slice/row input. This tests full E37 up/gate decoder shape but excludes DPE/scales/quantizer/fullFFN allocation lifetimes. No score/performance claim.'
    coverage_pattern=r'^CPU_CASE case=1 name=coordinate_tagged_ordered_pairs elements=58982400 packed_bytes=29491200 codes=16 nibble_phases=2 ordered_pairs=256 all_codes_at_each_mod64_position=true unique_row_sequences=15360 unique_cluster_slice_sequences=480$'
    coverage_matches=re.findall(coverage_pattern,log,re.M)
    coverage={'runtime_assertions_passed':len(coverage_matches)==1,'fp4_elements':58982400,'packed_bytes':29491200,'fp4_codes':16,'nibble_phases':2,'ordered_nibble_pairs':256,'all_pairs_in_first512_elements_of_every_row':True,'all_codes_at_every_mod64_position':True,'unique_row_sequences':15360,'unique_cluster_slice_sequences':480,'proof':'Executed make_case recovers all15360 row identities from k=0,512,1024,1536 and asserts unique; coverage line is printed only after these assertions.'}
    if complete:assert coverage['runtime_assertions_passed']
    record={'recorded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'official_score':False,'performance_measurement':False,
            'job':status,'log':str(LOG.relative_to(ROOT)),'log_sha256':sha(LOG),'package_files':package,'package_tar_sha256':sha(hp),
            'uploaded_package_matches_frozen_build':True,'build_evidence':'moa-dev/experiments/candidates/fp4-decode-probe-fullshape-tagged/evidence.json',
            'registry_file':'moa-dev/experiments/candidates/fp4-decode-probe-fullshape-tagged/registry-summary.json',
            'probe_complete':complete,'probe_passed':all_exact,'input_coverage':coverage,'total_bf16_values_compared':sum(r['checked'] for r in results.values()),'total_mismatches':sum(r['mismatches'] for r in results.values()),'results':list(results.values()),'input_roundtrip_cases':roundtrips,'launch_markers':launches,
            'legacy_all_cases_exact':all(('legacy',c) in results and results[('legacy',c)]['mismatches']==0 for c in [1]),
            'fused_all_cases_exact':all(('fused',c) in results and results[('fused',c)]['mismatches']==0 for c in [1]),
            'available_results_format_and_histograms_valid':True,'new_npu_jobs_created_by_collector':0,'scope':scope}
    OUT.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'output':str(OUT.relative_to(ROOT)),'probe_complete':complete,'probe_passed':all_exact,'legacy_exact':record['legacy_all_cases_exact'],'fused_exact':record['fused_all_cases_exact'],'mismatches_by_case':{f'{m}/{c}':r['mismatches'] for (m,c),r in results.items()}},indent=2),flush=True)
    return 0

if __name__=='__main__':sys.exit(main())
