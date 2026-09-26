#!/usr/bin/env python3
"""Local read-only independent validation; no package/screen imports or execution."""
import hashlib,json,re,statistics,struct,subprocess,tarfile
from pathlib import Path
R=Path('/home/jihwan38/moa-workspace')
C=R/'moa-dev/experiments/candidates/c09-output-first-npu'
A=R/'moa-dev/experiments/artifacts/c09-output-first-npu/c09-first-npu-v1'
KS=['sliding_project_qkv','sliding_attention_output','decoder_feedforward']; O=KS[1]
STAGES=['control_before','control_info','control_trace','candidate_info','candidate_trace','candidate_whole','control_after']
def d(raw): return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def f(p): return {'path':str(p.relative_to(R)),**d(p.read_bytes())}
def j(p): return json.loads(p.read_text())
def manifest(folder): return {str(p.relative_to(folder)):d(p.read_bytes()) for p in folder.rglob('*') if p.is_file() and (p.is_relative_to(folder/'src') or p.name in ('Cargo.toml','Cargo.lock','rust-toolchain.toml'))}
def abi(raw):
    assert raw[:4]==b'FOPT'
    pos=4
    def u(n):
        nonlocal pos
        assert pos+n<=len(raw)
        v=int.from_bytes(raw[pos:pos+n],'little');pos+=n;return v
    def vec(fun):
        n=u(8);assert n<100000
        return [fun() for _ in range(n)]
    def blob():
        nonlocal pos
        n=u(8);start=pos;pos+=n;assert pos<=len(raw)
        return start,raw[start:pos]
    version,chips,pes=u(4),u(1),u(1);assert version==1
    sizes=vec(lambda:vec(lambda:u(4)))
    weight=None
    if u(1): weight=(u(4),d(blob()[1]))
    stacks=vec(lambda:[u(4),u(8)]);slots=vec(lambda:[u(4),u(8)])
    inputs,outputs=vec(lambda:u(4)),vec(lambda:u(4));depth=u(4)
    def span():
        offset,name=blob();begin,end=u(2),u(2)
        return dict(name=name.decode(),offset=offset,size=len(name),begin=begin,end=end)
    spans=vec(lambda:vec(span));header=pos;tasks=[]
    for row in sizes:
        rowtasks=[]
        for n in row:
            pos=(pos+255)//256*256;assert pos+n<=len(raw)
            rowtasks.append({'offset':pos,**d(raw[pos:pos+n])});pos+=n
        tasks.append(rowtasks)
    assert (pos+255)//256*256==len(raw)
    return dict(version=version,chips=chips,pes=pes,weight=weight,stacks=stacks,slots=slots,inputs=inputs,outputs=outputs,depth=depth,header=header,spans=spans,tasks=tasks)
def inspect_probe(original_host,probe,raw,metadata_path,map_path,expected_count):
    assert original_host.count(raw)==1 and len(original_host)==len(probe)
    start=original_host.index(raw);patched=probe[start:start+len(raw)]
    old,new=abi(raw),abi(patched)
    allowed=set();assert len(old['spans'])==1
    for before,after in zip(old['spans'][0],new['spans'][0]):
        assert {k:v for k,v in before.items() if k!='name'}=={k:v for k,v in after.items() if k!='name'}
        assert after['name']==format(before['begin'],'x').rjust(before['size'],'0')
        allowed.update(range(before['offset'],before['offset']+before['size']))
    assert {k:v for k,v in old.items() if k!='spans'}=={k:v for k,v in new.items() if k!='spans'}
    changes=[i for i,(a,b) in enumerate(zip(raw,patched)) if a!=b]
    assert changes and set(changes)<=allowed
    assert probe[:start]==original_host[:start] and probe[start+len(raw):]==original_host[start+len(raw):]
    md=j(metadata_path);mapping=j(map_path)
    assert md['original_image']==d(raw) and md['labelled_image']==d(patched)
    assert md['tasks']==old['tasks']
    native=sorted([s for s in old['spans'][0] if s['name'] not in {'Task','Core','Cluster'}],key=lambda row:row['begin'])
    assert len(native)==len(mapping['operations'])==expected_count
    for parsed,row in zip(native,mapping['operations']):
        marker=row['marker'];assert marker['name']==parsed['name'] and marker['begin_marker']==parsed['begin'] and marker['end_marker']==parsed['end']
    return {'native_operations':expected_count,'name_bytes_changed':len(changes),'task_code':old['tasks'],'whole_host_other_bytes_identical':True,'all_other_ABI_metadata_identical':True,'original_image':d(raw),'labelled_image':d(patched)}
recipe=j(R/'baseline/e122.json');audit=j(R/recipe['source_build_audit']);build=j(R/audit['measured_build']);manifest_expected=j(R/recipe['compile_inputs'])
assert recipe['id']=='E122' and build['source_inputs']==manifest_expected
for path,expected in manifest_expected.items():
    if path=='Cargo.lock':
        assert d((R/recipe['tracked_lock']).read_bytes())==expected
    else:
        assert d(subprocess.check_output(['git','show',recipe['git']['commit']+':furiosa-opt-gemma4-12B/'+path],cwd=R))==expected
control=R/'moa-dev/experiments/artifacts'/Path(audit['measured_build']).parent.name/build['label']
cp=control/'package';ch=(cp/'test_runtime').read_bytes()
assert {p.name:d(p.read_bytes()) for p in cp.iterdir()}==build['package_files']
images={}
for k in KS:
    raw=(control/'kernels'/('furiosa_opt_gemma4::ops::'+k+'.bin')).read_bytes()
    assert d(raw)=={x:build['images'][k][x] for x in ('bytes','sha256')} and build['images'][k]==audit['images'][k] and ch.count(raw)==1
    images[k]=raw
report={'scope':'Independent raw-file checks by packer author; no package/screen validation functions imported, no build/Arena invocation. Separate check method, not an independent human reviewer.','control':{'recipe':f(R/'baseline/e122.json'),'commit':recipe['git']['commit'],'source_inputs':len(manifest_expected),'three_image_hashes':{k:d(v) for k,v in images.items()},'each_image_embedded_once':True,'actual_package_matches':True},'reviewed_sources':{p:f(C/p) for p in ('package.py','screen.py')},'package_ready':False,'npu_result_available':False,'blocking_findings':[],'nonblocking_findings':['Malformed DIAG_SPAN in completed-stage exception handler can halt later structured parsing; raw log and preceding stages remain and valid stays false.']}
pkgpath=C/'package-c09-first-npu-v1.json'
if pkgpath.exists():
    pkg=j(pkgpath);assert pkg['status']=='ready'
    gate_path=C/'descriptor-analysis-c09-first-npu-v1.json';gate=j(gate_path)
    assert gate['passed'] is True and f(gate_path)==pkg['descriptor_gate']
    def check_bound_records(value):
        if isinstance(value,dict):
            if {'path','bytes','sha256'}<=value.keys():
                assert f(R/value['path'])==value
            else:
                for child in value.values(): check_bound_records(child)
    check_bound_records(gate['inputdigests'])
    report['descriptor_gate_files_bound']=True
    compiled=j(C/'build-c09-first-npu-v1-repro.json');old=j(R/'moa-dev/experiments/candidates/c09-output-pipeline/build-c09-two-stage-v1-repro.json')
    assert compiled['exact_image']==old['exact_image']==d((A/'exact-kernel.bin').read_bytes())
    actualmanifest=manifest(A/'crate');assert actualmanifest==compiled['source_inputs']==pkg['source_inputs']
    assert {k:v for k,v in actualmanifest.items() if k!='src/device/sliding/projection.rs'}=={k:v for k,v in manifest_expected.items() if k!='src/device/sliding/projection.rs'}
    assert d((A/'crate/src/device/sliding/projection.rs').read_bytes())==old['source_inputs']['src/device/sliding/projection.rs']
    assert {p.name:d(p.read_bytes()) for p in (A/'package').iterdir()}==pkg['package_files']
    for name in ('fixtures.safetensors','remote_entrypoint.sh'): assert (A/'package'/name).read_bytes()==(cp/name).read_bytes()
    standard=(A/'package/test_runtime').read_bytes();unlabelled=(A/'diagnostic/unlabelled-host').read_bytes()
    candidateimages={}
    for k in KS:
        standardimage=(A/'standard-kernels'/('furiosa_opt_gemma4::ops::'+k+'.bin')).read_bytes();diagimage=(A/'diagnostic-kernels'/('furiosa_opt_gemma4::ops::'+k+'.bin')).read_bytes()
        assert standardimage==diagimage==((A/'exact-kernel.bin').read_bytes() if k==O else images[k])
        assert standard.count(standardimage)==unlabelled.count(standardimage)==1
        candidateimages[k]=d(standardimage)
    for name,digest in pkg['diagnostic_files'].items(): assert d((A/'diagnostic'/name).read_bytes())==digest
    for name,record in pkg['frozen_files'].items(): assert f(R/record['path'])==record and (R/record['path']).read_bytes()==(A/'diagnostic'/name).read_bytes()
    oldprof=R/'moa-dev/experiments/artifacts/e104-weight-path/profile-v1'
    oldbuild=j(R/'moa-dev/experiments/candidates/e104-weight-path/build-profile-v1.json')
    oldhost=(oldprof/'unlabelled-host').read_bytes();assert d(oldhost)==oldbuild['unlabelled_host']
    assert d((A/'diagnostic/test_kernels.rs').read_bytes())==oldbuild['source_inputs']['src/bin/test_kernels.rs']
    assert (oldprof/'kernels/furiosa-opt-gemma4'/('furiosa_opt_gemma4::ops::'+O+'.bin')).read_bytes()==images[O]
    report.update(package_ready=True,package_record=f(pkgpath),compile_record=f(C/'build-c09-first-npu-v1-repro.json'),candidate_images=candidateimages,source_manifest_inputs=len(actualmanifest),diagnostic_source_matches_frozen_transform=True,frozen_small_evidence_files_match=True,probe_checks={})
    for side,rawhost,raw,count in [('candidate',unlabelled,(A/'exact-kernel.bin').read_bytes(),26),('control',oldhost,images[O],22)]:
        report['probe_checks'][side]=inspect_probe(rawhost,(A/'diagnostic'/f'{side}-probe').read_bytes(),raw,A/'diagnostic'/f'{side}-o.image-profile.json',A/'diagnostic'/f'{side}-marker-map.json',count)
    assert (A/'schedule.json').read_bytes()==(C/'evidence/c09-first-npu-v1/schedule.json').read_bytes()
    for side,schedule_path in [('candidate',A/'schedule.json'),('control',R/'moa-dev/experiments/candidates/e104-qkv-output-profile/evidence/profile-v1/sliding_attention_output.json')]:
        schedule=j(schedule_path);mapping=j(A/'diagnostic'/f'{side}-marker-map.json')
        engine=[]
        for node in schedule['instructions']:
            kind=None
            if node['tpe'].startswith('Dma'): kind='DMA'
            elif node['tpe']=='Main': kind='Renegade::TuExec'
            elif node['tpe']=='Sub' and node['contexts']!=['Noop']:
                src=node['description']
                if '.to_trf()' in src: kind='Renegade::StoTrf'
                elif not src.strip(): kind='Renegade::StoTab'
                elif 'fetch_cast' in src or 'vector_' in src: kind='Renegade::TuExec'
                else: kind='Renegade::StoVrf'
            if kind: engine.append((node,kind))
        assert len(engine)==len(mapping['operations'])
        for (node,kind),row in zip(engine,mapping['operations']): assert row['schedule']==node and row['marker']['name']==kind
    report['whole_native_kind_sequence_and_frozen_schedule_nodes_match']=True
    results=list(C.glob('e126-c09-first-npu-screen-job-*-results.json'))
    assert len(results)<=1
    if results:
        result=j(results[0]);job=result['job']['id'];raw=(R/result['log']).read_bytes();assert d(raw)==result['log_digest'];text=raw.decode()
        runbuild=j(C/'e126-c09-first-npu-screen-build.json');submission=j(C/'e126-c09-first-npu-screen-submission.json')
        assert submission['job_id']==job and result['job']['status'] in {'SUCCEEDED','COMPLETED'} and result['job']['exit_code']==0
        archivepath=R/result['archive']['path'];assert f(archivepath)==result['archive']
        with tarfile.open(archivepath) as tar:
            members=[m for m in tar.getmembers() if m.isfile()];assert len({m.name for m in members})==len(members)
            contents={m.name:tar.extractfile(m).read() for m in members}
        assert {k:d(v) for k,v in contents.items() if k!='job.yaml'}==runbuild['package_files']
        for side,host in [('control',ch),('candidate',standard)]:
            assert contents[side+'_test_runtime']==host and contents[side+'_probe']==(A/'diagnostic'/f'{side}-probe').read_bytes()
        assert contents['fixtures.safetensors']==(cp/'fixtures.safetensors').read_bytes() and contents['remote_entrypoint.sh']==(cp/'remote_entrypoint.sh').read_bytes()
        sections=re.findall(r'^DIAG_STAGE_BEGIN (\w+)\n(.*?)^DIAG_STAGE_END \1$',text,re.M|re.S)
        assert [n for n,_ in sections]==STAGES and text.count('DIAG_SCREEN_COMPLETE')==1
        stages={};whole={};count=0
        for name,section in sections:
            assert re.findall(r'^DIAG_STAGE_EXIT '+name+r' (\d+)$',section,re.M)==['0']
            comparisons=re.findall(r'^\[([^]]+)\].*within tol=\s*([\d.]+)%\s*-> (PASS|FAIL)$',section,re.M)
            whole_stage=name in {'control_before','candidate_whole','control_after'}
            kernels=KS if whole_stage else [O]
            expected={f'{k}{" "+out if out else ""} run{r}' for k in kernels for out in (['q','k','v'] if k==KS[0] else ['']) for r in range(3)}
            assert len(comparisons)==len(expected) and {label.strip() for label,_,_ in comparisons}==expected
            assert all(float(tol)==100.0 and status=='PASS' for _,tol,status in comparisons)
            count+=len(comparisons)
            if whole_stage:
                assert section.count('all 3 tests passed')==1
                osection=re.search(r'^==> '+O+r'\s*\n(.*?)(?=^==>|\Z)',section,re.M|re.S)[1]
            else:
                assert section.count('DIAG_CHECK_COMPLETE selected=1 runs=3 failures=0')==1;osection=section
            matches=re.findall(r'median cycles=(\d+) \(of 3 runs: (\[[^]]+\])\)',osection);assert len(matches)==1
            reported,encoded=matches[0];cycles=json.loads(encoded);assert len(cycles)==3 and min(cycles)>0 and statistics.median(cycles)==int(reported)
            stages[name]={'passes':len(comparisons),'O_cycles':cycles,'median':int(reported)}
            if whole_stage: whole[name]=int(reported)
            else:
                side='control' if name.startswith('control') else 'candidate';mapping=j(A/'diagnostic'/f'{side}-marker-map.json')
                spans=[json.loads(s) for s in re.findall(r'^DIAG_SPAN (.+)$',section,re.M)]
                assert spans and all(s['kernel']==O for s in spans) and {s['run'] for s in spans}=={0,1,2}
                opnames=[row['marker']['label'] for row in mapping['operations']];taskname=mapping['task_label'];taskcycles=[]
                for r in range(3):
                    cluster=[]
                    for c in range(2):
                        subset=[s for s in spans if s['run']==r and s['cluster']==c];byname={s['name']:s for s in subset};assert len(subset)==len(byname)
                        task=byname[taskname];assert all(task['begin']<=s['begin']<=s['end']<=task['end'] for s in subset)
                        if name.endswith('trace'):
                            assert set(opnames)<=byname.keys();starts=[byname[n]['begin'] for n in opnames];assert starts==sorted(starts)
                        cluster.append(task['end']-task['begin'])
                    taskcycles.append(max(cluster))
                assert taskcycles==cycles;stages[name]['raw_spans_count']=len(spans);stages[name]['task_cycles_match']=True
        assert count==57
        bracket=statistics.median([whole['control_before'],whole['control_after']]);change=100*(whole['candidate_whole']/bracket-1)
        assert result['valid'] and result['correctness_passes']==count and abs(result['whole_o_comparison']['candidate_change_percent']-change)<1e-10
        report.update(npu_result_available=True,actual_upload_and_log={'job':job,'archive':f(archivepath),'log':f(R/result['log']),'all_members_exact':True,'regular_member_count':len(members),'stage_order':STAGES,'all_exit_zero':True,'unique_expected_PASS_comparisons':count,'stages':stages,'whole_o_medians':whole,'control_bracket':bracket,'whole_o_change_percent':change,'scope':'Standard-host complete O uses PROFILE=info; diagnostic info/trace cycles excluded from comparison. One allocation, original three inputs; no promotion claim.'})
Path('/tmp/moa-c09-independent-review.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'package_ready':report['package_ready'],'npu_result_available':report['npu_result_available'],'checks_passed':True}))
