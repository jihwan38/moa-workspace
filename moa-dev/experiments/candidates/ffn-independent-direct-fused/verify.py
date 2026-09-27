#!/usr/bin/env python3
"""Revalidate E135 frozen runtime-failure evidence; PASS means faithful evidence, not a passing kernel."""
import argparse, json, re, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(HERE))
from package import read,write,file_record,checked_package
from screen import checked_screen,parse_stages,archive_files
import probe
LABEL='e135-direct-six-row-v1';SCREEN='e135-direct-six-row-screen'

def analyze():
    subprocess.run([sys.executable,str(HERE/'verify_graph.py'),LABEL,'--check'],check=True)
    pp,pkg,_,_=checked_package(LABEL)
    bp,batch,_=checked_screen(SCREEN)
    path=HERE/f'{SCREEN}-job-110314-results.json';r=read(path)
    assert r['job']['id']==110314 and r['job']['status']=='FAILED' and r['job']['exit_code']==1
    assert r['valid'] is False and r['actual_upload_matches'] and r['screen_complete']
    log_path=ROOT/r['log'];raw=log_path.read_bytes()
    from package import digest
    assert digest(raw)==r['log_digest'] and file_record(ROOT/r['archive']['path'])==r['archive']
    files=archive_files(ROOT/r['archive']['path'])
    assert files==r['package_files'] and {k:v for k,v in files.items() if k!='job.yaml'}==batch['package_files']
    stages,names=parse_stages(raw.decode());assert stages==r['stages'] and names==list(batch['stages'])
    for side in ('control_before','control_after'):
        assert stages[side]['valid'] and stages[side]['correctness_passes']==15
    candidate=stages['candidate_whole'];assert not candidate['valid'] and candidate['observed_exit_codes']==['101']
    assert len(candidate['observed_comparisons'])==12 and all(v['status']=='PASS' for v in candidate['observed_comparisons'])
    section=raw.decode().split('DIAG_STAGE_BEGIN candidate_whole\n')[1].split('DIAG_STAGE_END candidate_whole')[0]
    assert 'the launch ran past its time limit' in section and 'ended a launch with 4:' in section
    assert not re.search(r'^\[decoder_feedforward',section,re.M)
    assert 'median cycles' not in section.split('==> decoder_feedforward')[1]
    pb=probe.verify_build();pr=read(probe.RESULTS)
    assert pr['valid'] is False and pr['job']['id']==110316 and pr['job']['status']=='FAILED' and pr['job']['exit_code']==101
    assert pr['build']==file_record(probe.BUILD) and pr['actual_upload_matches'] and pr['observed_projection_checks']==[]
    for k in ('raw_log','archive'):assert file_record(ROOT/pr[k]['path'])==pr[k]
    pf=archive_files(ROOT/pr['archive']['path']);assert pf==pr['package_files']
    assert {k:v for k,v in pf.items() if k!='job.yaml'}==pb['package_files']
    pl=(ROOT/pr['raw_log']['path']).read_text()
    assert 'ended a launch with 4:' in pl and 'the launch ran past its time limit' in pl
    assert probe.case_lines(pl)==probe.case_lines((probe.EVID/'oracle-only.log').read_text())[:3]
    assert not re.search(r'^E135_PROBE ',pl,re.M) and 'E135_PROBE_COMPLETE' not in pl
    gate=read(HERE/f'graph-verification-{LABEL}.json')
    weight_ranges=[x['weight_byte_range'] for x in gate['upgate_streams']]
    assert sorted(weight_ranges)==sorted([[i*11520,(i+1)*11520] for i in range(5)]*2)
    return {'evidence_verified':True,'kernel_correctness_passed':False,'candidate_performance_valid':False,
            'classification':'runtime_liveness_unresolved','build_label':LABEL,'source_base_commit':pb['source_base_commit'],
            'source':pb['production_source'],'graph_gate':file_record(HERE/f'graph-verification-{LABEL}.json'),
            'package':file_record(pp),'screen_build':file_record(bp),'screen_results':file_record(path),
            'screen_log':file_record(log_path),'screen_archive':r['archive'],'screen_upload_verified':True,
            'whole':{'job':110314,'candidate_first_ffn_launch_timeout_code':4,'candidate_ffn_comparisons':0,'candidate_ffn_cycles':None,
                     'candidate_unchanged_qkv_o_passes':12,'control_passes':30,
                     'control_before_ffn_cycles':stages['control_before']['kernels']['decoder_feedforward']['median_cycles'],
                     'control_after_ffn_cycles':stages['control_after']['kernels']['decoder_feedforward']['median_cycles']},
            'oracle':{'job':110316,'build':file_record(probe.BUILD),'results':file_record(probe.RESULTS),'log':pr['raw_log'],
                      'archive':pr['archive'],'upload_verified':True,'first_case':0,'launch_timeout_code':4,
                      'arithmetic_outputs_checked':0,'planned_cases':8,'planned_outputs':245760,
                      'cpu_oracle_and_quantizer_only_passed':True,'actual_production_prefix_verified':True},
            'odd_packet_count_check':{'packed_per_slice_bytes':57600,'packed_256B_packets':225,'six_row_tile_bytes':11520,
                                     'six_row_256B_packets':45,'joint_or_materialized_per_slice_bytes':115200,
                                     'joint_or_materialized_256B_packets':450,'all_bytes_mod128':0,
                                     'conclusion':'Odd count alone does not change128B bank phase or demonstrate consecutive-bank starvation; no supported padding repair follows.'},
            'sdk_profile_excerpt':file_record(HERE/'evidence/sdk-timeout-profile-excerpt.txt'),
            'sdk_profile_origin':file_record(HERE/'evidence/sdk-timeout-profile-origin.json'),
            'verifier':file_record(Path(__file__)),'adopted':False,'official_submission':False,
            'scope':'Verification PASS means the preserved failure and provenance are reproducible. Graph/CPU gates do not mean any E135 FFN/projection hardware arithmetic output passed.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args();v=analyze();path=HERE/'verification.json'
    if a.check:assert read(path)==v
    else:write(path,v)
    print('E135 failed-attempt provenance verified; whole/projection liveness unresolved, no candidate performance result')
