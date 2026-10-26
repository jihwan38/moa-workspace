#!/usr/bin/env python3
"""Replay frozen original/metadata-repair provenance and full resource timelines."""
import hashlib,json,subprocess
from pathlib import Path
import analyze_resource as original
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git_sources(commit,hashes):
    tracked=set(subprocess.check_output(['git','ls-tree','-r','--name-only',commit],cwd=ROOT).decode().splitlines())
    names=[p for p in hashes if p in tracked]
    for p in hashes.keys()-tracked:
        assert p=='Cargo.lock' and sha(ROOT/p)==hashes[p],p
    query=''.join(f'{commit}:{p}\n' for p in names).encode()
    raw=subprocess.check_output(['git','cat-file','--batch'],input=query,cwd=ROOT);offset=0
    for name in names:
        end=raw.index(b'\n',offset);header=raw[offset:end].split();assert header[1]==b'blob'
        size=int(header[2]);data=raw[end+1:end+1+size];offset=end+2+size
        assert hashlib.sha256(data).hexdigest()==hashes[name],(commit,name)
    assert offset==len(raw)

def records(dirname):
    out=[];verified={}
    for d in sorted((HERE/dirname).iterdir()):
        r=json.loads((d/'run.json').read_text());h=json.loads((d/'source-hashes.json').read_text())
        assert r['exit_code']==0 and r['result']=='command_succeeded' and (d/'source.diff').read_bytes()==b''
        key=r['commit']
        if key not in verified:git_sources(key,h);verified[key]=h
        else:assert verified[key]==h
        out.append({'name':r['name'],'commit':key,'verified_source_inputs':len(h),'exit_code':0,
                    'command':r['command'],'recorded_files':{p.name:sha(p) for p in sorted(d.iterdir())}})
    return out

def main():
    prior=records('runs');new=records('metadata-runs')
    assert all(x['commit'].startswith('55a55aff') for x in prior)
    assert all(x['commit'].startswith('f6ad9c83') for x in new)
    proof=json.loads((HERE/'metadata-store-source-proof.json').read_text())
    old=(ROOT/'src/ops_int4_grouped_activation_quant.rs').read_text()
    added=(ROOT/'src/ops_int4_grouped_activation_store.rs').read_text()
    assert hashlib.sha256(old.encode()).hexdigest()==proof['original_kernel_sha256']
    assert hashlib.sha256(added.encode()).hexdigest()==proof['new_kernel_sha256']
    reverted=added.replace('//! Same per960 quantizer with one real group-leader metadata store repair.','//! Per-token/per-K960 exact-rational quantization and packed I32 ownership.')
    for ax,g in [('H',4),('L',16)]:
        block=f'''    // Real gather of compact scalar rows; one1024B stripe perK960 group.
    let offset_store: DmTensor<i32, Chip, Cluster, m![{ax} / 960, 1 # {256//g}], m![P512 % 256]> =
        dense_offsets.to_dm(&mut device.tdma);
    offset_store.view().to_hbm_view(&mut device.tdma, offsets.view_mut());'''
        reverted=reverted.replace(block,'    dense_offsets.view().to_hbm_view(&mut device.tdma, offsets.view_mut());')
        block=f'''    let scale_store: DmTensor<f32, Chip, Cluster, m![{ax} / 960, 1 # {256//g}], m![P512 % 256]> =
        scale_dense.to_dm(&mut device.tdma);
    scale_store.view().to_hbm_view(&mut device.tdma, scales.view_mut());'''
        reverted=reverted.replace(block,'    scale_dense.view().to_hbm_view(&mut device.tdma, scales.view_mut());')
    assert reverted==old
    op=(ROOT/'src/bin/test_int4_grouped_activation_quant.rs').read_text()
    np=(ROOT/'src/bin/test_int4_grouped_activation_store.rs').read_text()
    assert np.replace('ops_int4_grouped_activation_store as op','ops_int4_grouped_activation_quant as op')==op
    result={k:original.schedule(p) for k,p in [('H4','input-store-schedule.json'),('L16','hidden-store-schedule.json')]}
    assert [v['complete_unfolded_schedule_cycles'] for v in result.values()]==[29069,67271]
    assert all(not v['has_dram_stack'] for v in result.values())
    images={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted((HERE/'metadata-resource-images').iterdir())};assert len(images)==6
    out={'source_commit':new[0]['commit'],'prior_source_commit':prior[0]['commit'],
         'preregistered_plan_sha256':sha(HERE/'metadata-store-plan.json'),'source_delta_proof':proof,
         'prior_runs':prior,'new_runs':new,'prior_result_sha256':sha(HERE/'resource-result.json'),
         'results':result,'resource_images':images,'source_review':{'agent':'/root/e2e_evidence','commit':new[0]['commit'],'result':'PASS','scope':'Read-only mapping/source delta/probe alias plus independent resource endpoints, exact512KiB stage allocations/nonoverlap and store-before-gather dependency; no runtime proof.'},
         'comparison':{'H4':{'prior_cycles':60292,'candidate_cycles':29069,'removed_tiny_store_each':17823,'added_DM_each':716,'replacement_HBM_each':451},
                       'L16':{'prior_cycles':74636,'candidate_cycles':67271,'removed_tiny_store_each':8457,'added_DM_each':609,'replacement_HBM_each':477}},
         'remaining_costs':{'H4_LUT_gather_cycles':7886,'L16_LUT_gather_cycles':28521,'H4_grouped_input_cycles':4783,'L16_grouped_input_cycles':15950},
         'decision':'SOURCE_HOST_RESOURCE_PASS_ONLY; fixed metadata mechanism removes expensive fragmented stores including real movement. No new sweep, standalone ELF, device execution or model activation.',
         'scope':['Both grouped graphs still exceed different-math whole-row resource context23918/45745; not an actual runtime comparison.',
                  'L16 whole delta7365 is smaller than sum of changed operation spans because of overlap.',
                  'All offsets/group scales and copied bytes still require strict actual device oracle.',
                  'Quantization-error/full-FFN/model-quality remain unknown; old failed complete FFN is preserved.']}
    print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__':main()
