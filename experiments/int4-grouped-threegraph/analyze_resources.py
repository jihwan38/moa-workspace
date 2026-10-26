#!/usr/bin/env python3
"""Replay compiler resources and standalone-image identities; no build/device work."""
from pathlib import Path
import hashlib,json,collections,importlib.util
HERE=Path(__file__).resolve().parent
TREE=HERE.parents[1]
ROOT=TREE.parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def union_size(ranges):
    total=0;end=-1
    for a,b in sorted(ranges):
        if b>end:total+=b-max(a,end);end=b
    return total

def schedule(name):
    p=HERE/(name+'-schedule.json');d=read(p);t=d['timeline'];ops=d['description']['operators']
    lifetimes=[x['lifetime'] for rows in t['operators'].values() for x in rows]
    a=[{'tensor':key,**x} for key,rows in t['tensors'].items() for x in rows if x.get('allocation')]
    spills=[x for x in a if x['allocation']['buffer_type'].startswith('DramStack')]
    assert not spills,name
    kinds=collections.Counter(ops[key]['kind'] for key,rows in t['operators'].items() for x in rows)
    resources={}
    for kind in ['Sram','Trf','Spm','Vrf']:
        all_rows=[x for x in a if x['allocation']['buffer_type']==kind]
        unknown=[x for x in all_rows if x['allocation']['address'] is None]
        rows=[x for x in all_rows if x['allocation']['address'] is not None]
        points=sorted({x['lifetime'][k] for x in rows for k in ['begin','end']})
        peak=0;when=None;live=[]
        for point in points:
            active=[x for x in rows if x['lifetime']['begin']<=point<x['lifetime']['end']]
            used=union_size((x['allocation']['address'],x['allocation']['address']+x['allocation']['size']) for x in active)
            if used>peak:peak=used;when=point;live=active
        limit=t['resources'][kind]['limit'];assert peak<=limit,(name,kind,peak,limit)
        resources[kind]={'configured_limit':limit,'peak_known_address_union_bytes':peak,'unaddressed_lifetime_records':len(unknown),
            'largest_unaddressed_record_bytes':max([x['allocation']['size'] for x in unknown] or [0]),
            'at_cycle':when,'peak_live_tensors':[{'tensor':x['tensor'],'address':x['allocation']['address'],
                'size':x['allocation']['size'],'lifetime':x['lifetime'],'physical_pages':x['allocation'].get('physical_pages')} for x in live],
            'largest_single_allocation_bytes':max([x['allocation']['size'] for x in rows] or [0]),
            'scope':'Union of explicit scheduled addresses only, at half-open tensor lifetimes. Address-null views are listed separately and excluded; not a complete physical/runtime peak proof.'}
    dma=[]
    for key,rows in t['operators'].items():
        op=ops[key]
        if op['kind'] in ['dma.dto_s','dma.sto_d']:
            dma.append({'operator':key,'kind':op['kind'],'source':op['description'],
                'lifetimes':[x['lifetime'] for x in rows],'inputs':op['inputs'],'outputs':op['outputs']})
    return {'path':str(p.relative_to(TREE)),'sha256':sha(p),'timeline_begin':min(x['begin'] for x in lifetimes),
        'whole_endpoint':max(x['end'] for x in lifetimes),'operator_instances':len(lifetimes),'operator_kinds':dict(sorted(kinds.items())),
        'resources':resources,'DramStack':spills,'hbm_movement':dma}

def main():
    source=read(HERE/'source-proof.json')
    parser=ROOT/'b16-f4-mse-lm-fp8/experiments/sdk09-cache-audit/audit_registry.py'
    spec=importlib.util.spec_from_file_location('registry',parser);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    actual=ROOT/'runs/a002-int4-grouped-complete-ffn-device-build/built/target/release/test_int4_grouped_complete_ffn'
    artifact=m.read_artifact(actual);assert artifact['elf_sha256']=='3444ef561e2d39d6a15ec6340548d07e65b873d2884b62ac04f9ea2ccd66cdd4'
    refs={x['path']:x for x in artifact['records']}
    labels=['prefix','input','upgate','geglu','hidden','down','tail']
    single={label:schedule('standalone-'+label) for label in labels}
    graphs={label:schedule(label) for label in ['A','B','C']}
    images=[];runs={}
    names=['a001-int4-grouped-threegraph-hostcheck','a002-int4-threegraph-A-resource',
        'a003-int4-threegraph-B-resource','a004-int4-threegraph-C-resource']
    names +=[f'a{i:03d}-int4-threegraph-standalone-{label}' for i,label in enumerate(labels,5)]
    for name in names:
        runpath=ROOT/'runs'/name;r=read(runpath/'run.json');assert r['exit_code']==0
        hashes=read(runpath/'source-hashes.json')
        for path,h in hashes.items():assert sha(TREE/path)==h,(name,path)
        assert hashes['Cargo.lock']=='a99652661c588fe5c3a8b23278f2390d8cd2e99f2b33bcab6f52e97dbba04e2e'
        assert hashes['Cargo.toml']=='a4154d908d25a512d031dea859b1b013794f821a47deb739334332bd0e4fdf85'
        runs[name]={'commit':r['commit'],'exit_code':r['exit_code'],'started_at':r['started_at'],'finished_at':r['finished_at'],
            'source_hash_count':len(hashes),'log_sha256':sha(runpath/'output.log'),'run_sha256':sha(runpath/'run.json')}
    for i,label in enumerate(labels,5):
        files=list((HERE/f'kernel-out-a{i:03d}-int4-threegraph-standalone-{label}'/'furiosa-opt-gemma4').glob('*.bin'))
        assert len(files)==1,(label,files)
        p=files[0];r=refs[p.stem];h=sha(p)
        assert h==r['sha256'] and p.stat().st_size==r['bytes'],(label,h,r['sha256'])
        images.append({'stage':label,'path':str(p.relative_to(TREE)),'sha256':h,'bytes':p.stat().st_size,'actual323213_image_equal':True})
    for i,label in enumerate(['A','B','C'],2):
        files=list((HERE/f'kernel-out-a{i:03d}-{label}'/'furiosa-opt-gemma4').glob('*.bin'));assert len(files)==1
        p=files[0];images.append({'graph':label,'path':str(p.relative_to(TREE)),'sha256':sha(p),'bytes':p.stat().st_size})
    comparison={}
    for label,parts in {'A':['prefix','input'],'B':['upgate','geglu','hidden'],'C':['down','tail']}.items():
        old=sum(single[x]['whole_endpoint'] for x in parts);new=graphs[label]['whole_endpoint']
        comparison[label]={'standalone_parts':parts,'sum_separate_endpoints':old,'combined_endpoint':new,
            'endpoint_delta':new-old,'reduction_percent':100*(old-new)/old}
    old=sum(x['whole_endpoint'] for x in single.values());new=sum(x['whole_endpoint'] for x in graphs.values())
    return {'functional_source':'d5e70b98','runs':runs,'source_identity':source,'standalone':single,'graphs':graphs,'images':images,
        'comparisons':comparison,'sum_seven_endpoints':old,'sum_three_endpoints':new,'endpoint_reduction_percent':100*(old-new)/old,
        'retained_hbm':{'prefix_bf16_bytes':512*3840*2,'joined_bf16_bytes':2*512*15360*2,'hidden_bf16_bytes':512*15360*2,
            'down_bf16_bytes':512*3840*2,'packed_H_bytes':512*3840//2,'packed_L_bytes':512*15360//2,
            'H_group_scale_bytes':8192,'L_group_scale_bytes':32768,'H_offset_bytes':8192,'L_offset_bytes':32768,
            'residual_bytes':512*3840*2,'candidate_scratch_bytes':63979520,
            'scope':'Logical retained owners. All stage stores/reads remain; this is not DMA traffic or peak-memory accounting.'},
        'scope':['Compiler endpoints and address-liveness union are not device microseconds, launch overhead or measured complete speed.',
            'No actual device probe or standalone executable build was run in this lane.',
            'All seven standalone images equal actual323213 despite helper extraction.',
            'Current failed seven-call gate remains HOLD; >=10% versus q020 in both row orders and independent model-quality gates remain.',
            'Three-graph composition changes scheduling freedom while retaining all HBM boundaries. Any compiler prefetch overlap is separate from host wait removal.']}
if __name__=='__main__':print(json.dumps(main(),indent=2))
