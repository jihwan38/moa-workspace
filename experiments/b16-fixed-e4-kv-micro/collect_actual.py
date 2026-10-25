"""Collect root-submitted arena.log files; never submits work or changes guards."""
import hashlib,json,pathlib,re,statistics,sys

def fields(line):
    out={}
    for k,v in re.findall(r'(\w+)=([^ ]+)',line.strip()):
        if v in ('true','false'):v=v=='true'
        else:
            try:v=int(v)
            except ValueError:
                try:v=float(v)
                except ValueError:pass
        out[k]=v
    return out

def union(xs):
    merged=[]
    for a,b in sorted(xs):
        if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
        else:merged.append([a,b])
    return sum(b-a for a,b in merged)

def parse(path):
    r={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'codec':[],'exact':[],'drift':[],'times':[],'profiles':[]}
    current=None
    for line in path.read_text().splitlines():
        for prefix,key in [('FIXED_E4_CODEC_PASS ','codec'),('FIXED_E4_EXACT ','exact'),('FIXED_E4_DRIFT ','drift'),('FIXED_E4_TIME ','times')]:
            if line.startswith(prefix):r[key].append(fields(line))
        if line.startswith('FIXED_E4_ALL_CORRECT '):r['all_correct']=fields(line)
        elif line.startswith('PASS fixedE4 '):r['pass']=line
        elif line.startswith('PROFILE_VARIANT '):
            current=fields(line);current['spans']=[];r['profiles'].append(current)
        elif line.startswith('PROFILE_END'):current=None
        elif line.startswith('NPU_SPAN ') and current is not None:current['spans'].append(json.loads(line[9:]))
    for p in r['profiles']:
        s=p.pop('spans');p['raw_cycle_summary']={}
        for tid in sorted({x['tid'] for x in s}):
            group=[x for x in s if x['tid']==tid];summary={}
            for name in sorted({x['name'] for x in group}):
                xs=[x for x in group if x['name']==name]
                summary[name]={'count':len(xs),'sum':sum(x['dur'] for x in xs),'union':union([(x['begin_cycle'],x['end_cycle'])for x in xs])}
            p['raw_cycle_summary'][str(tid)]=summary
    return r

if __name__=='__main__':
    assert len(sys.argv)>1,'Pass collected arena.log paths'
    results=[parse(pathlib.Path(p))for p in sys.argv[1:]]
    times=[t for r in results for t in r['times']]
    means={str(b):statistics.mean(t['complete_us']for t in times if t['candidate']==b)for b in [False,True]} if times else {}
    totals={k:sum(r.get('all_correct',{}).get(k,0)for r in results)for k in ['cache_bytes','output_bytes','immutable_bytes']}
    report={'runs':results,'total_counts':totals,'complete_us':means,'all_runs_pass':all('pass'in r for r in results),
      'scope':'synthetic Q/K/V/history; supplied O only; intended BF16 roundtrip exactness does not establish original model accuracy; raw profile span sums may overlap'}
    if means:report['fraction_time_saved']=1-means['True']/means['False']
    out=pathlib.Path(__file__).parent/'actual.json';out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items()if k!='runs'},indent=2))
