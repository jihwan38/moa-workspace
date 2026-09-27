#!/usr/bin/env python3
"""Recheck E142 unchanged eight-case oracle and actual full30 mapping evidence."""
import argparse,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
import probe as p

def verify():
    p.check();b=p.verify_build();r=p.load(p.RESULTS)
    review=p.load(HERE/'independent-probe-review.json');assert review['passed']
    for rec in review['inputdigests'].values():assert p.file(p.ROOT/rec['path'])==rec
    inputs=[p.HERE/'probe.py',p.BUILD,p.RESULTS,HERE/'independent-probe-review.json',HERE/'independent_probe_review.py']
    return {'candidate':'E142','valid':True,'job':r['job']['id'],'case_count':8,'output_elements':245760,
        'exact_bitwise':True,'signed_zero_equivalent':True,'nonfinite_values':0,'actual_upload_matches':True,
        'production_source_full_file_byte_identical_prefix':True,'historical_e141_fixture_inputs_wrapper_truth_byte_identical':True,
        'production_source':b['production_source'],'results':p.file(p.RESULTS),'build':p.file(p.BUILD),
        'verifier':p.file(Path(__file__).resolve()),'inputdigests':{str(x.relative_to(p.ROOT)):p.file(x) for x in inputs},
        'performance_excluded':True,'adopted':False,'official_submission':False,
        'scope':'Actual new full30 production quantizer/up-gate plus unchanged E141 wrapper; first BF16 outputs before GeGLU. Full-file prefix, exact image/executable/upload and original CPU oracle identity rechecked.',
        'limits':['The new F32 partial/scaling consumer passes these eight finite cases; general BF16/model equivalence is not proven.',
                  'This standalone diagnostic does not measure complete FFN performance or execute GeGLU/down/RMS. Whole performance and full-FFN stress are separate gates.']}

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--check',action='store_true');args=a.parse_args();v=verify();dest=HERE/'verification-summary.json'
    if args.check:assert p.load(dest)==v
    else:p.write(dest,v)
    print('PASS E142 full30 actual-source oracle8cases/245760exact and independent partial/copy mapping')
