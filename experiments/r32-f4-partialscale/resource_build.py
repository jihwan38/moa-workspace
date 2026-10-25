import subprocess, pathlib, json, hashlib, time
root=pathlib.Path(__file__).resolve().parents[2]
out=root/'experiments/r32-f4-partialscale'
def hashes(): return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'src').rglob('*.rs'))}
source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
for tag,fn in [('a001-candidate','ops_decode32_f4_partialscale::batch_ffn'),('a002-control','ops_decode32_ffn::batch_ffn'),('a003-pre','ops_norm_r64::rmsnorm_r64'),('a004-post','ops_norm_r64::post_feedforward_r64')]:
 cmd=['docker','exec','--workdir','/round2/r32-f4-partialscale','-e','CARGO_BUILD_JOBS=4','-e','CARGO_INCREMENTAL=0','-e','CARGO_TARGET_DIR=/round2/tail256-model/target','moa2026-round2','cargo','furiosa-opt','compile',fn,'--exact','--dump-schedule','/round2/r32-f4-partialscale/experiments/r32-f4-partialscale/'+tag+'.json']
 before=hashes(); start=time.monotonic()
 with (out/(tag+'.log')).open('w') as f: code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
 after=hashes(); result={'source':source,'command':cmd,'elapsed_seconds':time.monotonic()-start,'exit_code':code,'source_hashes':before,'all_source_hashes_match':before==after}
 (out/(tag+'-manifest.json')).write_text(json.dumps(result,indent=2)+'\n')
 print(tag,code,result['elapsed_seconds'], 'source_match',before==after,flush=True)
 if code or before!=after: break
