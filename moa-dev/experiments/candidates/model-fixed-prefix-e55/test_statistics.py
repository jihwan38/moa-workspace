#!/usr/bin/env python3
"""Test the exact probe statistics functions without importing SDK/device code."""
from pathlib import Path
import subprocess
if not __debug__:
    raise RuntimeError('Statistics checks require assertions; do not use Python optimization mode.')
C=Path(__file__).resolve().parent
R=C.parents[3]
A=R/'moa-dev/experiments/artifacts/model-fixed-prefix-e55'
s=(C/'probe_model_logits.rs').read_text()
functions=s[s.index('fn top_summary('):s.index('#[tokio::main]')]
tests=r'''
#[test]
fn known_errors_and_raw_top_ids() {
    let v=compare_logits(&[1.0,2.0,3.0],&[1.0,3.0,1.0]);
    assert_eq!(v["elements_compared"],3); assert_eq!(v["max_abs"],2.0);
    assert_eq!(v["mean_abs"],1.0); assert_eq!(v["bit_equal_count"],1);
    assert!((v["rmse"].as_f64().unwrap()-(5.0f64/3.0).sqrt()).abs()<1e-14);
    assert_eq!(v["reference_top"]["top1_id"],2); assert_eq!(v["actual_top"]["top1_id"],1);
    assert_eq!(v["diagnostic_absolute_error_counts"][0]["count"],1);
    assert_eq!(v["diagnostic_absolute_error_counts"][8]["count"],3);
    assert!(v["accuracy_pass"].is_null());
}
#[test]
fn signed_zero_and_ties() {
    let v=compare_logits(&[0.0,-0.0],&[-0.0,0.0]);
    assert_eq!(v["bit_equal_count"],0); assert_eq!(v["max_abs"],0.0);
    assert_eq!(v["diagnostic_absolute_error_counts"][0]["count"],2);
    assert_eq!(v["reference_top"]["top1_id"],0);
    assert_eq!(v["reference_top"]["top1_tie_count"],2);
}
#[test]
fn wide_finite_f32_errors_do_not_overflow_f64() {
    let v=compare_logits(&[f32::MAX,1.0],&[-f32::MAX,1.0]);
    assert!(v["max_abs"].as_f64().unwrap().is_finite());
    assert!(v["rmse"].as_f64().unwrap().is_finite());
}
#[test]
fn all_finite_bf16_bitpatterns_json_roundtrip() {
    let values:Vec<f32>=(0u32..65536).map(|bits| f32::from_bits(bits<<16)).filter(|v|v.is_finite()).collect();
    assert_eq!(values.len(),65280);
    let encoded=serde_json::to_vec(&values).unwrap();
    let decoded:Vec<f32>=serde_json::from_slice(&encoded).unwrap();
    assert_eq!(values.len(),decoded.len());
    for (expected,actual) in values.iter().zip(decoded) { assert_eq!(expected.to_bits(),actual.to_bits()); }
}
'''
p=A/'statistics-tests.rs';p.write_text('use serde_json::json;\n'+functions+tests)
subprocess.run(['docker','cp',str(p),'moa2026-dev:/tmp/moa-model-fixed-prefix-e55/statistics-tests.rs'],check=True)
remote='''from pathlib import Path
import subprocess
root=Path('/tmp/moa-model-fixed-prefix-e55')
libs=list((root/'e55/target/release/deps').glob('libserde_json-*.rlib'))
assert len(libs)==1,libs
subprocess.run(['rustc','--edition=2024','--test',str(root/'statistics-tests.rs'),'--extern','serde_json='+str(libs[0]),'-L','dependency='+str(root/'e55/target/release/deps'),'-o',str(root/'statistics-tests')],cwd=root/'e55',check=True)
subprocess.run([str(root/'statistics-tests'),'--nocapture'],check=True)
'''
r=subprocess.run(['docker','exec','moa2026-dev','python3','-c',remote],capture_output=True,text=True)
(A/'statistics-tests.log').write_text(r.stdout+r.stderr)
print(r.stdout+r.stderr,end='')
r.check_returncode()
