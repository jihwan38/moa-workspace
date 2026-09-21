#!/usr/bin/env python3
"""Compile exact suite validation/statistics functions without SDK or NPU calls."""
from pathlib import Path
import subprocess
if not __debug__:
    raise RuntimeError('Guard tests require assertions; Python -O is forbidden.')
C = Path(__file__).resolve().parent
R = C.parents[3]
A = R / 'moa-dev/experiments/artifacts/model-prefix-suite-e55'
source = (C / 'probe_model_prefix_suite.rs').read_text()
prefix = source[:source.index('#[tokio::main]')]
prefix = '\n'.join(line for line in prefix.splitlines()
                   if not line.startswith('use furiosa_opt_'))
# The only SDK value these host validation functions reference is context size.
# This value is independently checked against the unchanged axes.rs below.
assert 'E = 32768' in (A / 'e55/crate/src/axes.rs').read_text()
prefix += '\nstruct E; impl E { const SIZE: usize = 32768; }\n'
tests = r'''
fn valid_reference() -> ReferenceSuite {
    ReferenceSuite { schema: SCHEMA.to_owned(), thinking: false, add_generation_prompt: true,
        case_count: 4, element_count_per_case: VOCABULARY, max_total_tokens: MAX_TOTAL_TOKENS,
        total_token_count: 4, cases: PROMPTS.iter().enumerate().map(|(index, prompt)| ReferenceCase {
            case_index: index, prompt: (*prompt).to_owned(), token_ids: vec![index],
            logits: vec![0.0; VOCABULARY],
        }).collect() }
}
#[test]
fn exact_four_case_order_and_metadata_are_required() {
    let mut r = valid_reference(); validate_reference(&r).unwrap();
    r.cases.swap(0, 1); assert!(validate_reference(&r).is_err());
    r.cases.swap(0, 1); r.cases.pop(); assert!(validate_reference(&r).is_err());
    let mut r = valid_reference(); r.thinking = true; assert!(validate_reference(&r).is_err());
    let mut r = valid_reference(); r.cases[3].prompt.push('!'); assert!(validate_reference(&r).is_err());
    let mut r = valid_reference(); r.total_token_count += 1; assert!(validate_reference(&r).is_err());
}
#[test]
fn token_budget_boundary_and_overflow_are_rejected_before_execution() {
    assert_eq!(validate_total_tokens([32, 32, 32, 32]).unwrap(), 128);
    assert!(validate_total_tokens([32, 32, 32, 33]).is_err());
    assert!(validate_total_tokens([usize::MAX, 1]).is_err());
    let mut r = valid_reference();
    for case in &mut r.cases { case.token_ids = vec![1; 33]; }
    r.total_token_count = 132; assert!(validate_reference(&r).is_err());
    assert!(validate_encoded_tokens(&vec![vec![1; 33]; 4], None).is_err());
}
#[test]
fn every_case_has_exact_finite_full_vocabulary() {
    for index in 0..PROMPTS.len() {
        let mut r = valid_reference(); r.cases[index].logits.pop(); assert!(validate_reference(&r).is_err());
        let mut r = valid_reference(); r.cases[index].logits[VOCABULARY - 1] = f32::NAN;
        assert!(validate_reference(&r).is_err());
        let mut r = valid_reference(); r.cases[index].logits[0] = f32::INFINITY;
        assert!(validate_reference(&r).is_err());
    }
}
#[test]
fn every_case_requires_exact_encoded_ids_and_valid_token_domain() {
    let r = valid_reference(); validate_reference(&r).unwrap();
    let ids: Vec<_> = r.cases.iter().map(|case| case.token_ids.clone()).collect();
    assert_eq!(validate_encoded_tokens(&ids, Some(&r)).unwrap(), 4);
    for index in 0..4 {
        let mut bad = ids.clone(); bad[index][0] += 1;
        assert!(validate_encoded_tokens(&bad, Some(&r)).is_err());
        bad[index].clear(); assert!(validate_encoded_tokens(&bad, None).is_err());
        bad[index] = vec![VOCABULARY]; assert!(validate_encoded_tokens(&bad, None).is_err());
    }
    assert!(validate_encoded_tokens(&ids[..3], Some(&r)).is_err());
}
#[test]
fn suite_json_roundtrip_and_no_accuracy_verdict() {
    let mut r = valid_reference(); r.cases[3].logits[17] = -0.0;
    let encoded = serde_json::to_vec(&r).unwrap();
    let decoded: ReferenceSuite = serde_json::from_slice(&encoded).unwrap();
    validate_reference(&decoded).unwrap();
    assert_eq!(decoded.cases[3].logits[17].to_bits(), (-0.0f32).to_bits());
    let stats = compare_logits(&[0.0, 1.0], &[1000.0, -1000.0]);
    assert!(stats["accuracy_pass"].is_null());
    assert!(stats["accuracy_acceptance_threshold"].is_null());
    assert_eq!(stats["max_abs"], 1001.0);
    let mut value = serde_json::to_value(r).unwrap(); value["unexpected"] = serde_json::json!(1);
    assert!(serde_json::from_value::<ReferenceSuite>(value).is_err());
}
'''
test_file = A / 'guard-tests.rs'
test_file.write_text(prefix + tests)
subprocess.run(['docker', 'cp', str(test_file), 'moa2026-dev:/tmp/moa-model-prefix-suite-e55/guard-tests.rs'], check=True)
remote = '''from pathlib import Path
import subprocess
root=Path('/tmp/moa-model-prefix-suite-e55')
deps=root/'e55/target/release/deps'
args=['rustc','--edition=2024','--test',str(root/'guard-tests.rs'),'-L','dependency='+str(deps)]
for name in ['serde','serde_json']:
    libs=list(deps.glob('lib'+name+'-*.rlib'))
    assert len(libs)==1,(name,libs)
    args += ['--extern',name+'='+str(libs[0])]
args += ['-o',str(root/'guard-tests')]
subprocess.run(args,cwd=root/'e55',check=True)
subprocess.run([str(root/'guard-tests'),'--nocapture'],check=True)
'''
result = subprocess.run(['docker', 'exec', 'moa2026-dev', 'python3', '-c', remote], capture_output=True, text=True)
(A / 'guard-tests.log').write_text(result.stdout + result.stderr)
print(result.stdout + result.stderr, end='')
result.check_returncode()
