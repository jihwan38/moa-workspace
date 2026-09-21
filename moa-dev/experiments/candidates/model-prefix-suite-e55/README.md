후속 [실제 Arena71239 결과](../../../../docs/experiments/e55-model-prefix-suite.md)는4개 prefix·82token·1,048,576개 finite logits의 비교 완료를 기록한다. 아래는 원격 실행 전 준비한 source/build 검증 기록이며 정확도 판정은 여전히 null이다.

원본 `8559916`과 E55 `d50eb15`를 네 개의 고정 chat prefix로 비교하기 위한 **별도 진단 host**다. 두 release build, 실제 컴파일 입력, 기존 kernel image 귀속, CPU guard 검증을 완료했다. 이 후보 준비 과정에서는 NPU 실행·샘플링·정답 채점·정확도 합격 판정을 수행하지 않았다. 기존 [단일-prefix 진단](../model-fixed-prefix-e55/README.md)은 변경하지 않았다.

고정 순서는 다음과 같다.

1. `Hi`
2. `What is 2 + 2?`
3. `대한민국의 수도는?`
4. `Complete the Python expression: len([1, 2, 3])`

## 입력·실행·비교

[probe_model_prefix_suite.rs](probe_model_prefix_suite.rs)는 `load::model_dir`와 `load::tokenizer_path_in`으로 tokenizer를 먼저 열고, 각 user message에 공개 `Tokenizer::encode_chat(..., false)`를 호출한다. 이는 `thinking=false`, `add_generation_prompt=true`, `add_special_tokens=false`인 기존 공식 text 경로다. Reference metadata와 모든 case의 ID·shape·finite 검증, 실제 encode ID 일치, **네 prefix 전체 토큰 수 ≤128** 검사를 `Device::new`와 모델 로딩 전에 끝낸다. Chat template가 추가한 토큰도 128개에 포함된다. 이 제한은 70초 예산을 돕는 guard일 뿐, 로딩·실행 시간 상한을 보장하지 않는다.

```text
reference/네 prompt/token ID/전체 token budget 검사
→ Device::new → load_model (host당 한 번)
→ 각 case: 새 Workspace::new → 새 begin_decode
→ pos0부터 고정 token ID로 Decode::run; 마지막 위치만 compute_logits=true
→ read_logits: 전체262,144개 finite 확인
→ Workspace/Decode drop 후 다음 case
```

`Workspace::new`는 모든 KV cache·position offset·logits를 새로 할당하고 초기화한다. 매 case는 독립 Workspace/Decode scope를 가지며 앞 case의 prefix/cache를 재사용하지 않는다. Model/Device는 같은 host의 네 case 동안 재사용한다. Model 로딩 후 `load::tokenizer_path(&model)`도 사전 검증 경로와 같은지 확인한다. Text-only이므로 multimodal 삽입은 없고 argmax 결과를 다음 입력으로 넣지 않는다.

- `--emit reference.json`: 고정순서4개 case의 prompt·token IDs·전체 f32 logits를 **하나의 suite JSON 파일**에 저장한다. 각 case 완료 때 짧은 `reference_case_ready`를 flush하고, 네 case 완료·파일 flush/sync 후 `suite_reference_written`을 출력한다. 기존 파일을 덮어쓰지 않는다.
- `--compare reference.json`: schema·모든 prompt/index/순서·case 수4·각262,144원소·모든 finite·각 ID·전체 token count를 검증하고, 실제 encode IDs가 모두 같을 때만 실행한다. 각 case의 `case_logits_compared`에는 max/mean absolute error, RMSE, f32 bit-equal 수, raw top1/top2/margin, 일반 absolute-error threshold별 개수가 들어간다. 모든 case가 끝나면 `suite_comparison_complete`를 출력한다.

Reference에는 총 **1,048,576개** logits가 저장되며 전체 logits를 stdout으로 출력하지 않는다. 통계는 f64로 누적하고 기존 단일-prefix host의 `top_summary`/`compare_logits` 함수 본문을 그대로 사용한다. Top1은 suppressed token을 제외하지 않은 raw-logits 통계이며 답변이나 sampling 결과가 아니다. Signed zero는 bit-equal 수에서 구분한다.

파일·metadata·ID·원소 수·nonfinite·device/model 오류는 nonzero exit다. 큰 수치 차이에는 임의 합격 기준을 붙이지 않는다. `accuracy_acceptance_threshold`와 `accuracy_pass`는 항상 `null`이다. **Exit0나 구조적 완료는 정확도 PASS가 아니다.** Collector는 중간 summary만으로 완료를 판단하면 안 되며, 원본의 `suite_reference_written`, 후보의 정확히4개 case summary와 `suite_comparison_complete`, runner exit0를 함께 확인해야 한다.

Reference 자체에는 모델 자산이나 host hash가 없다. [remote_entrypoint.sh](remote_entrypoint.sh)는 같은 job 환경에서 **asset inspector 기본모드 → 원본 emit → reference SHA256 → E55 compare → asset inspector `--header-only`**를 순차 실행하고 두 host와 inspector SHA도 출력한다. 기본 inspector는 weight 전체 SHA256을 40초 cooperative budget으로 시도하고, 후단은 header/stat와 작은 metadata 파일의 hash를 다시 확인한다. Inspector exit0는 weight full hash 완료를 보장하지 않으므로 collector가 첫 `full_file.status=complete`와 SHA, 마지막 header/stat 및 metadata hash를 실제로 확인해야 한다. 전후 header/stat 일치는 실행 중 전체 payload가 불변이었다는 암호학적 증명이 아니다. ID 일치만으로 weight 동일성도 증명되지 않는다.

시간 필드는 단일 진단의 관측값이며 공식 성능 점수나 통제된 benchmark가 아니다. `load_seconds`는 Device 생성과 모델 로딩, case `workspace_seconds`는 새 Workspace/Decode 준비, `decode_and_read_seconds`는 prefix 실행과 logit 다운로드를 포함한다. 별도 실행에서 full-weight hash가 약5.5초였어도 이 네-case package가 70초 안에 끝난다는 보장은 없다.

## 소스·빌드·검증

[prepare_sources.py](prepare_sources.py)는 원본 [45개 audited 파일](../qkv-boundary-probe-e53/original-source-audit.json)+pinned Cargo.lock과 E55 [46개 frozen 입력](../qkv-bf16-fused-channel-scale-e54/compiled-source-manifest.json)을 검증·선택 복사한다. 각 crate에 기존46개를 그대로 두고 동일한 `src/bin/probe_model_prefix_suite.rs` 하나만 추가한다. 기존 probe, device/ops, 공식 harness, dependency를 수정하지 않는다. 각 snapshot은 추가 파일 없이 정확히47개 입력이다.

[source-verification.json](source-verification.json)은 frozen snapshot과 실제 Docker 컴파일 입력47개를 대조한다. [original](original-compiled-source-manifest.json)과 [E55](e55-compiled-source-manifest.json)의 첫 target은 모두 비어 있었다. 각각 독립 경로에서 다음 명령을 실행했다.

```bash
# variant는 original 또는 e55; build.sh가 두 경로를 모두 처리한다.
cd /tmp/moa-model-prefix-suite-e55/original
CARGO_HOME=/home/moa/.cargo \
CARGO_TARGET_DIR=/tmp/moa-model-prefix-suite-e55/original/target \
FURIOSA_OPT_OUT_DIR=/tmp/moa-model-prefix-suite-e55/original/target/furiosa-opt/kernel \
cargo furiosa-opt build --release --locked --bin probe_model_prefix_suite
```

[build.sh](build.sh) / [build.py](build.py)는 입력 준비·실제 manifest·빌드 로그·host·standalone kernel 복사를 재현한다. 설치된 SDK0.8.1과 `nightly-2026-05-01`/Cargo cache를 재사용했고, original46.18초·E55 57.60초에 첫 release build가 통과했다. Lock SHA는 `aaff8dd41f9c643dd8c9faa358b9bc780e88a42c09172c1a5d7a0726d07efe09`다. [환경 로그](../../artifacts/model-prefix-suite-e55/toolchain-versions.log)에 실제 버전을 기록했다. 재현 후 `verify_source_inputs.py`, `verify_registry.py --workspace <workspace>` 결과를 각각 해당 JSON에 저장하고 `test_guards.py`, [prepare_package.py](prepare_package.py)를 실행하면 동일 검증·package 구성을 확인할 수 있다.

[verify_registry.py](verify_registry.py)의 [registry 결과](registry-summary.json)는 각 host에 정확히10개 text kernel과 empty generic key, 완전히 파싱된 sorted ELF registry가 있음을 확인한다. Embedded image와 standalone `.bin`이 같고, **각 버전의 기존 단일-prefix host와10개 모두 byte-identical**하다. Stage1 세 kernel도 각 원본/E55 기존 `test_kernels` host와 같다. 원본과 E55 사이에는 Stage1 세 개와 `full_attention_output` 총4개가 다르고 나머지6개가 같다. 네 번째 변경의 범위는 [shared-caller 감사](../../../../docs/experiments/e47-shared-caller-audit.md)를 따른다. 이러한 image 귀속 검증은 모델 정확도 검증이 아니다.

[test_guards.py](test_guards.py)는 실제 새 host의 validation/statistics 함수만 추출해 SDK 없이 CPU 테스트5개를 수행했다([로그](../../artifacts/model-prefix-suite-e55/guard-tests.log)). Case 누락/순서/metadata, 128/129토큰 및 overflow, 모든 case의 truncated/nonfinite logits, 모든 case의 ID 불일치·token 범위, suite JSON 왕복·큰 오차에서도 null 판정을 확인했다. SDK context 값 `E=32768`은 원본 axes와 대조한 stub만 사용하므로 NPU나 runtime allocation 검증은 아니다. 독립 source review에서도 false accuracy PASS나 stale cache/logits 비교 결함은 발견하지 못했다.

| Host | Bytes | SHA256 |
|---|---:|---|
| [원본](../../artifacts/model-prefix-suite-e55/original/probe_model_prefix_suite) | 11,423,152 | `d0ae91f22f2b272fcf2fb77462c02bbc5b2a8ec5eaf79e0dc614bb62ce61c4e5` |
| [E55](../../artifacts/model-prefix-suite-e55/e55/probe_model_prefix_suite) | 11,421,872 | `52bd86d125e34da224738bf3a26a0672f8cefbcfa720b376ccf3a890c51afd57` |

Probe source SHA는 `07cfaf4646da18a78e03bb386632ac455ccf545d98b858385bded281c3104da4`다. [package](../../artifacts/model-prefix-suite-e55/package)는 두 host, runner, read-only asset inspector의 네 파일을 포함한다. Inspector ELF SHA는 `1fb587ee436c5eb27099fe6177b4b04911b9b6ea09699de4a781c41bb932cf11`이며 이 작업에서는 재빌드하지 않았다. [Evidence](evidence.json), [package manifest](package-manifest.json), [전체 manifest](manifest.json)에 최종 runner·빌드·검사·소스·package hash를 묶었다. Arena 제출·실행·결과 수집은 root가 별도로 관리한다.
