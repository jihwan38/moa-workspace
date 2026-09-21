E55 모델에서 **`ops::sliding_project_qkv`만 원본 `8559916`의 실제 커널 이미지로 복원한 수치 대조군**이다. 첫 격리 release build가 통과했고, [ELF registry 비교](registry-diff.json)에서 QKV는 원본과 byte-identical, 나머지9개 text kernel은 E55와 byte-identical임을 확인했다. 모델 정확도·성능 결과가 아니라 **실행 전 소스/빌드/이미지 귀속 검증**이다. 이 준비 과정에서 NPU·Arena·Git·live crate를 변경하거나 실행하지 않았다.

Subsequent root-managed [Arena job71290](../../../../docs/experiments/e55-model-control-qkv.json) completed.
See the [combined model results](../../../../docs/experiments/e55-model-single-kernel-controls.md).
The build evidence below remains a preparation-time snapshot.

이 대조군은 QKV 변경이 모델 logits 차이에 얼마나 기여하는지 분리하기 위한 것이며, 원본 logits를 정답으로 정의하거나 원본 구현을 채택하는 변경이 아니다. 원본은 Q/K/V full dot을 BF16으로 반올림한 뒤 channel scale을 곱하고 다시 BF16으로 반올림한다. E55의 scale-before-first-BF16 경로를 포함하여 **QKV kernel 전체**를 원본으로 되돌리므로 단일 반올림 변경만 분리하는 실험도 아니다.

## 변경 범위

[prepare_sources.py](prepare_sources.py)는 [기존 네-prefix suite](../model-prefix-suite-e55/README.md)의 original/E55 frozen47개 입력을 전부 재해시한 뒤 E55를 기준으로 다음 네 파일만 바꾼다.

- [src/ops.rs](src/ops.rs): 공개 `sliding_project_qkv`의 **본문만** 원본으로 복원한다. Signature와 본문 밖 전체 텍스트, 나머지9개 공개 ops 함수는 E55 그대로다.
- [src/device/sliding/projection.rs](src/device/sliding/projection.rs): `project_query`, `project_one_kv_matrix`, `project_key_value` 및 QKV prefix를 원본으로 복원한다. E55 `project_output`부터 파일 끝까지는 그대로 두고 해당 O suffix가 사용하는 `Dummy2` import만 유지한다.
- [src/device/sliding/rmsnorm.rs](src/device/sliding/rmsnorm.rs), [rope.rs](src/device/sliding/rope.rs): 두 파일을 원본 그대로 복원한다. 현재 실제 caller는 QKV ops뿐이다.

Shared RMSNorm/MLP/residual, sliding O suffix, full-attention helpers, loader/runtime/tokenizer, official harness, Cargo/toolchain/lock는 E55 그대로다. 원본 generic `shared::rmsnorm::normalize`는 E55에서도 이미 동일한 함수여서 추가 수정 없이 원본 QKV 호출을 지원한다. [소스 범위 검사](source-scope-verification.json)는 함수·본문·suffix 동일성을 확인한다. 독립 read-only peer review도 이 source scope에서 다른 kernel caller로의 변경 전파를 발견하지 못했다.

[control.patch](control.patch)는 E55에서의 전체 변경이다. [부모 manifest](parent-source-manifests.json), [동결 입력47개](source-manifest.json), [실제 Docker 입력47개](compiled-source-manifest.json), [전후 입력 검증](source-verification.json)을 따로 보존했다. 47개 중4개만 E55와 다르고43개는 같다. 기존46개 crate 입력에 추가된 suite host는 같은 파일 하나이며 SHA는 `07cfaf4646da18a78e03bb386632ac455ccf545d98b858385bded281c3104da4`다.

## 빌드와 image gate

[build.sh](build.sh) / [build.py](build.py)는 입력 준비, 실제 컴파일 manifest, 빌드 로그/host/standalone image 복사, [source gate](verify_sources.py), [registry gate](verify_registry.py)를 순서대로 수행한다. Registry gate가 실패하면 전체 script도 nonzero로 끝난다. Snapshot과 Docker scratch는 이 대조군 전용이며 최초 target은 비어 있었다.

```bash
cd /tmp/moa-model-control-qkv-original-e55
CARGO_HOME=/home/moa/.cargo \
CARGO_NET_OFFLINE=true \
CARGO_TARGET_DIR=/tmp/moa-model-control-qkv-original-e55/target \
FURIOSA_OPT_OUT_DIR=/tmp/moa-model-control-qkv-original-e55/target/furiosa-opt/kernel \
cargo furiosa-opt build --release --locked --bin probe_model_prefix_suite
```

설치된 SDK0.8.1, `nightly-2026-05-01`, 기존 Cargo cache를 사용했다([실제 버전](../../artifacts/model-control-qkv-original-e55/toolchain-versions.log)). 첫 build는107.09초, exit0이었다([로그](../../artifacts/model-control-qkv-original-e55/build.log), [명령·결과](../../artifacts/model-control-qkv-original-e55/build-result.json)). 이는 빌드 시간이며 NPU 성능 측정이 아니다. Cargo.lock은 부모와 동일한 `aaff8dd41f9c643dd8c9faa358b9bc780e88a42c09172c1a5d7a0726d07efe09`다.

| 산출물 | Bytes | SHA256 |
|---|---:|---|
| [Control host](../../artifacts/model-control-qkv-original-e55/probe_model_prefix_suite) | 11,435,952 | `1f676b0be83dcda19202363bb143d2b770bc67521e717a3203b9a4c8b2d34a80` |
| 복원 QKV image | 183,040 | `75eacabd18926cbefdebdded06d8bb3313b810d25d83faf8006e807d7384fe5b` |

Registry는 정확히10개 entry, empty generic key, SDK 순서와 완전한 byte 소비를 검사한다. 각 embedded image는 별도 `.bin`과 같다. 원본 suite host `d0ae91f2…`와 E55 suite host `52bd86d1…`의 전체 SHA도 고정 검사한다. **E55와 다른 image 목록은 QKV 한 개뿐**이며, QKV는 원본과 해시만 비교하지 않고 실제 image bytes 전체를 비교한다. 다른9개도 E55와 실제 bytes 전체를 비교한다. Source 검증과 ELF 검증은 서로 다른 gate이며, 어느 쪽도 NPU 출력 정확성을 증명하지 않는다.

## 동일한 진단 host

`probe_model_prefix_suite`는 기존 네-prefix host source와 동일하다. 고정순서 `Hi`, `What is 2 + 2?`, `대한민국의 수도는?`, `Complete the Python expression: len([1, 2, 3])`를 사용한다. Device 생성 전에 metadata/token ID/총128토큰 제한을 확인하고, 각 case에 새 Workspace/Decode를 만들며 마지막 token에서만 logits를 계산한다. 전체262,144개씩 finite 검사 후 전 원소 통계를 내고 `accuracy_pass`/`accuracy_acceptance_threshold`는 `null`로 유지한다.

기존 원본 suite가 만든 `reference.json`에 대해 이 host를 `--compare reference.json`으로 실행할 수 있다. 실제 root-managed remote package는 원본 reference·동일 token IDs·모델 자산 증거와 묶어야 한다. 중간 case summary만 완료로 판단하지 말고4개 case·`suite_comparison_complete`·exit0를 확인해야 한다. 128토큰 제한은 70초 완료 보장이 아니다. 원래 host의 guard/통계는 수정하지 않았으므로 동일 CPU 테스트를 중복 실행하지 않았다.

[Evidence](evidence.json)와 [전체 manifest](manifest.json)는 준비 시점의 source/build/log/image hash를 고정한다. 원격 실행 및 결과 수집은 root가 별도로 관리한다.
