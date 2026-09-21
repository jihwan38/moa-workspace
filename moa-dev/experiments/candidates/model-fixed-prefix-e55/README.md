후속 실제 결과는 [E55 단독71198](../../../../docs/experiments/e55-arena-model-progress.json)과 [원본/E55 비교71207](../../../../docs/experiments/e55-model-fixed-prefix-pair.md)에 있다. 아래는 원격 실행 전 진단 host를 준비한 빌드 기록이다.

원본 `8559916`과 E55 `d50eb15`에서 같은 **Hi chat prefix의 마지막 위치 logits 262,144개**를 비교하는 진단이다. 샘플링 없이 실제 모델 커널을 실행하며, 공식 Stage2 허용오차나 accuracy PASS를 정의하지 않는다. 두 host의 release build와 소스·커널 귀속 검사는 완료했다. **아직 paired 모델 실행이나 logits 비교 결과는 없다.**

별도 [E55 smoke71173](../../../../docs/experiments/e55-arena-model-smoke.json)가 model-loaded 로그 전에70초 timeout으로 종료해,70초 안에 두 모델을 연속 실행하는 계획은 보류했다. 새 host에는 `Device::new`와 `load_model` 각각의 진입·완료를 stderr로 기록하는 progress가 있다. Root가 E55-only progress job71198을 제출했지만 그 runtime 결과는 이 build 기록에 포함하지 않는다.

## 고정 입력과 비교 경로

[probe_model_logits.rs](probe_model_logits.rs)는 공식 `host/generate.rs`의 text prompt 루프를 다음 공개 API로 수행한다.

```text
Device::new(ops::embed_token.topology())
→ load::load_model
→ Tokenizer::new → encode_chat([{role:user,content:Hi}], false)
→ Workspace::new → begin_decode
→ 모든 token에 Decode::run(token, pos, compute_logits=마지막위치)
→ read_logits(&workspace.logits)
```

`false`는 `thinking=false`다. 공식 Tokenizer는 `add_generation_prompt=true`, 내부 encode의 `add_special_tokens=false`를 사용한다. Text-only이므로 multimodal splice는 필요 없다. 다음 token을 뽑거나 생성 결과를 다시 입력하지 않는다. `--compare`는 reference의 token ID와 현재 같은 template로 생성한 ID의 완전 일치를 확인한 후 decode한다. `compute_logits=false` 위치의 오래된 workspace logits는 읽지 않는다.

- `--emit reference.json`: schema·prompt 설정·token IDs·전체 f32 logits를 **job 작업 폴더의 파일에만** 저장한다. 기존 파일은 덮어쓰지 않는다. stdout에는 ID·원소 수·top 요약·시간만 출력한다.
- `--compare reference.json`: metadata, ID, 양쪽262,144원소, 모든 값의 finite 여부를 검사하고 전 원소의 max/mean absolute error, RMSE, f32 bit-equal 수, raw top1/top2/margin, 일반 absolute-error threshold별 개수를 짧은 JSON으로 출력한다.

Shape·ID·finite·파일 구조·device/model 실행 오류는 nonzero exit다. 큰 수치 오차 자체는 nonzero로 바꾸지 않으며 `accuracy_acceptance_threshold`와 `accuracy_pass`는 `null`이다. **Exit0와 `PAIR_STRUCTURAL_COMPLETION`은 정확도 통과를 뜻하지 않는다.** Top1은 suppressed token을 제외하지 않은 raw-logits 통계이며 sampling 결과가 아니다. 모든 차이와 제곱합은 f64로 계산한다.

Reference는 모델 자산이나 host image 해시를 내부에 포함하지 않는다. [remote_entrypoint.sh](remote_entrypoint.sh)는 동일 job 환경에서 원본 emit→reference SHA256→E55 compare 순으로 실행하고 두 host SHA도 출력한다. 동일 weight 파일을 사용했다는 provenance는 runner 환경·별도 자산 증거와 묶어야 한다. Token ID 일치만으로 weight 동일성이 증명되지는 않는다. 이 작업에서 weight 전체를 재해시하거나 paired runner를 실행하지 않았다.

시간 필드는 정밀 성능 평가가 아니다. `load_seconds`는 Device 생성과 model load를 함께 포함하고, `workspace_seconds`는 Workspace/Decode allocation, `decode_and_read_seconds`는 prefix 실행과 logits download를 포함한다. `total_seconds`는 JSON 파일 저장·통계 출력 전까지다. 세부 정체는 네 progress event의 elapsed time으로 구분할 수 있다.

## 소스·build·커널 귀속

[prepare_sources.py](prepare_sources.py)는 원본의 [audited45개 파일](../qkv-boundary-probe-e53/original-source-audit.json)과 pinned Cargo.lock만 선택 복사한다. 기존 원본 snapshot의 추가 `probe_output.rs`는 복사하지 않는다. E55는 [frozen46개 입력 manifest](../qkv-bf16-fused-channel-scale-e54/compiled-source-manifest.json)를 전부 검증한 뒤 복사한다. 각각 **기존 Rust source43개 + Cargo/toolchain3개를 보존하고 같은 host probe1개만 추가**한다. 기존 device/ops/공식 harness와 dependency는 수정하지 않는다.

각 frozen crate는 정확히47개 파일만 허용하며 추가 파일도 거부한다. Preparation/verification은 `python -O`를 즉시 거부한다. [verify_source_inputs.py](verify_source_inputs.py)의 [검사 결과](source-verification.json)는 host snapshot과 실제 Docker compile 입력을 모두 대조한 것이다. [원본 실제입력](original-compiled-source-manifest.json), [E55 실제입력](e55-compiled-source-manifest.json)은47개 파일의 hash/size를 기록한다.

Docker `/tmp/moa-model-fixed-prefix-e55/{original,e55}`에 각각 독립 target을 만들고 다음 명령을 실행했다. 기존 Cargo cache와 pinned toolchain을 사용했으며 live source/target은 건드리지 않았다.

```bash
# original과 e55 각각의 scratch에서 variant 경로를 맞춰 실행한다.
CARGO_HOME=/home/moa/.cargo \
CARGO_TARGET_DIR=/tmp/moa-model-fixed-prefix-e55/original/target \
FURIOSA_OPT_OUT_DIR=/tmp/moa-model-fixed-prefix-e55/original/target/furiosa-opt/kernel \
cargo furiosa-opt build --release --locked --bin probe_model_logits
```

첫 release build는 양쪽 약79초였다. Progress만 추가한 incremental build는 원본5.67초/E55 5.70초로 완료했고, 전후10개 device image는 각각 모두 동일하다. Lock SHA는 `aaff8dd41f9c643dd8c9faa358b9bc780e88a42c09172c1a5d7a0726d07efe09`다. [build.sh](build.sh)는 동일 격리 build를 재현하며 NPU를 실행하지 않는다.

[verify_registry.py](verify_registry.py)는 **ELF registry만** 검사한다. [결과](registry-summary.json)에서 각 host에 정확히10개 text kernel, empty generic key, 완전하게 파싱된 sorted registry가 있고, standalone image와 embedded image가 동일하다. 각 host의 기존 Stage1 QKV/O/FFN3개는 그 버전의 실제 측정 test_kernels host와 byte-identical이다. 소스47개 검사는 앞의 별도 script/manifest로 수행했다.

원본과 E55 사이에는 Stage1 세 개에 더해 `full_attention_output` image도 달라 총4개가 다르고 나머지6개는 같다. 네 번째 차이는 [기존 shared-caller 감사](../../../../docs/experiments/e47-shared-caller-audit.md)의 shared residual add loop 전개 범위와 일치한다. 이를 모델 정확성 검증으로 해석하지 않는다.

| 최종 host | Bytes | SHA256 |
|---|---:|---|
| [원본](../../artifacts/model-fixed-prefix-e55/original/probe_model_logits) | 11,315,056 | `8ba9530dcbc2cf4862b0179611024d13e04cb85615f554970853f22f6ed676f7` |
| [E55](../../artifacts/model-fixed-prefix-e55/e55/probe_model_logits) | 11,313,776 | `4f2dc3a8e306f30f17f5eca45cdc4ec643d323f8d9445c9c1e02857233d175fc` |

최종 probe source SHA는 `dbd549d861686004d69bb4dbd7de3ebd4d17b1b27e0701587e51a927d508684c`다. [paired package](../../artifacts/model-fixed-prefix-e55/package)에는 두 host와 runner를 두었다. Arena 제출·시간 제한·실행 여부는 root가 별도로 관리한다.

[test_statistics.py](test_statistics.py)는 실제 probe의 통계 함수 본문만 추출해 SDK 없이4개 CPU 검사를 수행했다. 알려진 max/mean/RMSE, signed zero/tie, 큰 finite 오차, **모든 finite BF16 bit pattern65,280개의 JSON 왕복 bit 보존**이 통과했다([로그](../../artifacts/model-fixed-prefix-e55/statistics-tests.log)). 이는 모델 출력이나 NPU 검사가 아니다. 독립 source review에서도 잘린 zip 비교·stale logits 읽기·허위 accuracy PASS 결함은 발견되지 않았다.

[Evidence](evidence.json)와 [manifest](manifest.json)에 소스·두 build 로그·최종 host·registry·package 해시를 기록했다. 이 작업은 격리 source/build 및 진단 준비이며 live source, Git, NPU를 변경하거나 실행하지 않았다.
