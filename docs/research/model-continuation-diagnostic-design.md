# 원본 생성 토큰을 고정한 후속 logits 진단 설계

2026-09-22 KST. **설계 검토만 수행했다. 새 host 구현·빌드·Arena 실행은 하지 않았다.** 기준은 `model-prefix-suite-e55`의 원본/E55 frozen crate와 실제 `generate`/`sampling`/`Tokenizer`/`Workspace` 구현이다. 아래 읽은 host 파일과 `lm_head.rs`, `ops.rs`는 검사 시 live E55와 frozen E55가 byte-identical이었다.

기존 [4-prefix 결과](../experiments/e55-model-prefix-suite.json)는 네 문장의 마지막 분포 1,048,576개 원소를 비교했고, 네 top-1이 같았다. 다음 진단은 원본이 선택한 후속 토큰을 E55에도 그대로 입력하여 **동일한 문맥에서 다음 분포가 달라지는지** 확인한다. 원본의 정답성, 자유 생성한 두 답변의 동등성, 공식 정확도 기준을 입증하는 실험은 아니다. `accuracy_acceptance_threshold`와 `accuracy_pass`는 계속 `null`이다.

## 실제 선택 API와 종료 의미

공개 선택 함수는 [`host::sampling::sample`](../../moa-dev/experiments/artifacts/model-prefix-suite-e55/e55/crate/src/host/sampling.rs#L19)이다. `generate::sample_next`는 private이며 `read_logits` 후 이 함수를 호출한다. [`GenerationOutput`](../../moa-dev/experiments/artifacts/model-prefix-suite-e55/e55/crate/src/host/generate.rs#L169)은 문자열·토큰 개수·종료 이유만 반환하므로, `generate()` 결과 문자열을 다시 tokenize해서 원래 선택 ID를 복구해서는 안 된다.

진단의 선택 정책은 공개 API에 다음 값을 전달하는 것이다.

```rust
SamplingConfig { temperature: 1.0, top_k: 1, top_p: 1.0 }
// 기존 rand 의존성의 rand::rng() 사용
// banned = HashSet::from([generate::EOI_TOKEN, generate::EOA_TOKEN])
```

- `sample`은 먼저 상위 `top_k + banned.len()` 후보를 남기고, 금지 ID를 제거한 뒤 `top_k`개로 자른다. 유한한 262,144개 logits와 정확히 두 금지 ID에서는 한 개의 허용 최댓값만 남아 확률이 1이다. RNG 호출은 발생하지만 선택은 draw에 좌우되지 않는다. `temperature = 0`만 지정하면 내부에서 `1e-5`로 clamp하므로 그것만으로 greedy를 정의하지 않는다.
- 현재 [`generate.rs:13–33`](../../moa-dev/experiments/artifacts/model-prefix-suite-e55/e55/crate/src/host/generate.rs#L13)의 금지 ID는 **258882, 258883**뿐이다. EOS 집합 **1, 106, 50**은 금지하지 않는다. 생성 중 선택된 EOS는 `Decode::run`에 다시 입력하지 않고 종료한다. 프롬프트 안의 EOS/turn ID는 정상 prefill 입력이므로 여기서 종료하지 않는다.
- 정렬은 `total_cmp`만 비교하는 unstable selection/sort다. 동점의 최저 ID 선택은 공개 계약이 아니다. 기존 suite의 `top_summary`에 있는 `lowest_token_id`는 진단 통계의 규칙이므로 sampler 선택 규칙으로 재사용하지 않는다. 실제 반환 ID, 허용 최댓값 동점 수, margin을 따로 남긴다. `total_cmp`는 `-0.0`과 `+0.0`도 구분한다.
- CLI 기본값 [`GEMMA_RECOMMENDED`](../../moa-dev/experiments/artifacts/model-prefix-suite-e55/e55/crate/src/host/sampling.rs#L12)는 `top_k = 64, top_p = 0.95`다. 제안한 greedy는 동일한 공개 sampler의 다른 설정이며 CLI의 기본 확률적 생성과 같다고 표현하지 않는다.

## 정확한 입력·cache·checkpoint 순서

프롬프트와 순서는 기존 suite의 `Hi`, `What is 2 + 2?`, `대한민국의 수도는?`, `Complete the Python expression: len([1, 2, 3])`를 유지한다. [`Tokenizer::encode_chat(messages, false)`](../../moa-dev/experiments/artifacts/model-prefix-suite-e55/e55/crate/src/host/tokenizer.rs#L22)는 `add_generation_prompt: true`, `thinking: Some(false)`로 template을 적용하고 별도의 special-token 추가 없이 encode한다. 실제 파일로 재계산한 ID가 reference와 같아야 한다. text-only `splice_multimodal_tokens(ids, None, None)`는 ID를 그대로 유지한다.

case마다 [`Workspace::new`](../../moa-dev/experiments/artifacts/model-prefix-suite-e55/e55/crate/src/host/runtime.rs#L157)와 `begin_decode`를 새로 호출한다. `Workspace::new`가 KV cache를 0으로 초기화하며 `begin_decode` 자체는 cache 초기화가 아니다. 한 case의 prefill과 continuation 동안 같은 Workspace/Decode를 유지하고 case 종료 뒤 drop한다. 매 생성 토큰마다 Workspace를 새로 만들면 문맥이 사라진다.

원본의 case별 prompt 길이를 `P`, 생성 예산을 8로 두면 다음과 같다. 이는 [`generate.rs:202–228, 246–310`](../../moa-dev/experiments/artifacts/model-prefix-suite-e55/e55/crate/src/host/generate.rs#L202)의 계산·선택 순서를 따른다.

1. prompt ID를 `pos = 0..P-1`에 입력한다. 마지막 prompt token만 `compute_logits = true`다.
2. checkpoint `j = 0`에서 `read_logits`를 한 번 호출하고, 그 동일한 전체 배열을 저장·검사·`sample`에 사용한다. 이 분포는 첫 continuation ID `y[0]`를 선택한다.
3. 각 `j = 0..7`에서 선택한 원본 ID `y[j]`가 EOS면 해당 terminal checkpoint까지 저장하고 종료한다. EOS는 completion 개수에 포함하거나 feed하지 않는다.
4. EOS가 아니고 `j == 7`이면 길이 제한으로 종료한다. 마지막 여덟 번째 선택 ID도 다시 feed하지 않는다. 그렇지 않으면 `decode.run(..., y[j], ..., P + j, true)` 후 다음 checkpoint로 간다.

따라서 **case당 최대 8개, 전체 최대 32개 checkpoint**다. 원본이 먼저 EOS를 선택하면 더 적으며, 이를 불완전 실행으로 오인하지 않도록 reference에 실제 종료 이유와 개수를 저장한다. EOS 이후에도 강제로 32개를 채우는 정책은 현재 `generate`의 종료 의미와 달라 여기서는 채택하지 않는다. 여덟 토큰을 모두 feed하고 처음 분포까지 기록하면 case당 9개가 되는 off-by-one도 피한다.

E55는 동일한 prompt를 prefill한 후 각 checkpoint의 **원본 `y[j]`**를 feed한다. E55 자신의 sampler 결과는 비교 통계에만 쓴다. E55가 먼저 EOS를 선택했을 경우에도 reference가 계속되면 원본 ID를 따라가되, counterfactual EOS 차이를 기록한다. case별 `first_legal_greedy_disagreement_step`은 첫 sampler 선택 불일치의 `j`이며 끝까지 같으면 `null`이다. 이때 양쪽 허용-token margin도 남긴다. E55의 선택을 실제로 feed하면 이후 문맥이 달라져 같은 입력에 대한 logits 비교가 아니게 된다.

`separate_reasoning = false`, stop string 없음, callback 취소 없음으로 범위를 고정한다. `thinking=false`가 생성 channel ID 100/101을 금지한다는 뜻은 아니다. 선택된 ID를 strip하거나 chat template을 매 단계 다시 적용하지 않는다. 텍스트 표시가 필요하다면 기존 `Tokenizer::decode_stream(prompt_ids)`와 `step(id)`를 별도로 사용한다. token-level 진단만 수행할 경우 문자열 표현·callback의 동작까지 검증했다고 주장하지 않는다.

## 실행 전후의 필수 검증

실제 tokenization 결과로 `Device::new` 전에 다음을 검사한다. 기존 자산에서 prompt 길이는 `[14, 21, 19, 28]`, 합은 82였으나 이를 검사 대신 가정해서는 안 된다.

- 네 case의 순서·prompt·chat 설정·ID가 reference와 일치하고, ID가 비어 있지 않으며 전부 `W = 262144` 미만이어야 한다.
- 각 `P + 8 <= E = 32768`, 전체 `sum(P + 8) <= 128`을 checked arithmetic으로 검사한다. 현재 ID에서는 보수적 예산이 **114**다. 실제 forward 수의 최댓값은 모델 하나당 `sum(P) + 4 * 7 = 110`이다. 마지막 선택 ID는 feed하지 않기 때문이다.
- reference schema, sampler 설정, banned/EOS 집합, checkpoint 순서, `position = P - 1 + j`, 원본 선택 ID와 feed history, 종료 이유, 원소 개수·finite 상태를 모두 검증한다. EOS는 마지막 checkpoint에서만 선택될 수 있고, 비-EOS 길이 종료는 정확히 8개여야 한다. 선택 ID가 banned가 아닌 sampler 최댓값임을 확인하되 임의 최저-ID tie 규칙을 강제하지 않는다.
- 각 `read_logits`는 정확히 262,144개 finite 원소여야 한다. 실제 배열을 확보한 뒤 전체 원소를 f64 차이/RMSE 누적으로 비교한다. 여기서 읽는 값은 [`final_norm_and_logits`](../../moa-dev/experiments/artifacts/model-prefix-suite-e55/e55/crate/src/ops.rs#L252)의 softcap 이후 BF16 출력이 f32로 확장된 값이다. softcap 이전 accumulator를 관측하는 것은 아니다.

checkpoint별 max/mean absolute error, RMSE, bit-equal 개수, 기존 절대 오차 구간별 counts와 함께 실제 API의 원본/E55 선택 ID, 선택 일치 여부, EOS 여부, 허용 top-1 margin과 tie 수를 기록한다. 원본 선택 ID의 E55 내 rank를 표시한다면 `1 + 그 ID보다 total_cmp상 엄격히 큰 허용 logits 개수`처럼 동점 처리 규칙을 명시한다. 첫 선택 불일치 이후에도 원본 문맥을 유지한 전체 분포 비교를 계속한다. 낮은 margin 지점을 만나지 못할 수도 있으며, 32개 이하의 짧은 경로가 일반 정확도를 보장하지 않는다.

구조 오류·비유한 값·ID/파일 불일치·실행 중단은 nonzero exit로 처리한다. 수치 차이나 선택 차이 자체를 임의 tolerance로 accuracy PASS/FAIL 처리하지 않는다. collector는 terminal exit 0, 원본의 완결된 네 reference, 각 case의 정확한 실제 checkpoint 수, 전 원소 비교 완료, 마지막 suite-complete 이벤트를 함께 요구해야 한다. 중간 progress나 일부 case 완료를 전체 성공으로 간주하면 안 된다.

## 파일 크기와 70초 예산

최대 32개 분포는 **8,388,608개 f32**, 배열만 **32 MiB**다. JSON은 숫자 표현과 구분자 때문에 더 크며 이 진단에서는 아직 실측하지 않았다. 한 giant JSON을 반복 파싱하기보다 manifest와 case/step별 고정 상대경로 JSON으로 나누고 `BufWriter`/`BufReader`로 한 checkpoint씩 저장·비교하는 방식이 적합하다. 완성된 manifest는 모든 checkpoint 기록 후에만 publish하고 파일명·개수·sha256sum을 runner에서 고정한다. 전체 logits는 stdout에 출력하지 않는다. candidate는 검증한 checkpoint를 다시 읽더라도 reference 전체를 NPU 실행 전에 구조 검증하는 단계가 필요하다. serde/serde_json/std와 기존 rand로 구현 가능하며 새 의존성은 필요하지 않다.

기존 71239의 같은-job 자산 검사+원본/E55 네 prefill은 **34.074초**였다. 이는 새 workload의 시간 보장이 아니다. [`Decode::run_layers`](../../moa-dev/experiments/artifacts/model-prefix-suite-e55/e55/crate/src/host/runtime.rs#L365)는 `compute_logits=true`일 때마다 마지막 vocab projection을 실행한다. [`lm_head::logits`](../../moa-dev/experiments/artifacts/model-prefix-suite-e55/e55/crate/src/device/shared/lm_head.rs#L15)의 16-band loop는 `W * H * 2 = 2,013,265,920 B`의 BF16 weight를 논리적으로 읽는다. 기존 4회에서 최대 32회로 늘면 모델당 28회, 원본/E55 합계 56회의 logits head가 추가된다. 해당 weight payload 합은 **105 GiB**이며 여기에 56회의 decoder token 계산, logits 다운로드, sampler, JSON I/O가 더해진다. 이 숫자는 구현상 전송량 계산이며 실측 대역폭이나 실행 시간 추정이 아니다.

토큰 수만 비례시켜 70초 이내라고 단정할 근거는 없다. 구현한다면 기존 before/after load·case progress에 checkpoint별 decode/read/write 시간을 추가하고, 전체 70초 제한 아래 시도하는 bounded 진단으로 취급해야 한다. 호스트 deadline 확인은 다음 launch를 막을 수 있지만 이미 실행 중인 NPU 작업의 완료를 보장하지 않는다. 시간 부족 시 checkpoint를 조용히 생략하거나 완료로 표시하지 않고 불완전 실행으로 종료한다. 8-step 완료 여부가 확인되기 전에는 성능 측정이나 시간 보장으로 사용하지 않는다.

원본/E55 host·47-input manifests·10-device-image registry를 기존 방식으로 별도 묶고, 동일 job에서 full asset hash를 먼저 수집한 뒤 두 모델을 실행하며 마지막 header/stat·metadata 검사를 유지한다. 이는 두 실행의 자산 provenance를 강화하지만 마지막에 weight 전체를 재해시하지 않는 한 외부의 탐지 불가능한 payload 변경까지 배제하는 증명은 아니다. reference array를 job 밖에 내려받지 않는다면 collector가 수치통계를 독립 재계산한 것이 아니라 host 보고를 검증한 것임도 계속 구분한다.

## 확인한 소스의 SHA-256

아래 경로는 모두 `moa-dev/experiments/artifacts/model-prefix-suite-e55/e55/crate/` 기준이다. 표의 host 파일 네 개와 `lm_head.rs`는 frozen 원본과 E55 사이에도 byte-identical임을 확인했다. 구현 시 해당 frozen 부모와 대조하고 새 host만 추가해야 한다.

| 경로 | SHA-256 |
|---|---|
| `src/host/generate.rs` | `03939905181ef43716feb9dd93e52a8e6e19ec51be0df9f680ad8b81949ae1cb` |
| `src/host/sampling.rs` | `71ee4a3dda3662f1242b421636c10c46b0255d6557d85f694b940882db1a6a11` |
| `src/host/tokenizer.rs` | `7622d6e546c79befc68a93dbaab8e45f07fec1ed3ddb381f83a0c868a9ca64ce` |
| `src/host/runtime.rs` | `207e7928f91bbb47dfd3742efc8f141f0529349e4b7bb243c5a735ff03ed75bc` |
| `src/device/shared/lm_head.rs` | `98fe14525460ea811452a97f38e543cdd37231b6269750b69a39b59a8aaa135e` |
| `src/ops.rs` | `ffdab428fd4c8126b77e145b75073c907732bf0fa9ca830a123fe828e7b08888` |
