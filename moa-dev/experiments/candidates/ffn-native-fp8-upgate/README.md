# Native FP8 up/gate: 설계와 scratch 검증

2026-09-21, SDK 0.8.1. E15의 up/gate에서 FP4→FP8→BF16 weight 확장을 FP4→FP8 한 번으로 줄이는 후보다. 실제 BF16 activation을 runtime power-of-two scale과 FP8 hi/lo 두 항으로 표현한다. **Post-reduce FpDiv 변형은 NPU 실행에 실패하여 미채택이다.** Compile 결과와 실행 결과는 분리해서 기록한다.

E22 job **70321**은 QKV 9개와 O 3개 비교를 통과한 뒤 첫 `decoder_feedforward` launch에서 time limit 오류로 종료됐다(exit 101). FFN 결과와 cycles가 없어 수치 오차를 판정할 수 없다. Capture의 `mlp.rs`가 고정한 `mlp.postreduce-div.rs`와 같은 hash임을 확인했다. 원인은 확정하지 않았고 재실행하지 않았다. 로그는 `../../../logs/e22-native-ffn-upgate-safe-qkv-job-70321.log`, 상세 기록은 `postreduce-div-evidence.json`이다. 세 pipeline 원본 `mlp.rs`는 NPU에 제출하지 않았다.

현재 `mlp.rs`의 4행 변형은 exact compile을 통과했다. 정적 makespan은 **176715 cycles**, 같은 FFN core의 BF16 기준 171429보다 3.08% 높다. Main은 132375→114726, Vector는 41849→59286으로 변했다. 정적 일정만으로 실제 NPU 성능을 판단할 수 없다. 기준과 후보는 E16 tail을 사용하지 않으며 `decoder_feedforward` 본문과 모든 shared FFN dependency hash가 같음을 확인했다. 원본 전체 `ops.rs` hash 차이는 QKV/O 경로의 변경 때문이다.

별도 QKV two-lane 후보에서 V 정확도 실패가 보고되어 pair-mode 경로의 hardware 동작도 조사 중이다. 이 FFN 후보의 compile 성공을 NPU 정확도 증명으로 사용하면 안 된다.

| 소스 | exact compile | 정적 cycles | 같은 E15 core 대비 |
|---|---|---:|---:|
| `mlp.rs` | PASS, F32 block DM + F32 row DM | 176715 | +3.08% |
| `mlp.postreduce-div.rs` | PASS, F32 block DM + post-reduce FpDiv | 173781 | +1.37% |

두 파일은 별도로 고정했다. 후자는 reciprocal power-of-two로 나누어 row DM만 제거한다. 이 표는 NPU 실측이 아니다.

입력 범위나 fixture를 가정한 상수 scale을 사용하지 않는다. Quantizer와 pair-mode API는 팀이 독립 작성한 `../qkv-fp8-split-two-lane/projection.rs`를 재사용했다. 외부 공개 fork 코드는 복사하지 않았다. QKV의 선행 정확도 통과는 이 FFN 후보의 정확도를 보장하지 않는다.

## 데이터 흐름과 정밀도 경계

`ops::decoder_feedforward`가 만든 BF16 정규화 입력 → `feedforward` → `prepare_ffn_fp8_split` → `project_up_and_gate` → 4행 helper → 기존 `geglu` → 기존 `project_down`으로 흐른다. Down, GeGLU, global weight scale 적용은 E15 그대로 둔다.

1. 실제 입력의 `amax(abs(x))`에서 scale `s`를 정하고, `z=x/s`, `hi=FP8(z)`, `lo=FP8(16*(z-hi))`를 만든다. Up와 gate가 같은 activation pair를 재사용한다.
2. FP4 weight는 기존 검증된 lookup으로 정확한 FP8 DM에 확장한다. 전체 BF16 weight DM은 만들지 않는다.
3. 64B FP8 DPE packet에 네 개 16원소 block을 넣고, TRF Lane 두 개로 hi/lo를 동시에 계산한다. 각 block 결과는 F32다. `hi_dot + lo_dot/16`에 기존 local scale을 F32로 곱하여 작은 F32 block DM에 저장한다.
4. 새 standalone VE pipeline이 block 결과를 F32로 합산하여 F32 row DM에 저장한다.
5. 마지막 짧은 pipeline이 row sum에 `s`를 곱한 뒤 기존 위치에서 BF16으로 변환한다. `s*hi_dot`를 먼저 계산하면 서로 상쇄할 유한 결과도 중간 overflow가 날 수 있어 합산 뒤에 `s`를 적용한다.

## SDK 매핑과 필요한 중간 저장

Up/gate는 E15와 같은 2 clusters × 240 valid slices × 32 local rows다. 현재 scale tile은 4행이다. 8행 변형도 API/lowering은 통과했지만 VRF allocator가 배치하지 못했다. 실패 원본은 `mlp.eight-row.rs`에 보존한다. Term 축 `FfnTerm=2`는 cluster의 `Dummy2`와 구분해 이름 충돌을 피한다.

| 단계 | Time / Lane / Packet |
|---|---|
| Activation TRF | Lane `[FfnTerm]`, Element `[H]`, FP8 |
| Weight stream | Time `[row4,H/32]`, Packet `[H%32]`, FP8 |
| `contract_outer` | Time `[row4,H/64]`, Lane `[FfnTerm]`, Packet `[H%64]` |
| `contract_packet` | Packet `[H/16%4]`; 각 16원소 dot을 분리 |
| `contract_time` | `[row4,H/64]` 유지, block 사이 합산 없음 |
| `contract_lane(Sequential)` | Time `[row4,H/64,FfnTerm]`, Packet `[H/16%4#8]` |
| `vector_intra_slice_unzip<FfnTerm>` | 두 term을 pair group으로 분리 |
| Pair `vector_narrow_split` | Time `[row4,H/64,1#2]`, Packet `[H/16%4]` |
| Pair arithmetic | `Mul0`: low만 1/16, `AddF` zip, `Mul1`: local scale |
| `vector_widen_concat` / commit | Packet `[H/16%4#8]`에서 padding trim, DM `[row4,H/16]` F32 |

SDK `vector/tensor/vector_tensor.rs`의 `vector_intra_slice_reduce`는 `Standalone`에만 있다. `vector_fp_zip` 이후는 `Zipped`여서 같은 pipeline에서 block sum을 할 수 없다. SDK에는 해당 조합을 금지하는 `compile_fail` 예제도 있다. 또한 FP ALU는 intra-slice reduce보다 앞에 있어, sum 뒤 `s`를 곱하려면 별도 pipeline이 필요하다. 본 후보의 두 F32 DM 저장은 이 제약을 회피하면서 기존 BF16 반올림 위치를 유지한다.

F32 row sum은 transpose engine의 `max_in_rows=2`에 맞춰 2행씩 저장한다. 한 번에 8행을 packet으로 옮기는 초기 변형은 lower 단계에서 거절됐다.

Pair 모드의 `vector_narrow_trim`은 hardware group execution-id 처리 때문에 의도적으로 제공되지 않는다. SDK `vector_tensor_pair.rs`의 지침대로 buffered `narrow_split`과 `widen_concat`을 짝으로 사용한다.

별도 `mlp.postreduce-div.rs`는 reduce 뒤 사용 가능한 전용 `vector_fp_div`로 마지막 row pipeline을 합쳤다. Canonical scalar에서 `vector_fp_div_with_mode(Mode10,1.0)`으로 reciprocal `1/s`를 한 번 만들고, block sum이 끝난 뒤 나눠서 기존 BF16 위치로 출력한다. `s`의 247개 가능한 지수 -126..120을 CPU IEEE binary32로 전수 확인했으며 reciprocal은 지수 -120..126의 정확한 finite normal 값이었다. 이 확인은 RNGD division의 rounding/FTZ 구현을 증명하지 않는다. 수학적 sum→rescale 순서는 유지하고 추가 FpDiv의 실제 오차와 비용은 NPU에서 확인해야 한다.

## 용량 계산

- TRF: `2 × 3840 × 1 = 7680B`; 기존 BF16 activation TRF와 같다.
- Weight DM의 보수적 동시 합: packed FP4 `61440B` + FP8 `122880B` = `184320B`. 기존 두 번 lookup 경로의 BF16 `245760B`가 사라진다. 실제 peak는 compiler schedule로 확인해야 한다.
- Local scale VRF: `4 × 240 × 4 = 3840B`. Dynamic scalar VRF `32B`와 합하면 `3872B`; stash는 사용하지 않는다. 초기 8행의 이론적 합계 7712B는 실제 scheduler에서 다른 live operand 및 배치 제약과 함께 수용되지 않았다.
- Tile별 추가 F32 block DM: `4 × 240 × 4 = 3840B`; F32 row DM은 `16B`다. 단순 용량 계산이며 allocation overlap이나 compiler liveness를 대신하지 않는다.
- Sequential lane fold accumulator: Lane 2 × identity-time inner extent 1 × padded packet 32 = 64 cells, SDK 한도 1024 이내다.

## 수치 보장과 한계

유한 BF16 입력과 IEEE round-to-nearest-even, FP8 subnormal 보존을 전제로 한다. `e=2^floor(log2(max(amax,2^-119)))`, `s=e/128`이면 `2^-126≤s≤2^120`, `|z|<256`, `|16*(z-hi)|≤128`이다. 두 FP8 cast가 overflow하지 않는다.

`|z|≥2^-6`인 BF16 값은 두 FP8 항으로 정확하게 표현된다. 더 작은 값의 복원 오차는 `|x-s*(hi+lo/16)|≤s*2^-14`다. 이 표현 오차만 고려한 한 행의 절대 오차 상한은 `s*2^-14 * sum_i(abs(q_i*local_scale_i))`이다. FP4 최대 6, 유한 FP8 scale 최대 448과 H=3840을 사용한 거친 상한은 `630*s`다. 이는 실용적인 정확도 예측치가 아니며, cancellation이 있으면 상대 오차에 일정한 상한이 없다.

정규화된 hi/lo와 유한 FP4/FP8 weight에 대해 한 16원소 hi dot의 절대 합은 최대 24576, lo dot은 12288이다. 따라서 `hi_dot+lo_dot/16`의 절대 합은 25344, local scale 적용 뒤에는 11354112, 240 blocks 전체에서는 2724986880이다. 이 값들은 최종 `s` 적용 전 F32의 범위에 충분히 들어간다. 보수적인 실수 절대합 계산으로, 모든 FP32 누산이 정확하다는 뜻은 아니다.

FP32 dot/reduction 반올림, NPU FP8 cast/FTZ 동작, 이후 BF16 반올림과 GeGLU는 위 표현 오차 상한에 포함되지 않는다. 최종 `s` 곱과 원래 global scales를 포함한 모든 유한 입력에 대해 F32 overflow가 없다고 주장하지 않는다. 원래 projection도 극단적인 입력과 weight 조합에서는 overflow할 수 있다. 공식 seed와 추가 seed로 별도 확인해야 한다.

검토한 구현은 컨테이너 Cargo registry의 `furiosa-opt-lower-0.8.1/src/verify/engine/contraction/lane.rs`, `furiosa-opt-std-0.8.1/src/engine/vector/tensor/vector_tensor.rs`, `vector_tensor_pair.rs`, `vector/stage/markers.rs`다. 후보 파일과 기준 파일의 hash는 `source-hashes.json`에 기록한다.
