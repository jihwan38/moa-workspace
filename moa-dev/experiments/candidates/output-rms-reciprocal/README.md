# O RMSNorm scalar fusion과 reciprocal 비교

2026-09-21, SDK/compiler 0.8.1. **권장 후보는 역수를 쓰지 않는 `rmsnorm.fused-div.rs` + `ops.fused-div.rs` / `fused-div.patch`다.** 독립 `ops::sliding_attention_output` 컴파일에서 baseline 32,842→**32,560 cycles(-282, -0.86%)**다. reciprocal를 추가해도 같은 값이므로 원래 원소별 `x / rms`를 유지한다. 모든 후보는 **NPU 미검증**이다.

처음 root에게 전달한 frozen `rmsnorm.rs` + `ops.rs` / `candidate.patch`는 **fused reciprocal variant**로 그대로 보존했다. 파일명을 구분하여 적용해야 한다. 전체 `ops.rs`를 라이브 파일 위에 복사하면 다른 실험을 되돌릴 수 있으므로 증분 patch를 사용한다. 원래 generic `shared::rmsnorm::normalize`는 모든 후보에서 byte-identical하게 유지하며, 파일 뒤에 별도 helper를 추가하고 O entrypoint의 호출 하나만 바꾼다.

## 유지하는 데이터와 연산

입력은 O projection/channel-scale을 마친 BF16 H3840 벡터다. 기존과 똑같이 8개 유효 slice에 H480씩 나누며 각 slice에서 BF16→FP32, 제곱, H480 합, `/3840`을 수행한다. 8-slice FP32 inter-slice sum, EPS=1e-6, Sqrt 순서도 유지한다. RMS scalar를 나타내는 `Dummy8`은 동일 scalar의 8개 복제이며, 기존과 같이 `H/480`으로 동일 물리 위치에서 이름 변경하여 각 H480 chunk에 broadcast한다.

기존 경로는 다음 세 Main pass를 사용한다.

```text
H480 squared sum /3840
→ inter-slice sum + EPS → FP32 scalar DM
→ Sqrt → FP32 scalar DM
→ x/rms * gamma → FP32 H480 DM → 기존 gather/BF16 cast
```

권장 fused-div 후보는 inter-slice sum 이후 같은 pass 안에서 EPS와 Sqrt를 수행하여 중간 scalar DM write/read와 Main pass 하나만 제거한다. 긴 H480의 `DivF` 및 gamma `Mul0`, FP32→BF16 위치, 마지막 gather는 원래 코드와 같다. 새 rounding 단계나 reciprocal 곱셈은 없다.

단, EPS는 기존 `vector_clip(Add, EPS)` 대신 `vector_fp_binary(AddF, EPS)`로 표현한다. SDK pipeline에서 Clip은 Fp stage보다 뒤라 Clip→Sqrt 순서로 같은 pass를 구성할 수 없기 때문이다. 둘 모두 소스 수학상 FP32 add이지만 다른 실행 유닛을 사용하므로 장치의 연산·fusion 정확성 확인은 필요하다. Sqrt의 입력이나 EPS 순서를 바꾸지는 않는다.

## reciprocal 실험과 SDK 제약

plain reciprocal 후보는 원래 Sqrt pass의 끝에서 FP32 `1/rms`를 계산한다. 이후 원소별 `x/rms`를 `Mul1(x, reciprocal)`로 바꾸고 gamma `Mul0`는 유지한다. fused reciprocal는 EPS/Sqrt/reciprocal까지 한 scalar pass로 합친다.

처음 제안된 Sqrt→Stash→BitAnd(0)→ClipAdd(1)→Div(Stash)는 SDK typestate에서 거절됐다. Sqrt는 Way4/Fp이고 Logic은 Way8의 Tag/Logic에서만 지원한다. 또한 Clip→FpDiv 역방향 transition도 지원되지 않는다. 이를 우회하려고 산술 `r*0`을 사용하지 않았다. SDK의 공식 `vector_fp_div_with_mode(BinaryArgMode::Mode10, 1.0f32)`가 `operand/mainstream = 1/r`를 직접 표현하므로 이를 사용했으며 exact compile에 성공했다. Infinite r에도 numerator가 상수 1이므로 `Inf*0` 문제가 없다.

`rmsnorm.stash-logic-failed.rs`와 ignored `artifacts/output-rms-reciprocal/stash-logic-way4-failure.log`가 실패 근거다. 관련 실제 SDK 위치는 `src/engine/vector/tensor/vector_tensor.rs`의 `vector_logic`, `vector_fp_div_with_mode`, `src/engine/vector/op/arg_mode.rs`의 `Mode10`, `src/engine/vector/stage/markers.rs`의 단계 전이다.

## 정적 결과

| 변형 | makespan | Main union | Vector union |
| --- | ---: | ---: | ---: |
| E15 O baseline | 32,842 | 10,626 | 10,122 |
| reciprocal만 | 32,842 | 10,626 | 10,122 |
| scalar fusion + reciprocal | 32,560 | 10,344 | 9,840 |
| **scalar fusion + 원래 DivF** | **32,560** | **10,344** | **9,840** |

DMA union 20,429, Sub union 3,010은 모두 동일하다. 별도 Sqrt Main node 282 cycles가 사라지고 fused inter-slice/scalar node는 2,579 cycles로 유지된다. 원소별 DivF를 Mul1로 치환하는 것만으로 정적 지연 개선은 없다. 개별 context 합계는 전체 makespan과 혼동하지 않는다. 실제 성능·정확성은 NPU 결과가 필요하다.

## CPU 수치 확인 범위

`compare_cpu.py`는 NumPy FP32 연산과 BF16 round-to-nearest-even 모델에서 reciprocal 치환의 추가 반올림을 조사한다. 장치 division/sqrt 근사, denormal 처리 또는 RNGD reduction tree를 재현하는 검사가 아니다.

- 65,280개의 모든 finite BF16 x code × 64개 RMS × 6개 gamma를 검사했다. finite output pair 24,987,846개 중 FP32 차이 6,010,766개, 최종 BF16 차이 154개였다. normal BF16 reference 기준 최대 상대 차이는 약 0.7692%였다.
- 여러 크기의 synthetic BF16 H3840 벡터 64개에서 같은 rounded RMS를 두 방식에 제공했다. 245,760개 결과 중 BF16 차이 2개, 최대 절대 차이 0.00390625였다.
- bitwise zero 생성/상수 numerator는 Inf에서도 `1/Inf=0`을 만들었다.

이 결과는 reciprocal가 bitwise 동치가 아님을 보여준다. **권장 fused-div 후보에는 이 reciprocal 치환 자체가 없다.** CPU probe를 competition correctness PASS로 해석하지 않는다. 원래 mean-square reduction의 grouping, gamma, 최종 BF16 경계 및 O residual 연결을 장치에서 확인해야 한다.

## 재현 및 파일

Docker scratch `/tmp/moa-output-rms-reciprocal`, baseline/candidate target을 각각 분리했다. `CARGO_HOME=/tmp/moa-extra-seeds-e5-four-row-extra/cargo-home`을 사용하고 다음 entrypoint만 컴파일했다.

```sh
CARGO_TARGET_DIR=/tmp/moa-output-rms-reciprocal/candidate-target \
FURIOSA_OPT_OUT_DIR=/tmp/moa-output-rms-reciprocal/candidate-target/furiosa-opt/kernel \
cargo furiosa-opt compile ops::sliding_attention_output --exact \
  --dump-schedule schedules/fused-div.json
```

`static-comparison.json`에 node/context 근거, `cpu-rounding.json`에 CPU 결과, `evidence.json`에 실제 compile source hash와 불변 조건이 있다. Raw JSON/log는 Git-ignored `moa-dev/experiments/artifacts/output-rms-reciprocal/`에 저장했다. 라이브 source·공유 target·Git·NPU는 변경하지 않았다. root의 하드웨어 검증 전에는 기존 baseline을 대체한 것으로 간주하지 않는다.
