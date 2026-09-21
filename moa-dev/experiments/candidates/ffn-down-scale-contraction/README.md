# Down BF16 block partial + 두 번째 scale contraction

2026-09-21, compiler/std 0.8.1. 마지막 허용 수정까지 적용한 `ops::decoder_feedforward --exact` compile이 성공했다. 전체 FFN 정적 makespan은 **170,270 cycles**, 동일 ops/residual을 쓴 `ffn-scale-contraction-streamed`의 171,727보다 1,457 cycles(0.85%) 작다. **E33 job70455에서 FFN 세 설정 모두 정확성 FAIL이었다.** Q/O는 PASS였으나 FFN max abs error는3.30/3.88/3.63, tolerance 내 비율은16.17%/12.94%/13.67%다. Median408,287 cycles는 실패 결과여서 최적화 성능으로 사용할 수 없다. [실험 기록](../../../../docs/experiments/e33-down-scale-contraction.json)이 근거다. 큰 오차를 BF16 rounding만으로 단정하지 않는다. 새 layout/zero-fill/copy/second-DPE 경로도 함께 바뀌었고 원인은 미확정이다. 이 bounded task에서 추가 디버그하지 않았다. 새 BF16 block 반올림 때문에 모든 입력에서 원본과 동치인 후보가 아니다.

기준 후보의 FFN은 E23 job70327에서 세 설정 모두 PASS, median384,728 cycles였다. 이 값은 새 down 후보의 예상 실측값으로 환산할 수 없다. 후보 source SHA256은 `a52366bd7aafbe6e3ebafdc0c735523d2bf4e7a090e2713390d99e9498e1ac80`다.

## 적용 범위와 데이터 흐름

`mlp.rs::project_down`과 새 local axis `DownScaleBlock=128`만 기준 source에서 변경했다. Up/gate/feedforward/GeGLU, ops, BF16 activation, 기존 down F4→F8→BF16 weight materialization은 유지한다. Global scale, post RMSNorm, residual/layer scalar도 유지한다. 두 clusters ×32 row groups ×8 column slices에서 slice마다 H60 rows, L1920 inputs를 처리하며, 한 column에는 16원소 scale blocks120개가 있다. 새 안은 row20씩 세 번 처리한다.

첫 BF16 contraction은 각 16원소 FP32 dot을 BF16로 반올림하여 `[row20,block120]`에 저장한다. 이 결과와 BF16 local scales를 두 번째 contraction에서 곱해 column-local FP32 합을 얻고, 기존 8-column FP32 inter-slice 합산과 최종 BF16 boundary를 유지한다.

120 blocks는 BF16 DPE의 최소16-element input packet에 맞지 않는다. 128개 **실제 live cells**인 buffer를 만들어 전체 `memset(0)` 후 첫120 cells를 실제 값으로 덮어쓴다. block120..127은 partial과 scale 모두 실제0이다. `#128`만 붙여 미초기화 값을0으로 가정하지 않는다. FP8 scale도 실제128-cell buffer에 옮겨 Main lookup으로 BF16 변환한 다음 Sub TRF에 적재한다.

Scale source의 전체 untiled view를 먼저 동일 wire order의 `DownScaleBlock=120`으로 rename한 뒤 row tile한다. Backend가 이미 tiled view를 reshape하지 못하므로 이 순서가 필요하다. 기존 주소 `120*(20*tile+row)+block`은 rename→tile 후에도 같다. `row=1920*cluster+60*group+20*tile+local`, `L=1920*column+16*block+offset` 전수검사는 H3840/L15360 값을 모두 보존한다. `validate_layout.py`는 이 좌표, 실제 zero tail 모델, 변경하지 않은 up/gate 및 ops, 아래 오차 반례를 검사한다. CPU 검사는 NPU 실행 검증을 대신하지 않는다.

## 수치 변경

각 block의 새 BF16 반올림 때문에 원래 계산과 동치가 아니다. 유한 normal block 결과 `p`에서 round error는 `2^-8*|p|` 이내이며, 최종 BF16 cast 전 추가 오차는 `sum(column,block) |scale|*|BF16(p)-p|`와 변경된 FP32 accumulation 오차로 나눠 볼 수 있다. Overflow/subnormal 및 최종 nonlinear normalization에는 이 normal-value bound를 그대로 적용할 수 없다.

정확한 BF16 입력 `[1,2^-8×15]`와 `[-1,-2^-8×14,0]`, FP4 weights1/FP8 scales1을 두 block에 놓으면 원래 합은 `2^-8`, block BF16 반올림 후 합은 `2^-7`이다. 이 경우 block sum은 FP32에서 정확하므로 차이는 새 BF16 boundary 자체에서 생긴다. 나머지 입력과 column 결과가0이어도 차이는 남는다. Cancellation 아래 상대오차 보장은 없다. 기존 up/gate scale-DPE가 공식 세 설정을 통과했다는 사실도 down의 새 반올림을 보장하지 않는다.

## 정적 schedule

| 구간 | 정적 비용/완료 |
|---|---:|
| 전체 FFN | 170,270 cycles, 217 instructions |
| 기준 전체 FFN | 171,727 cycles, 177 instructions |
| F4→F8 down weight decode | 9,263 cycles |
| F8→BF16 down weight decode | 18,263 cycles |
| 20-row 첫 block dot | 3×2,668 cycles |
| BF16 partial120→128 DMA | 3×2,995 cycles |
| FP8 scale120→128 DMA | 3×2,995 cycles |
| scale F8→BF16 lookup | 3×663 cycles |
| scale BF16 TRF load | 3×903 cycles |
| 두 번째 scale dot + 8-column FP32 reduce | 3×492 cycles |
| compaction 뒤 down output 준비 | 159,577 cycles; 기준160,981 |

위 구간들은 일부 겹치므로 합산해서 전체 시간으로 해석하면 안 된다. BF16 memset3회(각906cycles), FP8 memset3회(각586cycles)가 schedule 초반에 실제 실행된다. 두 번째 DPE 자체는 짧지만 실제120→128 DMA와 staging이 비용 대부분을 차지한다. 전체 정적 개선은0.85%에 그치므로 NPU의 정확성/실측 비용을 고려한 채택 우선순위는 낮다. Decode 완료 시점도 기준148,632에서 후보145,473으로 달라졌으므로 전체 차이를 새 dot 한 구간의 속도 차이로 해석하지 않는다.

row20 기준 slice payload는 compact partial4,800B, BF16 partial128 buffer5,120B, FP8 scale128 buffer2,560B, BF16 scale128 buffer5,120B다. 추가 scale TRF payload5,120B와 기존 activation TRF3,840B도 필요하다. 이는 payload이며 물리 peak allocation 수치가 아니다.

`schedule-summary.json`에 down의 source-correlated instructions를 보존했다. 전체 JSON은 `/tmp/moa-ffn-down-scale-final-schedule.json`, Docker scratch의 `schedules/down-scale.json`에 있으며 candidate tree에는 큰 raw schedule을 넣지 않았다. SHA256은 `evidence.json`에 있다.

## 제한된 수정 이력

1. scale120을 `8#16` packet으로 재배치한 첫 source는 collect mapping 불일치로 거절됐다. `mlp.attempt1.rs` 및 첫 log에 보존했다.
2. 실제 zero-initialized FP8 scale128을 Sub lookup으로 바로 TRF에 넣으려던 source는 Rust API에서 거절됐다. SDK0.8.1의 lookup은 Main Fetch Unit 전용이다. `mlp.attempt2.rs` 및 두 번째 log에 보존했다.
3. Main lookup→BF16 DM→Sub TRF로 수정했으나 `tile(...).reshape()`는 backend가 거절했다. `mlp.attempt3.rs` 및 세 번째 log에 보존했다.
4. 전체 untiled scale view의 axis rename을 tile 앞에 옮긴 현재 source는 exact compile에 성공했다. 최종 log는 `compile.log`다. 추가 변형/compile/NPU 실험은 하지 않았다.

## 재현과 동결

기준 후보의 `ops.rs` 및 `residual.rs`와 함께 사용한다. `ops.rs`는 여기 복사본과 기준이 동일하고, 필요한 residual SHA256은 `83d001788e3c1ae073ec8720435569fde062a47e52c181ab859a2d3f047accfc`다. `candidate.patch`는 기준 `ffn-scale-contraction-streamed/mlp.rs`에 대한 diff다.

```sh
python3 moa-dev/experiments/candidates/ffn-down-scale-contraction/validate_layout.py
./moa-dev/moa bash -c 'set -o pipefail; cd /tmp/moa-ffn-down-scale-contraction; export CARGO_TARGET_DIR=/tmp/moa-ffn-down-scale-contraction/target; export FURIOSA_OPT_OUT_DIR=/tmp/moa-ffn-down-scale-contraction/target/furiosa-opt/kernel; cargo furiosa-opt compile ops::decoder_feedforward --exact --dump-schedule schedules/down-scale.json 2>&1 | tee logs/attempt-4.log'
```

`evidence.json`에 source/compiled binary/schedule hashes, 실패 이력, 좌표 검사, 수치 반례, 정적 수치를 기록했다. Live source, Git, NPU를 변경하지 않고 후보를 동결했다.
