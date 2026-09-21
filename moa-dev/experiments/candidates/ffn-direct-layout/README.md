# FFN down: column 분할 배치에서 직접 dequant

상태: 두 후보 모두 독립 `cargo-furiosa-opt 0.8.1` 컴파일 성공. **최종 전체 load 후보는 Arena job 69992에서 FFN 정확성 실패**했다. 현재 채택하면 안 된다. 적용할 최종 후보는 `mlp.rs`이며 `mlp.batched.rs`와 같은 내용이다. live source, 공유 target, Git, Arena에는 변경하지 않았다.

## 변경

이전 two-row 후보에 이어 down weight의 HBM→DM load, FP4→FP8 LUT, local scale 적용을 처음부터 `DownRowsByColumns=m![H/120,L/1920]` 배치에서 실행한다. 기존은 `DownRows=m![H/120,1#8]`에서 전체 L=15360을 dequant한 뒤 8개 column slice로 재배치했다. 새 배치는 각 slice에서 L의 1920개 값만 처리하므로 재배치 자체와 변환의 직렬 길이를 함께 줄인다.

최종 후보는 packed down 행렬 전체를 한 번 load하고 한 번 LUT 변환한다. 이후 2-row views만 잘라 local scale·contraction을 수행한다. 이는 기존 up/gate가 이미 사용하는 전체 FP4 load/LUT 패턴을 down의 column 분할 배치에 적용한 것이다. up/gate·GeGLU·global scale 부분은 byte 단위로 기존과 같다.

두 단계가 각각 컴파일되므로 효과를 분리해 볼 수 있다.

| 구현 | Makespan | 기준 대비 정적 비율 |
| --- | ---: | ---: |
| 공식 원본 | 1918472 | 1.0000× |
| 이전 two-row | 1474463 | 1.3011× |
| 직접 column 배치, 60회 tile load/LUT (`mlp.direct-tiled.rs`) | 710106 | 2.7017× |
| 최종 직접 column 배치, 1회 전체 load/LUT (`mlp.rs`) | 616787 | 3.1104× |

최종 후보는 two-row 대비 정적 makespan **58.17% 감소, 2.3906×**다. 하드웨어 측정값이나 공식 점수 개선율을 의미하지 않는다.

## 논리적 데이터 순서

새 slice mapping은 output row group 32개와 input column group 8개의 곱이다. 각 slice의 row 범위는 120개, column 범위는 1920개이다.

```text
global_row    = row_group * 120 + 4 * i + 2 * j + local_row
global_column = column_group * 1920 + local_column
scale_column  = global_column // 16
              = column_group * 120 + local_column // 16
```

`i=0..29`, `j=0..1`, `local_row=0..1`, `row_group=0..31`이 3840 output rows를 정확히 한 번씩 덮는다. `column_group=0..7`, `local_column=0..1919`가 15360 reduction positions를 정확히 덮는다. 1920은 16과 2의 배수라 local-scale group과 packed FP4 nibble pair를 분할 경계에서 잘라먹지 않는다.

위 row·column 전체 인덱스, scale-group identity, FP4 byte-index identity를 Python 정수 계산으로 확인한 기록이 `logical-index-proof.json`이다. 이는 논리적 mapping 검증이며 compiler lowering와 실제 NPU 결과를 대신하지 않는다. 데이터 배치 변경은 typed DMA가 수행하며 새 `unsafe reshape`는 추가하지 않았다.

FP4 값 → FP8 LUT → F32 local-scale multiply → BF16 dequant → BF16 activation contraction → 8-way F32 reduction → BF16 결과라는 순서는 그대로다. two-row 후보의 최소 8-byte commit을 위한 padding 및 작은 output copy도 유지했다. global scale·post-RMSNorm·residual·layer scalar 경계는 변경하지 않는다.

## 정적 근거

| 항목 | two-row | 직접 배치 tile load | 최종 전체 load |
| --- | ---: | ---: | ---: |
| DmaEngine union | 606494 | 372087 | 290287 |
| MainContext union | 880611 | 351411 | 335894 |
| SubContext union | 129898 | 117298 | 41011 |
| VectorEngine union | 684060 | 280860 | 280860 |
| instruction 수 | 1622 | 1562 | 1326 |

주요 down 노드:

- BF16 tile redistribution: **256500 → 0 cycles**.
- local scale dequant: **477660 → 74460 cycles**. column을 처음부터 8개 slice에 나눈 효과다.
- FP4 LUT: **159780 → 33780 → 18263 cycles**.
- packed weight DMA: **81000 → 90660 → 58302 cycles**. 직접 배치의 작은 tile DMA는 잠시 악화하지만 전체 load가 회수를 줄인다.
- scale DMA: **6565 → 18998 cycles**. 새 배치의 scale access는 더 비싸졌으며 최종 후보에도 남는 비용이다.
- down contraction/reduction: two-row와 최종 모두 **30720 cycles**.
- 작은 2-row output copy: 모두 **27060 cycles**.

최종 전체 load/LUT는 반복되는 compiler-generated table setup도 줄여 SubContext 합을 낮춘다. Context들은 서로 겹치므로 위 숫자를 합산하여 makespan으로 사용하지 않는다. 소스 줄별 합, 파일 SHA256, 원본/two-row/두 직접 배치의 수치는 `static-comparison.json`에 있다.

## 적용과 다음 확인

`change-from-two-row.patch`는 이전 two-row 파일에, `change-from-baseline.patch`는 공식 원본 파일에 적용한다. 두 patch를 각각 임시 snapshot에 적용해 최종 `mlp.rs`와 byte 단위로 동일함을 확인했다.

동시에 많이 살아 있는 FP4/FP8 weight tensor가 커졌다는 점은 tile load 버전과의 차이다. 최종 정적 compiler는 메모리 계획을 성공적으로 생성했지만 최종 후보의 hardware 정확성은 실패했으며 원인을 조사 중이다. 정확성에 실패했으므로 측정된 실행 시간을 성능 성과로 인정하지 않는다. 공식 harness에서 세 커널 모두 통과 여부와 FFN 3-run cycles를 확인해야 한다. 실패 시 먼저 `mlp.direct-tiled.rs`를 검사하면 mapping 문제와 전체 load/liveness 문제를 나눌 수 있다.

독립 scratch는 `/tmp/moa-ffn-direct-layout`, target은 `/tmp/moa-ffn-direct-layout-target`이다. 최종 컴파일 명령:

```bash
docker exec \
  --workdir /tmp/moa-ffn-direct-layout \
  --env CARGO_TARGET_DIR=/tmp/moa-ffn-direct-layout-target \
  moa2026-dev \
  cargo furiosa-opt compile ops::decoder_feedforward --exact \
  --dump-schedule /tmp/moa-ffn-direct-layout/decoder_feedforward-batched.json
```

Raw 결과는 Git에서 제외되는 `../../artifacts/ffn-direct-layout/`에 있다.

- [최종 schedule](../../artifacts/ffn-direct-layout/decoder_feedforward-batched.json)
- [최종 컴파일 로그](../../artifacts/ffn-direct-layout/compile-batched.log)
- [첫 직접 배치 schedule](../../artifacts/ffn-direct-layout/decoder_feedforward.json)
- [첫 직접 배치 컴파일 로그](../../artifacts/ffn-direct-layout/compile.log)

이 구현 agent는 첫 직접 배치와 전체 load/LUT 후보를 각각 한 번 컴파일했고 NPU job은 실행하지 않았다. 이후 root가 전체 load 후보를 job 69992에서 검사했다. QKV/O는 원본으로 PASS, FFN은 세 입력 모두 FAIL: max error 0.67969/1.12500/0.92969, within tolerance 44.22%/20.81%/17.66%, 중앙값 1249479 cycles. 이 사이클은 부정확한 출력의 기록이며 채택 가능한 speedup이 아니다. Root는 two-row 후보와 직접 배치 변경을 분리 검증 중이다.
