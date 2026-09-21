# Native FP8 block dot → BF16 partial → scale contraction

**후속 E25 job70354에서 첫 FFN 실행이 timeout으로 종료했다(exit101).** Q/O는 모든 설정을 통과했으나 FFN output과 cycle은 하나도 얻지 못했다. 이 후보는 채택하지 않는다. [원시 로그](../../../logs/e25-native-fp8-scale-dpe-job-70354.log). 다른 F32-block/VE-reduce 경로의 E22도 첫 FFN launch에서 같은 timeout을 보였지만, 공통 경로 중 어느 instruction이 멈췄는지는 아직 확인되지 않았다. 소스는 실패 상태로 동결한다.

2026-09-21, compiler/std 0.8.1. `ffn-scale-contraction-streamed`의 up/gate 첫 BF16 contraction을 native FP8 hi/lo two-lane contraction으로 바꿨다. **32-row exact compile PASS: 166,220 cycles**, 같은 E16 epilogue의 streamed BF16 기준 171,727보다 5,507 cycles(3.21%) 적다. 16-row 변형도 compile PASS였으나 167,035 cycles로 더 느렸다. 최종 파일은 32-row `mlp.rs`다. **후속 NPU 실행은 위 timeout 기록처럼 완료되지 않았다**.

두 first-party 후보의 독립 구현을 결합했다. `ffn-scale-contraction-streamed/mlp.rs`의 두 번째 BF16 scale contraction과 GeGLU/down/tail을 유지하고, `ffn-native-fp8-upgate/mlp.postreduce-div.rs`의 runtime quantizer와 four-block native FP8/pair 매핑을 사용했다. 외부 competition source는 읽거나 복사하지 않았다.

## 호출과 데이터 흐름

`ops::decoder_feedforward`는 HBM residual을 BF16 DM으로 읽고 pre-FF RMSNorm을 수행한 뒤 `shared::mlp::feedforward`에 넘긴다. 새 `feedforward`는 실제 정규화 입력에서 runtime scale `s`와 FP8 hi/lo를 만든다. 두 clusters의 512 physical slices에 같은 입력을 복제하고, 기존 UpGateSlices의 마지막 16 padding slices를 제외한 240개 slices에 32 rows씩 배치한다. H3840와 L15360의 모든 유효 좌표를 사용한다.

`FfnTerm=2`는 cluster 축 `Dummy2`와 별개다. 각 slice의 activation TRF는 Lane=`FfnTerm`, Element=`H`이며 FP8 payload 7,680B다. FP4 weight는 기존 lookup으로 FP8 DM까지만 확장한다. 큰 BF16 weight DM은 만들지 않는다.

첫 contraction은 FP8 packet 64개 원소를 처리하면서 `contract_packet<H/16%4>`로 16원소 dot 네 개를 분리한다. `contract_time<L%32,H/64>`는 60 packet 좌표를 그대로 유지한다. `contract_lane(Sequential)`의 Time=`L%32,H/64,FfnTerm`, Packet=`H/16%4#8`에서 unzip으로 두 항을 분리하고 다음 순서로 계산한다.

```text
block16 hi-dot + block16 lo-dot / 16
  → runtime s 곱하기
  → BF16로 반올림
  → DM [row32, block240]
  → 원래 FP8 local scale을 BF16 lookup하여 TRF에 적재
  → BF16 partial × BF16 scale contraction
  → 240 blocks 합산
  → 기존 BF16 up/gate 결과
  → 원래 global scales, GeGLU, down
```

Pair Mul0는 lo에만 1/16을 곱하고 AddF zip 뒤 Mul1이 `s`를 적용한다. `s*hi`와 `s*lo`를 각각 먼저 계산하지 않는다. quantizer의 두 번째 반환값은 **s 자체**다. 원본 `mlp.postreduce-div.rs`의 `1/s` 반환을 그대로 쓰지 않았으며 reciprocal 생성 pass도 제거했다.

BF16 cast 이후 packet의 유효 block 값 네 개가 8B이므로 `commit_trim<H/16%4>`로 바로 기록할 수 있다. scalar packet을 네 개씩 transpose할 필요가 없다. 첫 stage의 local scale VRF와 별도 F32 block DM/VE reduction이 사라져 32-row 전체 tile을 처리한다. 두 번째 contraction은 기준 streamed 후보와 같은 BF16 경로다.

## 정확성의 범위

이 변경은 임의 입력에 대해 기존 계산과 같지 않다. `b_j = BF16(s*(dot(q_j,hi_j)+dot(q_j,lo_j)/16))`를 만든 뒤 `BF16(sum_j local_scale_j*b_j)`를 계산한다. 원래 FP32 block dot에 없던 **각 block의 BF16 반올림**과 기존 [FP8 표현 오차](../qkv-fp8-split/README.md)가 모두 들어간다. FP4 weight와 FP8 local scale을 BF16로 표현할 수 있다는 사실은 activation과 16항을 곱해 더한 block dot까지 BF16-exact라는 뜻이 아니다.

`check_math.py`는 3,840 input positions와 240 block 경계의 전수 대응을 확인하고, 독립 CPU nearest-even codec으로 오류를 계산한다. RNGD의 FP32 reducer 순서나 cast 동작을 흉내 냈다고 주장하지 않는다. fixture를 읽지 않고 무작위 BF16 입력/FP4 weights/FP8 scales 32행을 세 크기로 확인했다. 최종 BF16 결과는 기준과 각각 9/32, 19/32, 15/32행에서 달랐다. 이 mismatch 자체는 공식 tolerance 실패를 뜻하지 않지만 exactness 주장도 불가능하다.

입력이 정확히 FP8 두 항으로 복원되는 반례도 있다. 첫 block의 BF16 입력을 `[1, 2^-8 ×15]`, 두 번째를 `[-1, -2^-8 ×14, 0]`, 다른 값을 0으로 놓고 FP4 weight와 FP8 local scale을 모두 1로 놓는다. 정확한 두 block 합은 `2^-8`이지만 BF16 block 반올림 뒤 합은 `2^-7`이다. CPU 검사가 이 차이를 확인했다. Cancellation 아래 상대오차 상한은 없다. block 중간 BF16 overflow/underflow, 실제 F32 누산, global scales 및 GeGLU를 포함한 전체 정확성은 oracle 검증이 필요하다.

기준 streamed BF16 scale-DPE는 후속 E23 job70327에서 FFN 세 설정을 모두 통과했고 median 384,728 cycles였다. 이는 새 native FP8 후보의 검증을 대신하지 않는다. 다른 native FP8 F32-block/VE-reduce 후보는 E22의 첫 FFN 실행에서 timeout했고, QKV two-lane에는 별도 V 오류가 보고되어 있다. 여기서는 다른 partial/reduction 경로를 사용하지만 compiler 성공만으로 hardware 동작을 보장하지 않는다.

## 정적 비교와 재현

| 경로 | 전체 static | 첫 dot/partial | 두 번째 scale dot |
| --- | ---: | ---: | ---: |
| streamed BF16, 32 rows | 171,727 | up/gate 각각 19,463 | 각각 745 |
| native FP8, 32 rows | **166,220** | up/gate 각각 7,964 | 각각 745 |
| native FP8, 16 rows | 167,035 | matrix당 4,124 ×2 | matrix당 505 ×2 및 짧은 tile commit |

개별 stage 합계와 makespan은 같지 않다. native 후보에는 runtime quantization과 packed input broadcast가 추가되고 DMA/context overlap도 바뀐다. 실제 성능 개선 주장은 하지 않는다.

scratch는 `/tmp/moa-ffn-native-fp8-scale-contraction`이다. `ffn-scale-streamed`의 source/manifests를 복사했고 `ops.rs` 및 shared dependencies는 그대로 두었다. 최종 32-row source로 복구한 뒤 exact compile을 다시 완료했다.

```sh
./moa-dev/moa bash -c 'set -o pipefail; cd /tmp/moa-ffn-native-fp8-scale-contraction; export CARGO_TARGET_DIR=/tmp/moa-ffn-native-fp8-scale-contraction/target; export FURIOSA_OPT_OUT_DIR=/tmp/moa-ffn-native-fp8-scale-contraction/target/furiosa-opt/kernel; cargo furiosa-opt compile ops::decoder_feedforward --exact --dump-schedule schedules/rows32-frozen.json 2>&1 | tee logs/rows32-frozen.log'
```

`evidence.json`에 source/schedule hashes와 stage 요약을 기록했다. `candidate.patch`는 streamed BF16 기준 diff, `mlp.rows16.rs`는 비교 변형, `mlp.baseline.rs`는 기준 snapshot이다. `math-evidence.json`은 Docker Python/NumPy에서 실행한 CPU 결과다. source는 동결하며 live source, Git, NPU 제출은 하지 않았다.
