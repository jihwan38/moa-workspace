# RNGD Gemma 최적화 실행 계획

목표는 MOA 공식 평가에서 상위 3위에 진입하는 것이다. 현재 `baseline` 브랜치에는
실제 RNGD에서 반복·추가 입력 검증을 마친 E47을 채택했다. 공식 제출과 순위 검증은
아직 완료되지 않았다. 조사·측정 기준일은 2026-09-21이다.

## 현재 결과와 검증 범위

원본과 E47 각각 같은 업로드 패키지를 독립 Arena job 3개에서 실행했다.
아래 값은 각 job의 입력 구성 3개 중앙값을 구한 뒤, job 중앙값 3개의 중앙값을 취한 것이다.

| 커널 | 원본 cycles | E47 cycles | 원본 대비 속도 | E47 job 중앙값 상대 범위 |
|---|---:|---:|---:|---:|
| QKV projection | 244,051 | 94,662 | 2.578× | 0.153% |
| Attention output projection | 404,222 | 44,588 | 9.066× | 0.794% |
| Decoder FFN | 4,264,297 | 377,564 | 11.294× | 0.088% |

- 원본 jobs: 69783/69913/69931. [기준 기록](experiments/baseline-rngd.json).
- E47 jobs: 70664/70681/70688. 세 job 모두 공식 harness의 15개 출력 비교를 통과했고,
  실제 업로드한 실행 파일·fixture·entrypoint·manifest 4개 파일의 SHA256이 동일하다.
  [반복 기록](experiments/e47-output-sixteen-columns-repeats.json).
- 별도 개발 harness의 run3..8, 입력 구성 6개·출력 비교 30개도 job70694에서 통과했다.
  공식 구성 3개를 합하면 서로 다른 구성은 9개다. 반복 job을 별도 입력으로 세지 않는다.
  [추가 입력 검사](experiments/e47-output-sixteen-columns-extra-correctness.json).
- E47은 E37에서 O projection만 16분할로 바꿨다. QKV/FFN device image는 동일하며,
  새 패키지에서 관측한 두 커널의 작은 차이를 해당 커널의 소스 개선으로 해석하지 않는다.
  O 반복 중앙값은45,748→44,588(−2.54%)이다.
- 같은 QKV image를 검사한 E37 sentinel job70589에서 출력15개와 cache 검사12개를 통과했고,
  세 슬롯 이외의 BF16 원소12,570,624개가 bit-exact로 보존됐다.
  [Cache 검사](experiments/e37-cache-preservation-correctness.json). 진단 cycles는 성능표에 섞지 않는다.
- E47 소스 복구 뒤 전체 빌드가 실측 실행 파일과 byte-identical했다. 별도 추가 입력 빌드도
  세 device image가 모두 일치했다. [소스·빌드 검사](experiments/e47-source-build-audit.json).
  E37의 빈 target 빌드 검사는 [이전 기록](experiments/e37-clean-build-audit.md)에 보존한다.

이 값은 세 커널의 개발용 NPU cycles이며, 전체 Gemma 추론 처리량이나 공식 순위가 아니다.
기존 공개 기준 cycles로 계산한 참고 기하평균은 약 6.176이다. 9월 21일 이전 공개 3위의
약 8.399와 같은 평가 환경이라고 확인되지 않았으므로, 이를 현재 순위로 환산하지 않는다.
공식 점수의 기하평균에서 각 커널의 상대 개선 비중은 같다. FFN의 절대 cycle 수가
가장 크다는 이유만으로 QKV/O 개선을 무시해서는 안 된다.

공식 사이트는 평가 기준 수정으로 제출을 중단했고 Arena 개발은 계속 가능하다고 안내한다.
별도 `moa-submitter` 인증도 미완료다. 공개 리더보드는 참고용이며 최종 순위는 동일 조건의
재실행으로 결정된다. [공식 공지](https://micro2026-moa.github.io/leaderboard.html),
[상태 기록](experiments/official-submission-status-20260921.json).

## 채택한 구현과 데이터 흐름

실제 진입점은 [src/ops.rs](../furiosa-opt-gemma4-12B/src/ops.rs)의
`sliding_project_qkv`, `sliding_attention_output`, `decoder_feedforward`다.
공식 입력 생성·비교 harness와 device 함수 signature는 유지한다.
[E37 소스 계약 검사](experiments/e37-source-contract-audit.json)에서 source43개 중 허용 경로7개만
변경됐고, 공개10개 함수의 본문 바깥 텍스트와 공식 harness가 원본과 같음을 확인했다.
[E47 검사](experiments/e47-source-build-audit.json)에서는 private O 코드만 추가 변경됐고 나머지는 E37과 같다.

**QKV:** HBM 입력 → H480×8 배치의 RMS → FP8 hi/lo 분해 → Q/K/V projection →
head RMS → Q/K RoPE → Q 출력과 지정 K/V cache 슬롯 저장.
[projection.rs](../furiosa-opt-gemma4-12B/src/device/sliding/projection.rs)의
`prepare_fp8_split`은 정규화된 입력을 두 FP8 성분과 scale로 만든다.
Q/K는 두 lane의 native FP8 연산을 쓰고, V는 검증된 두 개의 독립 dot을 쓴다.
8-way 행 interleave와 두 cluster를 사용한다. 입력 RMS는 E29 경로를 유지한다.
Q에도 Switch-gather RMS를 적용한 E31은 반복 측정에서 더 느려 E37에서 되돌렸다.

**Attention output:** attention 결과 → 256열씩 16개 부분 projection → FP32 부분합 →
channel scale → RMS → residual → 원래 HBM residual 전체 갱신.
`project_output`은 native FP8 hi/lo 연산 후 별도 FP32 DM에서 열 부분합을 합친다.
후단을 H480×8로 유지해 gather/scatter를 줄인다. 여기에는 attention softmax 계산이 없다.

**FFN:** residual → pre-RMS → FP4 up/gate → GeGLU → FP4 down → global scale →
post-RMS → residual 및 layer scale → HBM 저장.
[mlp.rs](../furiosa-opt-gemma4-12B/src/device/shared/mlp.rs)의 `feedforward_sharded`는
FP4→FP8→unscaled BF16 가중치를 matrix별로 한 번 준비해 row tile에서 재사용한다.
16원소 FP32 dot마다 해당 local scale을 적용하고 합한다. local scale은 reduction 축을
따라 달라지므로 전체 GEMV 뒤 한 번 곱하는 것으로 바꿀 수 없다. Down 이후 H480×8을
유지하는 E36 후단을 결합했으며, 기존 shared helper 호출자를 위한 wrapper는 보존한다.

FP8 입력 분해와 합산 순서 변경은 모든 유한 입력에서 원본과 동일한 연산이 아니다.
실제 outlier/cancellation 반례를 [독립 수치 검토](../moa-dev/experiments/candidates/output-native-fp8-two-lane/review.md)에 보존한다.
현재 PASS 범위는 명시한 9개 입력 구성이다. 새 평가 입력·모델 경로에도 통과한다고 가정하지 않는다.

## 남은 병목과 다음 실험의 조건

1. **새 공식 입력·채점 조건을 먼저 확인한다.** 서버 복구 뒤 허용 수정 범위와 입력 계약을
   다시 대조하고 채택 E47을 공식 평가한다. 허용오차를 벗어나면 실패 출력과 중간 값을 분리해
   더 정확한 projection 경로를 선택한다. fixture 값·seed·특정 출력 좌표에 맞춘 분기는 사용하지 않는다.
2. **FFN은 복호화와 DMA를 함께 줄여야 한다.** 현재 static schedule 162,209 cycles에서
   FP4→FP8→BF16 6개 변환의 Main 작업량은 86,178, DMA 합집합은 135,607이다.
   두 자원이 상당 부분 겹치므로 비용을 단순 합산하지 않는다. 현재 DMA 순서·시간을
   그대로 두면 decode만 공짜로 만들어도 이 정적 모델의 최대 개선은 약 1.196배다.
   다음 후보는 실제 weight/scale 전송, materialization, activation 복제를 줄이는 경로가
   확인될 때 시작한다. [현재 병목 분석](../moa-dev/experiments/candidates/ffn-parallel-epilogue/remaining-bottlenecks.md).
3. **Native FP8 FFN은 가능하지만 현재 더 느리다.** E35 full256 경로는 처음 정확성을
   통과했으나 811,219 cycles였다. Direct write E38은 482,511, tail-only 초기화 E39는
   415,457로 개선됐지만 E37의 379,741보다 느리다. E41 실제30행/row2도 정확성을 통과했으나
   426,442였다. 실제30행과 기존 row256 배치를 결합한 E42도 정확성을 통과했지만
   456,849로 느려 채택하지 않았다.
   서로 다른 정적 예측과 실제 cycles의 차이를 특정 compiler 결함으로 단정하지 않는다.
4. **배치 변경은 논리 좌표와 실제 전송을 함께 검증한다.** E40 row2+tail은 CPU 좌표 검사와
   compile을 통과했지만 NPU의 FFN 세 입력 모두 non-finite였다. 반면 E41 row2는 통과했다.
   물리 dump에서도 예상한 60 BF16/120B 출력과 같은 주소의 alias를 확인했으므로
   row2 자체나 숨은64행 padding을 원인으로 확정하지 않는다.
   [실패 원인 범위](../moa-dev/experiments/candidates/ffn-native-fp8-row2-tail-zero/physical-layout-audit.md).
5. **QKV/O의 작은 개선도 독립 측정한다.** O row2는 전체 정확성 통과에도 기존보다 4.31%
   느려 제외했다. QKV amax Switch-gather는 local 연산을 줄였지만 전체 schedule이 늘었다.
   O의 FP32 부분합 compact 저장은 현 SDK stream/packet 제약에서 실패했다.
   재시도는 실제 지원 API와 줄어드는 전체 전송 경로를 확보한 경우로 제한한다.

E22/E25/E28/E32 timeout, E33 down scale-DPE의 큰 오차, E40 non-finite 등 실패 기록도
[실험 원장](experiments/README.md)에 남긴다. 컴파일 성공·논리 좌표 증명·정적 예상 cycles만으로
실제 정확성이나 개선을 선언하지 않는다. 느리거나 실패한 prototype은 채택 소스와 분리한다.

## 조사 자료에서 실제로 가져온 원칙

- [MOA 규칙·평가 계약](research/moa-rules-and-target.md): 수정 범위, 모든 원소 비교,
  stage별 metric을 우선한다. Stage1의 입력·weight 자체를 재양자화하거나 host로 계산을 옮기지 않는다.
- [RNGD/SDK 문서](research/rngd-furiosa-opt.md): cluster/slice, DM/TRF, stream packet,
  TDMA/Switch의 실제 제약에 맞춰 consumer 배치를 설계한다. Padding은 초기화된 0이 아니다.
- [실제 구현 분석](research/kernel-analysis.md): caller→projection→norm→scatter/write 경로와
  scale 축·BF16 경계·cache 부수효과를 함께 추적한다.
- [국내외 대회·논문 조사](research/competition-techniques.md): AMD FP8/MI300 수상 구현의
  layout/split-K, FlashInfer 수상 보고서의 workload별 검증, 국내 KCC·LG Aimers의 단계별
  제출 계약, FlashAttention의 중간 저장 제거 관점을 검토했다. GPU instruction, 완화된 오차,
  재양자화 허용 범위를 RNGD Stage1에 그대로 적용하지 않는다.

## 재현 및 채택 절차

1. 소스·compiler·Cargo.lock·fixture·binary·schedule hash를 고정한다.
2. 변경 커널을 exact compile해 layout, 용량, DMA/Main 중첩과 전체 schedule을 확인한다.
3. 공식 harness를 그대로 빌드해 Arena에서 입력 3개와 출력 비교 15개를 모두 통과시킨다.
4. 최종 결합 패키지는 같은 업로드 파일로 독립 job 3개를 실행한다. 각 job 중앙값과 범위를
   보관하며, 다른 패키지에서 좋은 커널 결과만 골라 합치지 않는다.
5. 정밀도·합산·배치가 바뀌면 별도 개발 harness로 추가 입력을 검사한다. Cache 보존 같은
   부수효과 검사는 해당 device image와 검사한 슬롯 범위를 명시한다.
6. 채택 소스에서 다시 빌드하고 diff·device signature·공식 harness 불변을 확인한 뒤
   공식 평가 결과와 현재 순위를 기록한다. 이 마지막 단계 전에는 상위3위 달성을 선언하지 않는다.

```bash
./moa-dev/moa cargo furiosa-opt build --release --bin test_kernels
./moa-dev/moa env RNGD_SUBMIT_TIMEOUT=70 ./scripts/rngd_test.sh --no-build --no-wait
python3 moa-dev/measure.py --jobs JOB_ID --reruns 2 \
  --label candidate-name --output docs/experiments/candidate-name.json \
  --provenance PATH_TO_PROVENANCE.json \
  --leaderboard-snapshot docs/experiments/leaderboard-20260921-initial.json
```

Arena 상한은 작업당70초이고 동시 queued/running 작업은 최대2개다. `capture.py`는 관찰한
소스와 바이너리를 각각 저장하므로, 빌드 성공→즉시 캡처→동결 패키지 제출 순서를 지킨다.
Collector는 실제 업로드 archive를 내려받아 SHA를 확인한다. 원시 로그·대용량 artifact는
Git 제외 경로, source patch·결과·해시는 저장소에 보관한다.

## 2라운드로 이어갈 범위

공식 workload가 확정되면 [src/host/runtime.rs](../furiosa-opt-gemma4-12B/src/host/runtime.rs)의
실제 모델 호출과 KV 상태를 기준으로 초기화, prefill TTFT, decode TPOT, host-device 동기화,
NPU 시간, 메모리를 나누어 측정한다. Shared helper가 영향을 주는 full attention·vision·audio도
다시 검사한다. Stage1 세 커널 PASS를 전체 모델의 정확성·처리량 보장으로 확장하지 않는다.


## E37 이후의 후속 검사

E47의 반복·추가 입력 검사가 끝나 O16×256 경로를 채택했다. FFN은 검증된 E37 구현을 유지한다.
실험용 소스·바이너리와 채택 패키지를 구분하며, 다음 표의 단일 screen을 반복 결과와 합치지 않는다.

| 후속 실험 | 확인된 결과 | 판단 |
|---|---|---|
| [E44 paired V](experiments/e44-paired-value.json), job70625 | 공식15개 PASS, QKV95,892 | E37 반복 중앙값94,195보다1.80% 느려 미채택. Paired V 자체가 불가능하다는 결론은 지지하지 않음 |
| [E45 down scale staging](experiments/e45-down-scale-coalesced.json), job70641 | 공식15개 PASS, FFN394,044 | E37보다3.77% 느려 미채택. Static -0.43%였지만 전체 DMA는1,898 증가했으며, 실측 회귀의 완전한 원인은 미확정 |
| [E46 native FP8 F32 block reduce](experiments/e46-native-f32-block-reduce.json), job70652 | 공식15개 PASS, FFN434,284 | E37보다14.36%, 직접 부모 E41보다1.84% 느려 미채택. Block당 BF16 반올림 제거가 성능 개선을 보장하지 않음 |
| [E47 O16×256](experiments/e47-output-sixteen-columns-repeats.json), jobs70664/70681/70688 | 세 job 모두15개 PASS, 반복 O44,588; 추가30개 PASS70694 | E37 O보다2.54% 개선하여 채택 |
| [E48 up-only fused decode](experiments/e48-up-only-fused-decode.json), job70683 | 첫 screen 공식15개 PASS, FFN381,532 | E37 반복 중앙값보다0.47% 높음. Up decode만 fused로 바꾼 전체 FFN의 세 공식 입력 실행은 확인했으나, 추가 입력·반복 성능 검증과 채택은 별도 |

E44/E45/E46/E48은 각 단일 job의 공식 입력3개 중앙값이고 E47은 같은 패키지3회 반복 결과다.
E48의 QKV/O 소스는 E37과 같으며 E47 O 변경을 섞지 않았다. E48은 up만 fused로 바꾸고
기존 gate/down을 유지했지만 성능 개선을 입증하지 못해 채택하지 않았다.

FP4 복호화의 [작은 독립 진단](experiments/fp4-decode-probe.json), job70620은 한 cluster,
256개 유효 slice, local32×512에서 legacy 두 단계와 fused 한 pipeline을 비교했다.
세 패턴·두 경로의 BF16 값25,165,824개가 bit-exact였다. 후속
[tagged fullshape 진단](experiments/fp4-decode-probe-fullshape-tagged.json), job70658은
E37 up/gate와 같은 두 cluster, 각각240개 유효/256개 physical slice, local32×3840을 사용했다.
15,360개 행과480개 cluster/slice별로 구별되는 입력을 넣었고, 두 경로 합117,964,800개
BF16 값에서 mismatch가0이었다. 두 진단은 decode만 검사하며 DPE, scale, activation quantizer,
전체 FFN의 allocation/lifetime을 포함하지 않는다. 따라서 E13 실패를 API 체인의 일반적
비지원으로 단정할 수 없지만, 전체 FFN의 실패 원인을 규명하거나 fused 경로를 채택한 것은 아니다.

[E43 activation broadcast](../moa-dev/experiments/candidates/ffn-down-activation-broadcast-four/README.md)는
네 compile 실패 뒤 explicit anonymous4+Custom32 조합으로 컴파일됐다. DMA는3,889 줄었지만
Main에4,103이 추가되어162,209→166,181(+2.45%)로 회귀했고 NPU는 보류했다.
[O4×1024](../moa-dev/experiments/candidates/output-fp8-four-columns/README.md)도 컴파일됐으나
24,997→25,873(+3.50%)로 회귀해 NPU 미실행이다. 실제 weight DMA packet은 원본·후보 모두256B이며,
논리 column 폭 증가를 hardware1KB burst로 해석하지 않는다. E47과 함께 column 분할을
비교했지만 유효 연산 자원·quantizer·합산·전송을 포함한 전체 비용으로 판단한다.

E49는 up·gate를 모두 fused로 바꾸자 정적 Main 연산은 줄었지만 pre-RMS와 두 번째 weight
전송이 늦어져 전체189,967 cycles로 회귀했다. [실행 순서 분석](../moa-dev/experiments/candidates/ffn-up-gate-fused-decode-e37/README.md)에
근거해 NPU 없이 제외했다. E50은 down만 fused로 바꿔 job70700에서 공식15개 PASS,
FFN376,184를 얻었다. 차이가 작아 E47 O16과 결합한 E51의 반복·추가 입력 검증으로
판단하며, 그 전까지 채택 FFN은 E37을 유지한다.

O32×128 분할도 exact compile로 비교했으나 E47의24,621→37,344(+51.68%)로 회귀했다.
Weight DMA가13,383→26,218로 늘어 [O32 후보](../moa-dev/experiments/candidates/output-fp8-thirtytwo-columns/README.md)는
NPU 없이 제외했다. O4/O8/O16/O32 비교에서는 검증된 O16을 유지한다.
