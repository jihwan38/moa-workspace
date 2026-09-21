# MOA 2026 최적화 실행 계획

2026-09-21. 목표는 공식 평가에서 최소3위 진입이며, **아직 공식 제출·점수·순위를 확보하지 못했다.** 현재 기준 구현은 **E53**이다. 빠른 E47의 O 경로에서 발견한 상쇄 입력 오류를 수정했고, 실제 RNGD에서 기존9개 입력 구성과 별도 O 경계8개를 통과했다. 이 범위 밖의 모델 전체 정확성·처리량을 보장하는 기록은 아니다.

## 현재 기준과 채택 근거

E53은 E47의 QKV·FFN 및 O16열 분할 배치를 유지하면서, O activation을 BF16으로 보존한다. O의 FP8 hi/lo 근사와 channel scale 이전의 추가 BF16 반올림을 제거했다. [구현·수치 검토](../moa-dev/experiments/candidates/output-bf16-activation-e47/math-review.md).

| 커널 | 원본 반복 중앙값 cycles | E53 반복 중앙값 cycles | 원본 대비 |
|---|---:|---:|---:|
| QKV | 244,051 | 95,621 | 2.55× |
| Attention output | 404,222 | 50,882 | 7.94× |
| FFN | 4,264,297 | 377,880 | 11.28× |

위 E53 수치는 동일 패키지 job70806/70818/70827의 job 중앙값들의 중앙값이다. 세 job 모두15개 비교를 통과했고 job 중앙값의 상대 범위는 QKV2.207%/O1.254%/FFN0.374%였다. [반복 측정](experiments/e53-bf16-output-repeats.json). QKV/FFN image는 E47과 같으므로 이 두 커널의 작은 cycle 차이는 소스 개선·회귀로 해석하지 않는다.

- 공식 입력: 세 반복 job 각각 출력 비교15개 모두 PASS.
- 별도 입력 run3..8: job70814의 출력 비교30개 모두 PASS. 같은 입력의 반복 job을 새 입력으로 세지 않는다. [추가 검사](experiments/e53-bf16-output-extra-correctness.json).
- O 경계: 같은 host·oracle·payload로 원본70799와 E53 70810은8/8 PASS. E47 70785는 상쇄 입력에서3840개 모두 기대0.609375 대신0.5를 출력했다. 세 job의 입력 checksum40개씩과 업로드 해시·device image를 확인했다. [직접 비교](experiments/output-boundary-comparison.md).
- QKV image는 E37/E47과 같다. 동일 image의 sentinel job70589는 비대상 KV BF16원소12,570,624개를 bit-exact로 보존했다. [KV 검사](experiments/e37-cache-preservation-correctness.json).
- 전체 빌드 후 동결했고, 사용하지 않는 두 O helper를 삭제한 소스로 다시 빌드해 실측 host와 byte-identical함을 확인했다. 별도 추가 입력 빌드도 세 device image가 각각 정확히 한 번 일치한다. [빌드 감사](experiments/e53-source-build-audit.json).

E47보다 O 반복 중앙값은14.12% 높지만 확인된 정확성 실패를 해소했다. E47은 실패 범위를 명시한 성능 후보로 보존한다. E53의 정확성 결과를 모든 finite 입력의 수학적 동치로 확대하지 않는다. F32 합산 순서, RMS 계산, 극단값 overflow/FTZ는 별도 한계다. Tiny case는 결과가 atol보다 훨씬 작아 통과만으로 작은 값 보존을 증명하지 못한다.

공식 사이트는 평가 기준 수정으로 제출을 중단했고 Arena 개발은 계속 가능하다고 안내한다. 별도 `moa-submitter` 인증도 미완료다. 리더보드는 참고용이며 최종 순위는 동일 조건의 재실행으로 결정된다. [공식 공지](https://micro2026-moa.github.io/leaderboard.html), [상태 기록](experiments/official-submission-status-20260921.json). 과거 공개 기준에서 계산한 참고 점수를 현재 순위로 환산하지 않는다.

## 구현의 실제 실행 경로

진입점은 [src/ops.rs](../furiosa-opt-gemma4-12B/src/ops.rs)의 `sliding_project_qkv`, `sliding_attention_output`, `decoder_feedforward`다. 공식 입력 생성·비교 harness와 공개 device signature는 유지했다. [E53 계약 검사](experiments/e53-source-contract-audit.json)에서 source43개 중 허용된7개 경로만 변경됐고 공개10개 signature와 공식 harness6개 파일은 원본과 같음을 확인했다. E37의 [소스 계약 감사](experiments/e37-source-contract-audit.json) 이후 E47/E53은 private O 구현만 바꿨다.

**QKV:** HBM x → H480×8 pre-RMS → 전체 H의 amax 기반 FP8 hi/lo 분해 → Q/K paired dot 및 V의 독립 두 dot → head RMS → Q/K RoPE → Q 출력과 지정 KV cache 슬롯 write. [projection.rs](../furiosa-opt-gemma4-12B/src/device/sliding/projection.rs)의 `prepare_fp8_split`은 저장이480원소 shard여도 scale은 전체3840원소의 amax다. 두 cluster와8-way row interleave를 쓴다. 현재9개 구성은 통과했지만 FP8 분해가 임의 BF16 activation을 정확히 표현하지는 않는다.

**O:** HBM BF16 attention 결과 → 16개256-column shard의 BF16 TRF → FP8 weight를 정확히 BF16으로 lookup한 단일 dot → F32 column reduction → F32 H480×8 gather → F32 channel scale → 첫 BF16 cast → RMS → residual → HBM 갱신. `project_output`과 `apply_output_channel_scale_f32`가 이 경계를 구현한다. 여기에는 attention softmax 계산이 없다. 원본 Python reference와 scale/cast 순서는 맞지만 F32 reduction grouping은 달라질 수 있다.

**FFN:** residual → pre-RMS → FP4 up/gate → GeGLU → FP4 down → global scale → post-RMS → residual 및 layer scale → HBM write. [mlp.rs](../furiosa-opt-gemma4-12B/src/device/shared/mlp.rs)의 `feedforward_sharded`는 matrix별 FP4→FP8→unscaled BF16 weight를 한 번 준비하고 row tile에서 재사용한다. 16원소 F32 dot마다 해당 local scale을 적용한 뒤 합한다. Reduction축을 따라 scale이 달라지므로 최종 GEMV 뒤 한 번 곱하는 것으로 바꿀 수 없다. Down 이후 H480×8 후단을 유지한다.

공유 코드의 범위는 [caller 감사](experiments/e47-shared-caller-audit.md)에 있다. `Decode::sliding`과 `Decode::full`이 모두 `self.mlp`→`decoder_feedforward`를 호출한다. Full-attention의 shared residual loop expansion도 image에 영향을 주며 exact compile 대조만 수행했다. Vision의 직접 shared 호출 `add_vision`은 원본과 같지만 이후 text decoder를 거치므로 멀티모달 전체에 영향이 없다고 주장하지 않는다.

## 다음 작업과 실험을 시작할 조건

1. **공식 평가가 열리면 계약부터 재확인한다.** 변경 입력·허용오차·채점식을 확인하고 별도 제출기 인증 후 E53을 공식 평가한다. 공식 결과를 얻기 전까지3위 달성을 선언하지 않는다. 입력에 fixture/seed/좌표 전용 분기를 넣지 않는다.
2. **QKV의 더 넓은 입력 정확성은 별도로 확보한다.** [CPU 반례](research/qkv-boundary-design-audit.md)는 현재 공식 QKV x 생성 범위 밖이다. 실제 새 입력 또는 모델 workload가 공개되면 확인한다. [BF16 QKV 대안](../moa-dev/experiments/candidates/qkv-bf16-sharded-rms-e47/README.md)은 E18 BF16 projection과 E47 sharded RMS를 결합해 exact compile52,282를 얻었지만 NPU 미검증이다. 이 후보 전체 파일의 O는 E47이므로 **QKV 부분만** E53에 결합해야 한다. 기존 BF16-before-channel-scale 경계도 별도 검토 대상이다.
3. **FFN은 DMA·복호화·중첩을 같이 줄이는 새 근거가 필요하다.** 현재 static162,209에서 FP4→FP8→BF16 Main 작업량86,178, DMA 합집합135,607이다. 현재 DMA 순서·시간을 고정하면 decode만 공짜가 되어도 정적 개선 상한은 약1.196배다. 일반적인 성능 roofline으로 확대하지 않는다. [병목 분석](../moa-dev/experiments/candidates/ffn-parallel-epilogue/remaining-bottlenecks.md).
4. **Stage2는 실제 host 실행을 기준으로 다시 측정한다.** workload가 공개되면 [host/runtime.rs](../furiosa-opt-gemma4-12B/src/host/runtime.rs)의 초기화·prefill TTFT·decode TPOT·KV 상태·host-device 동기화·NPU 시간·메모리를 나누어 측정한다. Shared full-attention 및 vision/audio 이후 decoder도 포함한다. Stage1 세 커널 PASS를 전체 모델의 정확성·처리량으로 대신하지 않는다.

E53에서 추가 검사한 [BF16 O8](../moa-dev/experiments/candidates/output-bf16-eight-columns-e53/README.md)은 정적+1.88%, [FFN sharded pre-RMS](../moa-dev/experiments/candidates/ffn-sharded-input-rms-e53/README.md)의 두 경로도 각각+0.32%/+1.24%로 회귀해 NPU 없이 제외했다. FFN Main/DMA 작업량 감소가 두 엔진의 중첩 감소를 보상하지 못했다.

[O channel-scale fusion](../moa-dev/experiments/candidates/output-bf16-fused-channel-scale-e53/README.md)도 수치 경계를 유지하며 컴파일됐지만 scale DMA가 dot 이전 의존성이 되어 정적+154 cycles로 회귀했다. 별도 Main401과 결과 DMA24를 줄이는 것보다 새 선행 전송579가 컸다. 이 후보도 NPU 없이 제외했다.

기존 실패 후보를 새 근거 없이 반복하지 않는다. Native FP8 FFN은 E35/E38/E39/E41/E42/E46에서 정확성 통과 경로를 확보했지만 채택 BF16 FFN보다 느렸다. E40은 non-finite, E33은 큰 수치 오차로 실패했다. Down-only fused decode의 첫 개선은 E51의3회 반복에서 재현되지 않았다. Up staging·O32·QKV16-row도 전체 정적 또는 장치 비용에서 이득을 입증하지 못했다. [실험 원장](experiments/README.md)에 패키지별 결과와 실패 원인을 확인한 범위까지만 기록한다.

RNGD SDK0.8.1에는 확인되지 않은 prefetch priority/barrier/lifetime 강제 설정을 가정하지 않는다. [공개 API 검토](../moa-dev/experiments/candidates/ffn-up-gate-fused-decode-e37/scheduling-options.md). 새 배치 후보는 실제 weight/scale 전송·materialization·activation 복제를 줄이는 경로, 정확한 좌표 대응, compiler schedule의 전체 비용을 먼저 제시해야 한다.

## 조사 근거와 재현 절차

- [MOA 규칙·평가 계약](research/moa-rules-and-target.md): 허용 수정 범위, 모든 원소 비교, 단계별 metric을 우선한다.
- [RNGD/SDK](research/rngd-furiosa-opt.md): cluster/slice, DM/TRF, packet, TDMA/Switch의 실제 제약에 맞춰 consumer 배치를 정한다. Padding을 초기화된0으로 가정하지 않는다.
- [구현 분석](research/kernel-analysis.md): projection→norm→scatter/write와 scale축·반올림·cache 부수효과를 추적한다.
- [국내외 대회·논문](research/competition-techniques.md): AMD/FlashInfer 수상 구현의 layout·split-K·workload별 검증, 국내 대회의 단계별 제출 계약, FlashAttention의 중간 저장 제거 원칙을 참고한다. GPU 명령·완화된 오차·재양자화 허용을 RNGD에 그대로 적용하지 않는다.
- [O 입력 경계](research/output-boundary-design-audit.md), [잔차 scale 대안](research/output-residual-scale-design-audit.md): 적은 seed의 PASS가 상쇄 입력에 대한 보장이 아님을 확인한다. Adaptive residual scale나3/4항 FP8 분해도 전체 BF16 범위의 동치를 보장하지 않는다.

각 후보는 source/compiler/Cargo.lock/fixture/image/host hash 고정 → exact compile → 공식 harness 장치 검사 → 독립 추가 입력과 변경 부수효과 검사 → 동일 패키지 반복 → 소스 재빌드·image 확인 순서로 검증한다. `capture.py`는 소스와 기존 바이너리를 각각 관찰하므로 빌드 성공 후 즉시 캡처한다. Collector는 실제 업로드 archive를 내려받아 비교한다.

```bash
./moa-dev/moa cargo furiosa-opt build --release --bin test_kernels
./moa-dev/moa env RNGD_SUBMIT_TIMEOUT=70 ./scripts/rngd_test.sh --no-build --no-wait
python3 moa-dev/measure.py --jobs JOB_ID --reruns 2 \
  --label candidate --output docs/experiments/candidate.json \
  --provenance PATH_TO_PROVENANCE.json \
  --leaderboard-snapshot docs/experiments/leaderboard-20260921-initial.json
```

Arena 작업 상한은70초, queued/running 합계는2개다. 진단 cycles는 공식 입력 반복과 합치지 않는다. 원시 로그·대용량 artifact는 Git 제외 경로에, 재현 source/patch·결과·해시는 저장소에 보관한다.
