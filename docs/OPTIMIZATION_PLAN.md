# MOA 2026 최적화 실행 계획

2026-09-21. 목표는 공식 평가에서 최소3위 진입이며, **아직 공식 제출·점수·순위를 확보하지 못했다.** 현재 기준 구현은 **E55**다. E54의 BF16 QKV 입력과 scale 뒤 BF16 경계를 유지하면서 dot·scale 실행을 합쳐 비용을 줄였다. 실제 RNGD에서 공식3개·추가6개 입력 구성과 별도 QKV 상쇄 진단2개를 통과했고 동일 패키지3회 반복을 완료했다. 이 진단은 현재 공식 QKV generator 지원 범위 밖이다. O 경계8개는 E53와 동일한 O image의 기존 검사 근거를 승계한다. 모델 전체 정확성·처리량이나 모든 입력의 정확성을 보장하는 기록은 아니다.

## 현재 기준과 채택 근거

E55는 E54의 O·FFN device image를 유지하고 QKV의 완전한 F32 dot 뒤 같은 Main에서 channel scale을 곱한다. 첫 BF16 cast 뒤 row Switch를 수행해 E54의 F32 재배치와 별도 scale pass를 없앴다. QKV 입력의 FP8 hi/lo 근사는 E54에서 제거됐고, O 입력 근사와 scale 전 BF16 반올림은 E53에서 제거됐다. [E55 소스·빌드 감사](experiments/e55-source-build-audit.json), [O 수치 검토](../moa-dev/experiments/candidates/output-bf16-activation-e47/math-review.md).

| 커널 | 원본 반복 중앙값 cycles | 이전 E54 반복 중앙값 cycles | E55 반복 중앙값 cycles |
|---|---:|---:|---:|
| QKV | 244,051 | 108,700 | 102,878 |
| Attention output | 404,222 | 50,970 | 51,249 |
| FFN | 4,264,297 | 382,268 | 380,339 |

E55는 [동일 패키지 job71027/71057/71072](experiments/e55-fused-qkv-repeats.json)의 job 중앙값들의 중앙값이다. 세 job 모두15개 비교를 통과했고 상대 범위는 QKV0.976%/O1.169%/FFN1.036%다. 원본 대비2.37×/7.89×/11.21×이며 [첫 screen](experiments/e55-fused-qkv-screen.json)을 최종 집계로 대신하지 않는다. 이전 E54 열은 [job70933/70974/70987 반복](experiments/e54-bf16-qkv-repeats.json)이다. E55의 O/FFN image는 E54와 같으므로 그 두 커널의 cycle 차이를 소스 개선·회귀로 해석하지 않는다.

- 공식 입력: E55 세 반복 job 각각 출력 비교15개 모두 PASS.
- 별도 입력 run3..8: E55 job71052의 출력 비교30개 모두 PASS. 같은 입력의 반복 job을 새 입력으로 세지 않는다. [추가 검사](experiments/e55-fused-qkv-extra-correctness.json).
- QKV 경계: E55 job71032의2개 case/QKV6개 비교 모두 PASS이며 checksum38/38과 비대상 KV cache8,380,416개 BF16 bits 보존을 확인했다. 같은 probe·oracle·payload에서 원본70944와 E54 70958도 PASS지만 E53 70943은8,072개 값이 실패했다. 요약값의 일치로 전체 출력 배열의 동일성을 추론하지 않는다. [기존 대조와 E55 후속](experiments/qkv-boundary-comparison.md).
- O 경계: **새 E55 실행이 아니라 같은 O image의 E53 job70810 근거를 승계**한다. 원본70799와 E53 70810은8/8 PASS, E47 70785는 상쇄 입력3840개가 기대0.609375 대신0.5였다. [O 비교](experiments/output-boundary-comparison.md).
- 동결된46개 source/Cargo/toolchain 입력을 독립 scratch와 live에서 대조했고 두 전체 host 빌드가 byte-identical하다. exact QKV image와 공식 host가 일치하며 별도 추가 입력 빌드도 같은 세 device image를 각각 한 번 포함한다. [빌드 감사](experiments/e55-source-build-audit.json).

E55 QKV 반복 중앙값102,878 cycles는 이전 BF16 기준 E54의108,700보다5.36% 낮다. 입력 근사가 있던 [E53 반복](experiments/e53-bf16-output-repeats.json)의95,621보다는7.59% 높아 일반 입력 정확도를 위한 비용이 남는다. 과거 E54는 E53보다13.68% 느렸고 E53의 O 반복 중앙값은 E47보다14.12% 높았다. E47/E53 모두 실패 범위를 명시한 과거 성능 후보로 보존한다. QKV 반례는 공식 generator 지원 범위 밖이므로 E53의 기존 공식 입력 PASS를 뒤집는 결과가 아니다. E55도 모든 finite 입력의 수학적 동치를 증명하지 않으며 F32 합산 순서·RMS·overflow/FTZ는 별도 한계다. O의 tiny case처럼 expected가 atol보다 작으면 통과만으로 작은 값 보존을 증명하지 못한다.

공식 사이트는 평가 기준 수정으로 제출을 중단했고 Arena 개발은 계속 가능하다고 안내한다. 별도 `moa-submitter` 인증도 미완료다. 리더보드는 참고용이며 최종 순위는 동일 조건의 재실행으로 결정된다. [공식 공지](https://micro2026-moa.github.io/leaderboard.html), [상태 기록](experiments/official-submission-status-20260921.json). 과거 공개 기준에서 계산한 참고 점수를 현재 순위로 환산하지 않는다.

## 구현의 실제 실행 경로

진입점은 [src/ops.rs](../furiosa-opt-gemma4-12B/src/ops.rs)의 `sliding_project_qkv`, `sliding_attention_output`, `decoder_feedforward`다. 공식 입력 생성·비교 harness와 공개 device signature는 유지했다. [E55 계약 검사](experiments/e55-source-contract-audit.json)에서 source43개 중 허용된7개 경로만 원본과 다르고, 공개10개 signature와 공식 harness6개 파일은 원본과 같음을 확인했다. E54 대비 변경은 `projection.rs`의 `project_query`, `project_one_kv_matrix` 두 함수 본문뿐이며 `ops.rs` 전체와 head norm·RoPE·cache 후단, 다른42개 source는 같다.

**QKV:** HBM BF16 x → H480×8 BF16 pre-RMS → full-H gather와 ring32 BF16 broadcast → FP8 weight의 정확한 BF16 lookup 및 BF16 DPE → 같은 Main에서 완전한 F32 dot에 channel scale 곱하기 → 첫 BF16 cast → BF16 row Switch → head RMS → Q/K RoPE → Q 출력과 지정 KV cache 슬롯 write. [projection.rs](../furiosa-opt-gemma4-12B/src/device/sliding/projection.rs)의 `project_query`와 `project_one_kv_matrix`는 contiguous scale DMA 뒤 inverse `InterTranspose`로 각 raw row에 맞는 F32 scale을 VRF에 놓는다. dot을 완전히 reduce한 뒤 곱하고 BF16으로 변환하므로 scale 전 추가 BF16 반올림이 없다. K/V는 `project_key_value`가 같은 BF16 TRF를 재사용하고 Q의 TRF 적재는 별도다. 두 cluster와8-way row interleave를 유지한다. 삭제된 `prepare_fp8_split`·`broadcast_quantized`는 [이전 E53 동결 소스](../moa-dev/experiments/candidates/qkv-bf16-f32-scale-e53/projection.rs.original)의 FP8 근사 경로이며 현재 실행되지 않는다.

**O:** HBM BF16 attention 결과 → 16개256-column shard의 BF16 TRF → FP8 weight를 정확히 BF16으로 lookup한 단일 dot → F32 column reduction → F32 H480×8 gather → F32 channel scale → 첫 BF16 cast → RMS → residual → HBM 갱신. `project_output`과 `apply_output_channel_scale_f32`가 이 경계를 구현한다. 여기에는 attention softmax 계산이 없다. 원본 Python reference와 scale/cast 순서는 맞지만 F32 reduction grouping은 달라질 수 있다.

**FFN:** residual → pre-RMS → FP4 up/gate → GeGLU → FP4 down → global scale → post-RMS → residual 및 layer scale → HBM write. [mlp.rs](../furiosa-opt-gemma4-12B/src/device/shared/mlp.rs)의 `feedforward_sharded`는 matrix별 FP4→FP8→unscaled BF16 weight를 한 번 준비하고 row tile에서 재사용한다. 16원소 F32 dot마다 해당 local scale을 적용한 뒤 합한다. Reduction축을 따라 scale이 달라지므로 최종 GEMV 뒤 한 번 곱하는 것으로 바꿀 수 없다. Down 이후 H480×8 후단을 유지한다.

공유 코드의 범위는 [caller 감사](experiments/e47-shared-caller-audit.md)에 있다. `Decode::sliding`과 `Decode::full`이 모두 `self.mlp`→`decoder_feedforward`를 호출한다. Full-attention의 shared residual loop expansion도 image에 영향을 주며 exact compile 대조만 수행했다. Vision의 직접 shared 호출 `add_vision`은 원본과 같지만 이후 text decoder를 거치므로 멀티모달 전체에 영향이 없다고 주장하지 않는다.

## 다음 작업과 실험을 시작할 조건

1. **공식 평가가 열리면 계약부터 재확인한다.** 변경 입력·허용오차·채점식을 확인하고 별도 제출기 인증 후 검증된 E55를 공식 평가한다. 설치된 공식 CLI1.2.0의 [로컬 전송 검사](experiments/e55-official-submit-preflight.json)에서는 크레이트·하위 폴더·명시적 `--source` 세 경로 모두 실측 E55의 허용 소스23개/160,829B를 정확히 전송했다. 이 검사는 loopback 수신기로 수행했고 공식 서버 인증·계약·점수는 미검증이다. 공식 결과를 얻기 전까지3위 달성을 선언하지 않는다. 입력에 fixture/seed/좌표 전용 분기를 넣지 않는다.
2. **새 QKV 실험에는 E55를 넘을 구체적인 전송·일정 근거가 필요하다.** [BF16 QKV dot·channel-scale 결합](../moa-dev/experiments/candidates/qkv-bf16-fused-channel-scale-e54/README.md)은 이미 E55로 채택했고 공식 입력·추가 입력·동일 경계·3회 반복 검증을 마쳤다. 정적52,282→51,293(−1.89%)와 실제 QKV 반복−5.36%는 서로 다른 측정이다. 후속 공통 TRF는 정적 회귀, broadcast→TRF 결합은 compile 실패로 제외했으며 현재 조사한 후보 중 바로 실측할 미검증 개선 후보는 남아 있지 않다. 아래 실패 근거를 새 사실 없이 반복하지 않는다. BF16 입력 보존·완전한 F32 dot 뒤 scale·첫 BF16 경계와 cache 부수효과를 유지하고, 새 공식 입력이나 실제 모델 workload가 공개되면 별도 검증한다. [QKV 반례와 실제 대조](experiments/qkv-boundary-comparison.md)는 현재 공식 x 생성 범위 밖의2개 진단에 한정된다. 이전 [BF16 QKV 초안](../moa-dev/experiments/candidates/qkv-bf16-sharded-rms-e47/README.md)의 E47 O와 scale 전 BF16 경계로 되돌리지 않는다.
3. **FFN은 DMA·복호화·중첩을 같이 줄이는 새 근거가 필요하다.** 현재 static162,209에서 FP4→FP8→BF16 Main 작업량86,178, DMA 합집합135,607이다. 현재 DMA 순서·시간을 고정하면 decode만 공짜가 되어도 정적 개선 상한은 약1.196배다. 일반적인 성능 roofline으로 확대하지 않는다. [병목 분석](../moa-dev/experiments/candidates/ffn-parallel-epilogue/remaining-bottlenecks.md).
4. **Stage2는 실제 host 실행을 기준으로 다시 측정한다.** workload가 공개되면 [host/runtime.rs](../furiosa-opt-gemma4-12B/src/host/runtime.rs)의 초기화·prefill TTFT·decode TPOT·KV 상태·host-device 동기화·NPU 시간·메모리를 나누어 측정한다. Shared full-attention 및 vision/audio 이후 decoder도 포함한다. Stage1 세 커널 PASS를 전체 모델의 정확성·처리량으로 대신하지 않는다.

E53에서 추가 검사한 [BF16 O8](../moa-dev/experiments/candidates/output-bf16-eight-columns-e53/README.md)은 정적+1.88%, [FFN sharded pre-RMS](../moa-dev/experiments/candidates/ffn-sharded-input-rms-e53/README.md)의 두 경로도 각각+0.32%/+1.24%로 회귀해 NPU 없이 제외했다. FFN Main/DMA 작업량 감소가 두 엔진의 중첩 감소를 보상하지 못했다.

[O channel-scale fusion](../moa-dev/experiments/candidates/output-bf16-fused-channel-scale-e53/README.md)도 수치 경계를 유지하며 컴파일됐지만 scale DMA가 dot 이전 의존성이 되어 정적+154 cycles로 회귀했다. 별도 Main401과 결과 DMA24를 줄이는 것보다 새 선행 전송579가 컸다. 이 후보도 NPU 없이 제외했다.

[O scale→RMS 결합 API 검토](research/output-scale-rms-fusion-audit.md)는 별도 scale Main의 BF16 반올림을 RMS square·분자 앞에 보존해야 한다는 제약을 확인했다. 조사한 SDK0.8.1 VE 경로에서 같은 Main 중간 BF16 반올림은 표현하지 못했고 DM commit을 넣으면 기존 scale pass가 돌아온다. 이는 조사한 API 경로의 미해결 설계이며 NPU 성능 결과가 아니다.

[QKV 입력 staging 감사](research/qkv-bf16-input-staging-audit.md)는 gather·ring broadcast가 이미 공유되고 Q/KV TRF 적재만 중복임을 확인했다. [공통 TRF 후보](../moa-dev/experiments/candidates/qkv-bf16-shared-trf-e55/README.md)는 첫 exact compile에서 Sub1,223과 TRF 예약 절반을 줄였지만 전체51,293→52,562로 회귀했다. K lookup 준비가 뒤로 밀린 일정이 원인이며 NPU 없이 제외했다.

[Broadcast→공통 TRF 결합](../moa-dev/experiments/candidates/qkv-bf16-broadcast-trf-e55/README.md)은 named output의 Switch bitmap 검증에서 첫 compile이 실패했다. 의도한 leader 주소와 실제 lowering을 구분하며, 유효 image·schedule이 없어 성능 수치는 없다. 검증되지 않은 TRF 변환이나 weight의 anonymous-copy 재해석으로 우회하지 않았다.

[Mixed-dtype DPE 감사](research/mixed-dtype-dpe-audit.md)는 공개 SDK0.8.1의 float contraction이 같은 dtype의 두 operand를 요구함을 확인했다. FP8 weight와 BF16 activation을 직접 곱해 lookup만 제거하는 공개 경로는 없다. 현재 lookup은 DPE와 같은 Main pipeline에 있고 전체 BF16 weight를 DM에 저장하지 않는다. 별도 FP8→F32→BF16 materialization으로 우회하면 Q/K/V 합계 BF16 DM write60MiB와 read60MiB가 추가되며, 이는 검증된 성능 후보가 아니다.

[Lookup table 공유 감사](research/fp8-lookup-table-sharing-audit.md)에서도 공개 prepared-table handle·수명·재사용 API를 찾지 못했다. E55의 세 table 준비는 각 DMA838+Sub1,293 cycles지만 합계6,393을 전체 시간에서 뺄 수 없다. K table은 dot보다 일찍 준비되며 late-K 현상은 회귀한 공통 TRF 후보에서만 관측했다. Compiler 내부 공유 정책이나 미래 SDK의 지원 가능성까지 부정하는 결론은 아니다.

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
