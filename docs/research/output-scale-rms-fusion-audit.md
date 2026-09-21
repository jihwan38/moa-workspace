2026-09-21의 **지원 API 설계 검토**다. 새 후보 실험이 아니며 구현·compile·NPU 실행은 하지 않았다. 결론은 E53의 첫 BF16 반올림을 보존하면서 `scale→square/mean`과 `scale 재계산→normalization`을 각각 한 Vector Engine pass로 만드는 직접적인 SDK 경로가 없다는 것이다. 이득을 입증할 근거가 없어 구현을 진행하지 않았다.

## 기준 경로와 필수 반올림

기준은 E53 clean commit `989461e3bf886141c158edef0420537fd03f16ac`이다. 실제 읽은 세 파일을 [동결 source](../../moa-dev/experiments/artifacts/e53-bf16-output-clean-source/src/ops.rs)와 byte 대조했다.

| 파일 | SHA-256 |
|---|---|
| `src/ops.rs` | `f2ad258f736dd8f3ebb64b4dc8c7b0e15579bded22e41a40ee355f4674f42c6a` |
| `src/device/sliding/projection.rs` | `34d1e03e3a6fcfce7997c1593774ab5d4382ec3a32ce839b0df450eb70e98581` |
| `src/device/shared/rmsnorm.rs` | `d962d6042c90ff51a080720e1f3c851e2375f70776b381a7048e62ccf4115ba5` |

`ops::sliding_attention_output` 140–145행은 `project_output→normalize_output_fused→add_output_residual`을 호출한다. [Projection](../../moa-dev/experiments/artifacts/e53-bf16-output-clean-source/src/device/sliding/projection.rs)의 `project_output` 282행부터는16-column F32 sum을 H480×8로 모은 뒤 `apply_output_channel_scale_f32` 323행에서 F32 scale을 곱하고 BF16으로 저장한다. [RMS](../../moa-dev/experiments/artifacts/e53-bf16-output-clean-source/src/device/shared/rmsnorm.rs) 115–131행은 그 BF16 값을 F32로 넓혀 제곱·480개 부분합·3840 나눗셈을 수행한다. 135–165행이8개 부분 평균 합산·EPS·sqrt·scalar 복제를 하고, 183–198행은 같은 BF16 값을 다시 읽어 나눗셈·gamma·BF16 cast를 수행한다.

필요한 값은 `z=BF16_RNE(F32(dot×scale))`다. 평균과 최종 numerator 모두 `F32(z)`를 사용해야 한다. Cast 없이 `(dot×scale)²`를 합산하거나 normalization에서 원래 F32 product를 사용하면 다른 수식이다. 예를 들어 product가 `1+2⁻⁸`이면 BF16 RNE는1이므로 제곱은1이지만, 반올림 전 제곱은1.0078277587890625다. SDK의 `src/cast.rs:487–505`도 이 halfway RNE 동작을 명시적으로 검사한다. 이는 중간 연산의 차이를 보이는 예이며 새 end-to-end NPU 반례를 실행했다는 뜻은 아니다.

## 실제 SDK guard

읽은 SDK root는 Docker `moa2026-dev` 안의 `/home/moa/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/furiosa-opt-std-0.8.1/`다. 아래 경로는 이 root 기준이다.

| 파일·근거 줄 | 확인한 제약 | SHA-256 |
|---|---|---|
| `src/engine/mod.rs:93–104,293–320` | CastTensor 이후 transpose/commit만 가능. `CanApplyVectorInit`는 Collect/Contraction에만 구현되며 Cast에 없음 | `79627b91df5d73f26843b8a1975851cca4c27260b4581a29e66434f4314791bd` |
| `src/engine/vector/scalar.rs:5–11` | VE scalar는 sealed f32/i32이며 BF16이 아님 | `21283ffe8cab16b8865077b52eb507162edad8bb4ce7c251704838a38a6484aa` |
| `src/engine/vector/stage/markers.rs:232–257` | Fp 이후 Logic/Fxp 단계로 돌아가는 transition이 없음 | `e6a05b9ad78695c7bb34978aef39e63e0778a35c8b51ed1d855b33d22bfc5621` |
| `src/engine/vector/tensor/vector_tensor.rs:1858–1869` | `vector_reinterpret`는 비트만 다시 읽고 기존 stage를 유지함 | `bb7f04f33b3d44254735c09d90ab4d59eed8b9719430f84770f342e5c35118e1` |
| `src/engine/vector/operand.rs:813–829,877–892` | VRF는 stream과 동일 Chip/Cluster/Slice, packet 연속값 또는 scalar 주소를 읽음. 재사용 가능 | `2324b08b20d4339abcd0f0e6a3c92e99d149d5e43b0015da19328a7ebf496ba6` |
| `src/constraints.rs:100–108` | VRF operand 용량은 slice당8192B 이하 | `89e6c23f71088182641ed110cf05ab202d9b8c46dd27cffd3d95d37ba1df1763` |
| `src/cast.rs:487–505` | F32→BF16 RNE 테스트 | `a5bc81ff9e06b89f4d5e25567592782af7e99ef90a2789a00c45a49012430d28` |

따라서 `MulF→vector_final→cast<bf16>→square`나 `→DivF`는 같은 stream에서 연결할 수 없다. Cast 뒤 DM에 commit하고 새 fetch로 F32로 넓혀야 한다. `vector_reinterpret<i32>`로 BF16 RNE를 직접 구현하려 해도 scale 곱셈의 Fp 단계에서 Logic/Fxp로 돌아갈 수 없다. BF16 반올림을 생략하는 근사 구현은 이 검토의 precision-preserving 조건을 만족하지 않는다. 이는 확인한 공개 API 경로의 제약이며 가능한 모든 하드웨어 알고리즘이 불가능하다는 주장은 아니다.

## VRF·전송·비용 판단

VRF 주소와 명목 용량은 막히는 이유가 아니다. Scale을 현재와 같은 `Cluster, Slice=[1#32,H/480], Element=H%480`에 한 번 읽으면 두 pass가 같은 handle을 재사용할 수 있다. Scale과 gamma는 각 packet의 H4 연속값을 읽고 RMS만 scalar broadcast다. Slice당 scale1920B+gamma1920B+RMS32B=3872B로8192B보다 작다. 이 합은 설계상 용량 확인이며 새3-operand pipeline의 allocator/port 성공을 compile로 검증한 결과는 아니다.

이 설계 자체 때문에 추가 HBM scale DMA나 layout TDMA가 필요한 것은 아니다. 다만 scale VRF와 F32 dot buffer를 최종 normalization까지 유지해야 한다. 논리적 activation DM read는 기존 `4H+2H+2H`와 제안 `4H+4H`가 모두8H bytes이고, 구현 가능하다면 중간 BF16 write2H를 줄이는 대신 scale 곱셈H개를 추가한다. 이는 byte 산정이며 cycle 절감을 증명하지 않는다.

현재 [E53 exact schedule](../../moa-dev/experiments/artifacts/output-bf16-activation-e47/candidate.json)의 scale Main은401 cycles다. 그러나 BF16 중간값을 합법적으로 materialize하면 바로 이 기존 pass가 남는다. Normalization에서 다시 scale+cast를 계산해도 두 번째 cast 뒤 다시 읽어야 하므로 제거한 pass를 되살리거나 추가 작업을 만든다. 앞선 [DPE/scale fusion](../../moa-dev/experiments/candidates/output-bf16-fused-channel-scale-e53/README.md)의579-cycle 선행 준비 문제와는 다른, VE 내부 cast 위치의 제약이다. 새 source·schedule·정확성·성능 결과는 없고 이 문서만 작성했다.
