# E53 QKV: BF16 activation과 scale 이전 F32 유지

**E54로 통합해 공식 입력15개 비교·추가30개 비교·QKV 경계2개를 NPU에서 통과했다.** [첫 screen](../../../../docs/experiments/e54-bf16-qkv-f32-scale.json), [추가 입력](../../../../docs/experiments/e54-bf16-qkv-extra-correctness.json), [경계 대조](../../../../docs/experiments/qkv-boundary-comparison.md). 정리한 live source는 호출하지 않는 FP8 helper를 제거했으며 전체 host가 이 prototype의 측정 파일과 byte-identical하다. [빌드 감사](../../../../docs/experiments/e54-source-build-audit.json).

아래는 최초 exact compile의 설계·동결 기록이다. `manifest.json`과 `verification.json`의 NPU 미실행 표시는 그 compile 단계의 상태이며 이후 root 결과는 위 링크로 구분한다. 첫 exact compile은 PASS, 정적 makespan은52,282 cycles였다. E53의 QKV FP8 activation 근사를 제거하고 channel scale 전에 생기던 BF16 반올림도 없앤다. E53 native FP8 QKV의 정적50,364보다1,918(+3.81%) 높고 이전 `qkv-bf16-sharded-rms-e47`의52,282와 같다. 정적 수치를 실측 성능이나 모든 입력 정확성으로 해석하지 않는다.

동결 파일은 `projection.rs`와 `ops.rs`, 원본과 patch도 같은 폴더에 있다. Image·graph·schedule·compile log와 전체 crate copy는 `../../artifacts/qkv-bf16-f32-scale-e53/`에 있다.

| 대상 | SHA256 |
| --- | --- |
| projection.rs | `ab01f626d4ddfc8920312df19bb153e501ebd24495f166d322c3883a92c675ba` |
| ops.rs | `ffdab428fd4c8126b77e145b75073c907732bf0fa9ca830a123fe828e7b08888` |
| QKV image, 171,776 B | `57a52d1aa37f16116d14638b30f34947d2c7e7ed653986436cdfa51ee54089dc` |

## 실제 경로와 변경 범위

Parent가 지정한 E53 checkpoint7ad11cd의 동결 source 기준으로 `e53-bf16-output-clean-source`를 사용했다. 기존 [BF16 QKV 후보](../qkv-bf16-sharded-rms-e47/README.md)의 helper prefix와 `ops::sliding_project_qkv`만 통합했다. 그 후보의 E47 O suffix를 복사하지 않았다. **현재 E53 `project_output`부터 파일 끝까지가 byte-identical**이고, 나머지41개 src 및 public signature·harness·head RMS·RoPE·cache write가 그대로다. Ops의 QKV 함수 밖도 변하지 않았다.

```text
BF16 HBM x → 기존 두 cluster H480×8 pre-RMS → BF16
→ ring32의 anchor에 H3840 assembly → 기존 BF16 broadcast → BF16 TRF
FP8 weight → 기존 BF16 lookup → BF16 dot/FP32 accumulation
→ F32 raw-row DM → F32 InterTranspose → F32 canonical-row DM
→ F32 per-channel scale → 첫 projection BF16 cast
→ 기존 Q/K/V head RMS → 기존 Q/K RoPE → 기존 cache/output write
```

`project_one_kv_matrix`를 K/V가 각각 원래 weight/scale로 호출한다. V도 기존 head RMS를 거치며 RoPE는 없다. 입력의 `prepare_fp8_split` 호출은 없어진다. O suffix 보존 때문에 사용하지 않는 QKV quantizer helper 정의는 파일 뒤에 남을 수 있지만 이 새 QKV 경로가 호출하지 않는다.

원래 E18/BF16 helper는 raw dot을 BF16으로 cast하고 행을 복원한 뒤 channel scale을 곱했다. 이번에는 네 `contraction` DM(각 helper의 raw와 transpose 결과)을 F32로 바꾸고 중간 BF16 cast 두 곳 및 이제 불필요한 F32 identity fetch cast를 제거했다. Channel scale 후 BF16 cast는 유지한다. 따라서 원래 Python `scaled_projection`의 `F32 matmul → F32 scale → BF16` 경계에 맞춘 변경이며 이전 device와 bit-identical한 결과를 목표로 하지 않는다.

## Row mapping과 SDK 정렬

Q row는 `q=2048*c+64*g+8*r+s`, physical slice=`8*g+s`, local row=`r<8`이다. InterTranspose 후 slice=`8*g+r`, local=`s<8`이 된다. KV는 `p=1024*c+32*g+8*r+s`, `r<4`이며 Switch의 local row `4#8` padding을 output slice에도 끝까지 유지한다. 마지막 head gather가 유효값만 복원하므로 padding을 값으로 승격하지 않는다. Weight와 scale의 논리 row는 바뀌지 않는다.

F32 transpose는 한 번에2행을 사용한다. Raw Q는 `Time=Qs/16%4`, `Packet=Qs/8%2#8`; raw KV는 `Time=Ps/16%2`, `Packet=Ps/8%2#8`이다. 각각 유효 F32 두 값만 `commit_trim`하여 **8B**를 쓴다. Switch input의 scalar packet은 `1#2`로 같은8B이고, output의8개 local row는 네 번의 2-F32/8B commit으로 저장한다. BF16의4행 transpose를 F32에 그대로 적용하지 않았다.

`verify_candidate.py`/`verification.json`은 Q4096·KV2048 전 좌표와 역변환, raw/canonical F32 commit 주소, head view, KV padding을 검사한다. 모든 public device signature 및 headnorm/RoPE/cache tail/O suffix 보존도 검사했다. 실제 Docker scratch의43개 src+Cargo3개 hash가 `compiled-source-manifest.json`과 같음을 확인했다.

Lowered `candidate-graph.json`에서는 세 projection branch 모두 raw tensor와 Switch 결과가 `Float32`다. Raw dot와 InterTranspose에 `data_type: Float32`, `in_rows:2`, `commit_in_size:8B (=2EA)`가 있고 BF16 compaction이 없다. 후속 channel-scale `FpMul0` 뒤 처음 `output_element_type:bf16` compaction이 나온다. 해당 operator/tensor 연결은 `graph-audit.json`에 남겼다. 이는 실제 lowering 자료이며 NPU 실행 검증을 대신하지 않는다.

## 비용과 수치 한계

`schedule-summary.json`의 전체 makespan/instruction 수는 **52,282 / 143**이다. 이전 BF16 early-cast 후보와 Main30,818, DmaLoad34,916, DmaGather1,868, DmaStore441, DmaScatter1,740, DmaStos5,604, Sub10,793의 duration sum까지 같다. Resource별 합은 겹칠 수 있으므로 전체 시간으로 합산하지 않는다. Static이 같다는 결과로 실제 device latency나 메모리 비용이 같다고 주장하지 않는다.

BF16 activation 소실과 이른 projection BF16 반올림은 제거했다. 그러나 DPE의 FP32 accumulation order는 CPU matmul과 다를 수 있으며 기존 pre/head RMS의 sqrt/div와 CPU rsqrt, overflow/underflow·FTZ 및 cancellation 문제가 전부 해결됐다고 보장하지 않는다. 기존 BF16 helper의 과거 PASS와 기존 E53 QKV9-input PASS도 이 새 조합의 NPU 정확성 증거가 아니다. 공식 및 독립 경계 검사는 Root가 별도 수행한다.

## 재현

설치된 compiler/std0.8.1과 기존 lock을 사용해 독립 scratch에서 **compile1회, correction0회**였다. Live source/target·Git·NPU를 이 agent가 변경하지 않았다. Root의 full host 통합·NPU 검증은 이 compile 작업과 별개다.

```bash
cd /tmp/moa-qkv-bf16-f32-scale-e53
export CARGO_HOME=/home/moa/.cargo
export CARGO_TARGET_DIR=$PWD/target
export FURIOSA_OPT_OUT_DIR=$PWD/target/furiosa-opt/kernel
cargo furiosa-opt compile ops::sliding_project_qkv --exact \
  --dump-schedule candidate.json --dump-graph candidate-graph.json
```

`prepare_source.py`는 동결 E53+이전 BF16 prefix에서 같은 후보를 격리 artifact에 재구성한다. 전체 해시와 상태는 `manifest.json`에 있다. 확인일 2026-09-21.
