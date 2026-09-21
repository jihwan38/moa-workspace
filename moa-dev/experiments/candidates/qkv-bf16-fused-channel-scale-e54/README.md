후속 E55 NPU 검사: [공식 입력 첫 실행](../../../../docs/experiments/e55-fused-qkv-screen.json)15개, [추가 입력](../../../../docs/experiments/e55-fused-qkv-extra-correctness.json)30개, [QKV 경계](../../../../docs/experiments/e55-qkv-boundary-probe.json)2개 case의6개 출력 비교가 모두 PASS다. [같은 패키지 세 번 반복](../../../../docs/experiments/e55-fused-qkv-repeats.json)71027/71057/71072도 모두 PASS다. QKV 중앙값102,878 cycles는 E54보다5.36% 낮고 job 중앙값 범위는0.976%로, E55를 채택했다. O/FFN 중앙값은51,249/380,339이며 두 image는 E54와 같다. 아래 본문과 manifest의 “NPU 미검증”은 **후보를 exact compile로 동결한 당시 기록**이다. 전체 빌드·측정 이후 근거는 [E55 빌드 감사](../../../../docs/experiments/e55-source-build-audit.json)와 위 결과에 별도로 남긴다.

E54의 Q/K/V projection에서 **완전한 F32 dot에 channel scale을 곱하고 첫 BF16 cast까지 같은 Main pipeline에서 수행하는 후보**다. 세 번째 exact compile에서 통과했고 정적 전체 makespan은 **52,282→51,293 cycles(−989, −1.892%)**다. NPU를 실행하지 않았으며 하드웨어 정확성·성능 개선은 아직 검증하지 않았다.

기준은 [clean E54 snapshot](../../artifacts/e54-bf16-qkv-clean-source)이다. `src/device/sliding/projection.rs`의 QKV helper 두 본문만 바꿨다. O suffix, `ops.rs`, BF16 activation, weight lookup, DPE 누적·reduction 축, head RMSNorm, RoPE, cache write, FFN, public signature와 harness는 그대로다. 나머지42개 source와 Cargo/toolchain 파일은 byte-identical이며, [46개 실제 Docker compile 입력](compiled-source-manifest.json)을 보존한 host crate와 대조했다.

## 최종 데이터 경로와 좌표

```text
원래 canonical BF16 channel-scale HBM→typed DMA
→Sub BF16→F32 cast→inverse InterTranspose→row-Time scalar VRF

원래 BF16 activation TRF + FP8 weight→BF16 lookup
→원래 완전한 F32 DPE dot→Mul0(channel-scale VRF)→첫 BF16 cast
→4개 BF16=8B commit→기존 E18 BF16 InterTranspose→원래 head gather
→변경 없는 head RMS/RoPE/cache
```

Q 행 `q=2048*c+64*g+8*r+s`에서 `c<2,g<32,r<8,s<8`이다. Weight는 raw slice `8*g+s`, local row `r`에 있고, scale은 기존 canonical slice `8*g+r`, local `s`에 DMA된다. Sub의 `InterTranspose{slice1:8,slice0:1,time0:1}`가 scale의 저위 slice와 Time을 교환해 동일 weight 행의 raw slice `8*g+s`, Time `r`에 전달한다. K/V는 `p=1024*c+32*g+8*r+s`, `r<4`이며 canonical low-slice의 `4#8` padding이 Sub 이후 Time의 `4#8` padding으로 이동한다.

VRF는 Q `[Qs/8%8,1#8]`, K/V `[Ps/8%4#8,1#8]`다. 세 projection 모두 physical slice당256B(전체512 slices에서131,072B)를 예약한다. 각 row-Time의 유효 scalar는 `32*r` 바이트에 있다. Q는8개, K/V는4개 scalar만 읽으며 packet padding과 K/V row4..7을 데이터로 승격하지 않는다. 이 최종 경로는 처음 의도했던 compact 32B VRF보다 큰 예약량을 사용한다.

[verify_candidate.py](verify_candidate.py)가 Q4096행/KV2048행 전체의 weight·scale 대응, inverse scale Switch와 forward output Switch, VRF scalar 주소, 8B BF16 commit, head reshape 좌표를 검사했다. K/V helper는 K와 V에 동일하게 적용된다. 소스에서 E54의 DPE fetch/lookup/contract chain이 그대로인지도 대조한다. [검사 결과](verification.json)는 소스·좌표 검증이며 하드웨어 비트 동일성 증명은 아니다.

공개 lowered graph의 Q `O19`, K `O33`, V `O45`에서 `operation_unit→FpMul0→cast_compaction(bf16)→4-row BF16 transpose/8B commit` 순서를 확인했다. 각각의 scale Sub는 `O16/O30/O42`, scale HBM은 `IO(4)/IO(5)/IO(6)`이다. Weight의 `IO(1)/IO(2)/IO(3)`와 함께 검사하여 K/V를 source 호출 순서나 schedule 실행 순서로 추정하지 않았다. 기존 E54는 F32 dot·F32 Switch 뒤 별도 scale Main에서 첫 BF16 cast를 했다. 후보는 같은 F32 scale 산술을 이동시키며 Switch는 산술 없이 이미 반올림된 BF16 값을 이동한다. NPU가 이 의도대로 실행하는지는 별도 검증이 필요하다.

SDK0.8.1의 확인한 근거는 설치된 `furiosa-opt-std-0.8.1/src/engine/vector/operand.rs:813–829`(같은 Chip/Cluster/Slice, Time 기반 scalar VRF index와 남은 padding 허용), `engine/switch.rs:57–72`(dtype/packet을 보존하는 routing), `engine/mod.rs:284–300`(fetch_cast→switch→collect 순서), `engine/collect.rs:90–98,133–142`(Sub `to_vrf`와 lowering 검증)이다. 실제 최종 compile의 허용 여부를 API 문서만으로 추정하지 않았다.

## 두 번의 제한 내 수정

1. [attempt1 소스](projection.attempt1-vrf4-failed.rs)는 raw interleaved row에 scale을 직접 typed DMA하고 compact VRF에 적재했다. K/V의4개 F32를 `.to_vrf()`하려다 `Collect output packet must be exactly 32 bytes ... 4 elements = 16 bytes`로 실패했다. [첫 로그](../../artifacts/qkv-bf16-fused-channel-scale-e54/compile-attempt1.log)에 남겼다.
2. [attempt2 소스](projection.attempt2-unaligned-dma-failed.rs)는 K/V collect/VRF를 valid4#8로 바꿨다. 그 제약을 통과했지만 Q scale의 raw interleaved typed DMA가 각 slice에2B씩 흩어 보내야 해 `strides ... is not aligned by 8`로 실패했다. [두 번째 로그](../../artifacts/qkv-bf16-fused-channel-scale-e54/compile-attempt2.log)의 source/destination mapping이 근거다. K/V에서도 이 직접 DMA 방식이 합법이라고 주장하지 않는다.
3. 최종 source는 원래 E54의 연속8개 BF16 scale DMA를 유지하고 Sub inverse Switch로 raw row에 전달한다. Scalar packet마다32B인 padded VRF를 사용해 두 제약을 함께 피했다. [세 번째 로그](../../artifacts/qkv-bf16-fused-channel-scale-e54/compile-attempt3.log)가 exact compile 성공 기록이다. 총3회로 종료했으며 추가 variant/build는 없다.

## 전체 정적 비용

| 항목 | E54 기준 | 후보 | 차이 |
|---|---:|---:|---:|
| 전체 makespan | 52,282 | 51,293 | −989 |
| Instruction 수 | 143 | 140 | −3 |
| Main duration 합 | 30,818 | 29,969 | −849 |
| Sub duration 합 | 10,793 | 10,979 | +186 |
| DmaLoad duration 합 | 34,916 | 34,916 | 0 |
| Q / K / V dot 각각 | 5,063 / 2,663 / 2,663 | 동일 | 0 |
| Q/K/V row Switch 각각 | 343 | 343 | 0 |
| 별도 channel-scale Main 각각 | 283 | 제거 | −283×3 |
| Scale Sub 각각 | 265 | 327 | +62×3 |

수치는 instruction lifetime에서 계산했다. Resource duration 합은 overlap을 포함하므로 전체 시간처럼 더하지 않는다. 최종 51,293에는 scale DMA/Sub, projection, RMS/RoPE, cache DMA와 compiler sync가 모두 포함된다. Scale DMA는 Q792, K/V각665 cycles로 이전과 같다.

스케줄 순서도 달라졌다. 기준은 Q→V→K dot이고 후보는 Q→K→V dot이다. Q dot 시작은23,702→26,900으로 오히려 늦어진다. 후보의 마지막 V dot은43,041–45,704이며 이후 짧은 V RMS/cache tail로 끝난다. 기준은 K dot41,966–44,629 이후 K RMS/RoPE/cache tail이 마지막이다. 최종 cache 완료는51,682→50,693, 그 뒤 고정600-cycle sync를 포함한 makespan이989 감소한다. 따라서 “Main849 감소가 그대로 전체 개선”으로 해석하지 않는다. [static-comparison.json](static-comparison.json)에 HBM identity, weight/scale DMA, Sub, dot/Switch lifetime과 VRF shape를 기록했고 [분석 코드](analyze_artifacts.py)로 재검증할 수 있다.

## 재현과 동결

Scratch는 Docker `/tmp/moa-qkv-bf16-fused-channel-scale-e54`이고 target/kernel output도 그 내부만 사용했다. `CARGO_HOME=/home/moa/.cargo`와 frozen E54 toolchain/Cargo.lock을 사용했다.

```bash
cd /tmp/moa-qkv-bf16-fused-channel-scale-e54
CARGO_HOME=/home/moa/.cargo \
CARGO_TARGET_DIR=/tmp/moa-qkv-bf16-fused-channel-scale-e54/target \
FURIOSA_OPT_OUT_DIR=/tmp/moa-qkv-bf16-fused-channel-scale-e54/target/furiosa-opt/kernel \
cargo furiosa-opt compile ops::sliding_project_qkv --exact \
  --dump-schedule candidate.json --dump-graph candidate-graph.json
```

[projection.rs](projection.rs) SHA256는 `380e57d6502e26086bd6d933ce762f4ad5de18587d52a8ef4aac7fa87e3b9c9f`다. Image는168,960B, SHA256 `5bdab7eafec5ac0aca352101deee334180b550c87db23e33c70d008892edba9e`다. [prepare_source.py](prepare_source.py)가 최종 source를 byte-identical하게 재생성하는지 확인했고 [patch](candidate.patch), [manifest](manifest.json), 전체 crate·schedule·graph·image를 보존했다. E54 52,282 기준 schedule은 동일 `57a52d1a…` QKV image의 이전 exact compile 결과다. Live source/target, 이전 candidate, Git, NPU는 변경하지 않았다.

후속 재현 근거: 기준 source43개와 Cargo.toml/toolchain은 Git commit `2fceb2d45c25510489c73a9c4db969bf6cc7421b`의 `furiosa-opt-gemma4-12B/`와 모두 byte-identical하다. Cargo.lock은 공식 crate에서 Git 제외되므로 별도로 추적한 [gemma4-Cargo.lock](../../../locks/gemma4-Cargo.lock)을 사용한다. 이 lock도 동결 입력과 같다. 새 clone에서는 위 commit을 별도 worktree로 꺼내 crate의 Cargo.lock을 복원하고, 이 폴더의 `projection.rs`만 `src/device/sliding/projection.rs`에 적용하면 같은46개 compile 입력을 복원할 수 있다. [E55 빌드 감사](../../../../docs/experiments/e55-source-build-audit.json)에 후속 전체 빌드와 source 귀속을 기록한다. 위 manifest의 compile-only 상태는 당시 동결 기록이다.
