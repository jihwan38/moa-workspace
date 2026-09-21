E53 O16에서 **16-column F32 sum 뒤 channel scale을 같은 Main pipeline에 곱하고, BF16 결과를 epilogue로 모으는 후보**다. 두 번째 exact compile에서 지원을 확인했지만 전체 정적 시간은 **27,239→27,393 cycles(+154, +0.565%)**로 늘었다. 별도401-cycle Main 제거만으로 개선을 주장하지 않는다. 지시에 따라 NPU·full host build 없이 동결했으며 추가 시도는 하지 않았다.

기준은 clean E53 commit `989461e3bf886141c158edef0420537fd03f16ac`이다. [projection.rs](projection.rs)의 `project_output` 본문만 변경했다. QKV prefix, O16 weight/input/reduction partition, channel-scale helper 정의, RMS·residual·FFN, public ops·harness는 그대로다. 기존 scale helper는 미사용 상태로 남겨 scope를 제한했다. 나머지42개 source는 commit과 byte-identical이고 [46개 입력 해시](compiled-source-manifest.json)를 실제 Docker compile scratch와도 대조했다.

## 지원 API와 구현

SDK0.8.1 `engine/vector/tensor/vector_tensor.rs:614`의 `vector_init→vector_inter_slice_reduce`는 `VeOrder::InterFirst`를 반환한다. 같은 파일649행부터 intra-slice tag로 진입할 수 있다. `engine/vector/operand.rs:813`은 VRF의 Chip/Cluster/Slice가 stream과 같아야 하고, Time에 따라 단일 주소를 broadcast하거나 packet의 연속값을 읽을 수 있다고 명시한다.

이를 이용해 scale HBM의 `[H]`를 wire-identical `[Dummy2,H%1920]`로 이름만 나누고, `OutputRows=[H/120%16,1#16]`의 유효 leader마다 연속 BF16120개=240B를 typed DMA로 읽는다. Sub가 이를 F32 VRF `Element=H%120`으로 변환한다. **VRF는 slice당480B**다. `Time=H%120`, `Packet=1#4`인 곱셈이 행별 scalar를 읽으므로 `[H%120,1#8]`로 부풀린 DM/VRF는 만들지 않는다. Schedule의 전체 physical VRF reservation은245,760B=512×480B이며, 유효 row group leader는32개다.

Graph의 동일 Main node에서 다음 순서를 직접 확인했다.

```text
원래 BF16 x / FP8 weight→BF16 lookup
→ 기존 BF16 DPE/F32 accumulation
→ GReduceAddF ratio16
→ FpMul0: row scale VRF
→ 첫 BF16 cast
→ 4행 BF16=8B commit
→ BF16 typed H480×8 gather
→ 기존 RMS/gamma/residual
```

Weight·input·F32 sum 순서는 E53과 같다. 달라진 것은 F32 중간 DM·copy를 거치던 scale 곱셈의 위치와 이후 copy dtype다. F32 scale 곱셈 뒤 BF16이라는 oracle 반올림 경계도 유지된다. `[h=1920*c+120*g+r]`의 scale 주소3,840개, row commit 주소와 Qs4096 좌표를 [검사](verification.json)했다. `[Dummy2,H%1920]`의 cluster와 H row를 임의 permutation하는 reshape는 없다. 이 source/graph 검사는 hardware 정확성이나 모든 유한 입력의 동등성 증명을 대신하지 않는다.

첫 [실패 소스](projection.padded-fetch-failed.rs)는 scale fetch에 `H%8#16`을 넣었다. F32 변환 후 collect가 추가 `1#2` Time을 요구해 `Collect time mismatch`로 실패했다. 최종 소스는 실제8개 BF16 값만 `H%8`로 읽는다. 불필요한 packet padding만 제거했으며 두 번째 compile에서 compact VRF와 fusion이 모두 통과했다. 두 로그와 실패 소스는 보존했다.

## 전체 비용이 증가한 이유

| 항목 | E53 | 후보 | 차이 |
|---|---:|---:|---:|
| 전체 makespan | 27,239 | 27,393 | +154 |
| Main/Vector 각각 | 7,178 | 6,777 | −401 |
| DMA 구간 합집합 | 18,229 | 18,238 | +9 |
| Sub | 3,036 | 2,946 | −90 |
| Fused lookup+dot+reduce(+scale) | 5,063 | 5,063 | 0 |
| Channel-scale DMA | 546 | 579 | +33 |
| Channel-scale Sub | 383 | 293 | −90 |
| Result gather | 470 (F32) | 446 (BF16) | −24 |
| 별도 channel-scale Main | 401 | 0 | −401 |

기존 scale DMA `[16,236,16,782]`는 dot `[16,157,21,220]`과 겹쳤다. 후보는 scale이 dot의 필수 입력이 되어 compiler가 DMA `[1,003,1,582]`를 먼저 실행한다. LUT·x·weight 전송이 그 뒤로 밀리고 dot은 `[16,736,21,799]`가 된다. 즉 dot 시작이579 cycles 늦어지고 뒤쪽의401-cycle Main과24-cycle gather만 줄어 **579−401−24=154**가 남는다. 이 schedule의 dependency와 배치에 대한 설명이며 NPU에서 동일한 지연이 실측됐다는 뜻은 아니다.

Scale DMA의 `src0=[H_120=2:240,H_240=2:480,None=1:240]`는 각 leader에 연속240B를 보낸다. 따라서 giant padded scale materialization 없이 합법적인 fusion은 가능하지만, 현재 schedule에서는 준비 전송을 충분히 겹치지 못한다. [static-comparison.json](static-comparison.json)에 node lifetime을 저장했다. 독립 DMA/Main 비용을 합산해 전체 개선으로 해석하지 않는다.

## 재현과 동결

Docker `/tmp/moa-output-bf16-fused-channel-scale-e53`와 그 내부 target/kernel output을 사용했다. `CARGO_HOME=/home/moa/.cargo`, offline mode, 고정 SDK0.8.1/toolchain과 E53 frozen Cargo.lock으로 다음 명령을 두 번 실행했다.

```bash
cargo furiosa-opt compile ops::sliding_attention_output --exact \
  --dump-schedule schedules/candidate.json --dump-graph schedules/candidate-graph.json
```

기준27,239는 E53과 같은 `35947381…` image의 [기존 exact schedule](../../artifacts/output-bf16-activation-e47/candidate.json)이다. Source SHA는 `f7e0c0a63169e8f30e78327a9686d6d9c910a70759cd0bf5ef5d31791239b08a`, 후보 image는69,376B / `4247aef7bb8e84b822533cc650a8327809d8a7502ad844919ebc3dafb66939de`다. [evidence.json](evidence.json), [patch](candidate.patch), [검증 코드](verify_candidate.py), [첫 로그](../../artifacts/output-bf16-fused-channel-scale-e53/attempt-1.log), [성공 로그](../../artifacts/output-bf16-fused-channel-scale-e53/attempt-2.log)를 보존했다. Live source/target·Git·NPU 변경은 없다.
