# M06 v4/v5 read-only mechanism audit — 2026-09-27

판정: **성능 후보 재개 보류. 특정 실패 경계와 검증된 수리 근거는 아직 없다.** 공통 입력 준비 경로와 weight DMA의 정적 중첩은 확인했지만, bank starvation, scalar Max, direct-packing aliasing, compiler bug 중 하나를 원인으로 확정할 증거는 없다. 이 보고서는 재실행 성공/성능 개선 보고가 아니다.

기존 자료와 Docker SDK를 읽어 수행한 감사이며, M06의 새 compile/Arena 실행이나 source 변경은 하지 않았다. 보고서와 원자료 발췌를 연구 기록으로 보존한다.

## 1. 런타임 실패가 실제 말해 주는 것

- `moa-dev/experiments/candidates/m06-sharded-input-pipeline/evidence/arena-104566.log:54–60`: v4 FFN 진입 뒤 `src/bin/test_kernels.rs:622:6`에서 `rank 1 ended a launch with 4: the launch ran past its time limit`.
- `.../evidence/arena-104588.log:54–60`: v5는 같은 지점에서 **rank 0**, 같은 code4.
- 두 로그 모두 candidate FFN의 첫 correctness 비교/유효 cycle count가 없다. 앞의 QKV/O, 뒤의 원본 M03 및 E104는 통과한다. 해당 device allocation 자체가 전부 고장 났다는 설명은 맞지 않는다.
- SDK 컨테이너 `/home/moa/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/furiosa-opt-ipc-0.8.1/src/status.rs:8–9,24`: code4는 generic launch timeout이다. code3 task panic, code7 bad image, code8 bad syscall과 구별한다. **code4는 NoC bank starvation error code가 아니다.**
- test host `moa-dev/experiments/artifacts/m06-sharded-input-pipeline/m06-sharded-packed-v4/crate/src/bin/test_kernels.rs:602–623`: `.await.unwrap()`에서 중단되므로 HBM 출력 읽기/비교 이전임만 확정된다. 마지막 성공한 내부 task를 특정하지 못한다.
- 동일 host `:675–759`: span collector는 cluster/begin/end만 저장하고 내부 task 이름을 보존/출력하지 않으며 `event`는 무시한다. package `remote_entrypoint.sh:21`은 `FURIOSA_OPT_PROFILE=info`를 기본 활성화하지만 이 사실만으로 미완료 task profile이 로그에 남았다고 볼 수 없다.

**보존된 v4/v5 artifact에는 schedule, source, compile/build log, exact-kernel image가 있다. repository와 Docker `/tmp/moa-c02-c03/m06-*`에서 별도 graph/visa/summary/partial-profile dump는 찾지 못했다. pe_log/debugfs/NoC exception 상세도 두 Arena log에 없다.**

## 2. v4/v5에서 변하지 않은 실행 경계

Source 공통 부분:

- `sharded-packed-v4/ops.rs:228–235` 및 v5 동일: residual DM -> H480×8 shards -> `normalize_sharded_inter_slice` -> private FFN.
- frozen crate `src/device/shared/rmsnorm.rs:211–242`: per-slice mean-square -> inter-slice Add -> EPS/Sqrt.
- `rmsnorm.rs:245–276`: weight와 rms를 VRF에 staging하고 BF16 sharded normalized x를 생성.
- v4/v5 `mlp.rs:466–478`: H480 shard에서 abs/max/floor -> local_amax.
- `mlp.rs:479–484`: **새 eight-shard inter-slice Max를 scalar Slice에 commit.** M05는 full-H 한 slice에서 intra-slice Max(`local-prepare-v1/mlp.rs:466–478`)였으므로 이 두 연산의 런타임 검증은 같지 않다.
- `mlp.rs:487–497`: exponent mask와 2^-14 곱으로 restoration scale 생성.
- `mlp.rs:500–509`: scalar scale -> Main ring8 broadcast -> reshape -> Sub scale VRF.
- `mlp.rs:440,446–454`: quantizer ownership은 cluster0 slices0..7, 이후 두 cluster leaders DMA -> ring32 replication -> native TRF.

v4는 세 term을 packed view에 바로 commit한다(`:510–559`). v5는 separate hi/lo/third buffer와 Sub sum 및 세 packing DMA를 복구한다(`sharded-dma-terms-v5/mlp.rs:510–563`). 둘 다 실패하므로 **v4의 직접 packed commit만이 유일 원인이라는 주장은 증거와 맞지 않는다.** 다만 두 방식이 동일한 잠재 오류를 별도로 가질 가능성까지 배제하지는 못한다.

## 3. 동일한 정적 초반 schedule과 동시에 예약된 DMA

두 frozen schedule은 one-line JSON이다. 아래 숫자는 `instructions[index]`가 아니라 각 record의 `index`/`history.resourcelir` 식별자다. v4/v5는 instruction 번호가 다르지만 10268까지 다음 기간은 동일하다.

| 경계 | v4 / v5 instruction | 정적 begin→end |
|---|---|---:|
| RMS mean-square | O95 / O62 | 2837→3238 |
| RMS inter-slice Add+EPS+Sqrt | O107 / O86 | 3238→5817 |
| RMS scalar to VRF | O126 / O125 | 5817→6084 |
| BF16 normalized shards | O130 / O129 | 6084→6485 |
| local amax | O135 / O134 | 6485→6886 |
| global inter-slice Max | O140 / O139 | 6886→9448 |
| restoration scale | O145 / O144 | 9448→9730 |
| Main ring8 scalar distribution | O152 / O151 | 9730→10001 |
| quant scale to VRF | O157 / O156 | 10001→10268 |
| first term quantization | O162 / O161 | 10268→10669 |
| gate weight HBM DMA | O114 / O93 | **4302→28914** |

실제 device timestamps가 아니라 compile schedule이다. 따라서 이 중 하나에서 실제로 멈췄다고 결론낼 수 없다. Gate DMA가 common RMS/quantizer와 겹치도록 예약되어 있다는 정도만 확정된다. v4/v5 quant third end는12977/12819, 그 뒤 처음 weight decode 시작은 둘 다28926이다.

## 4. 공통 tensor lifetime에 보이는 것과 보이지 않는 것

두 schedule에서 아래 common allocations의 aggregate virtual address·shape는 같다. 이 주소를 512로 나누거나 하위 비트를 직접 읽어 실제 slice bank 주소라고 해석하지 않았다.

| 역할 | aggregate address | aggregate size | 정적 lifetime (v4; v5 차이) |
|---|---:|---:|---|
| RMS input shards | 12451840 | 491520 | 2388→6485 |
| RMS mean-square | 12976128 | 16384 | 2837→5817 |
| RMS result | 0 | 16384 | 3238→6084, reshape 포함 |
| normalized BF16 x | 12976128 | 491520 | 6084→12977; v5→12819 |
| local_amax | 12451840 | 16384 | 6485→9448 |
| global_amax | 12582912 | 16384 | 6886→9730 |
| scale | 12451840 | 16384 | 9448→40373; v5→35358 |
| ring8 quant_scale | 13500416 | 16384 | 9730→10268, reshape 포함 |
| scale_vrf | 1523712 | 16384 | 10001→12977; v5→12819 |

- normalized x는 세 term 소비가 완료될 때까지 살아 있고 scale_vrf도 그 마지막 소비까지 보존된다.
- RMS input→local_amax→scale, mean-square→normalized x가 같은 aggregate 영역을 재사용하지만 관측된 정적 lifetime은 겹치지 않는다. 즉 **이 표에서 명백한 use-after-free를 찾지는 못했다.**
- v4 packed-view 일부 tensor는 `instructions`에는 있지만 `tensors`의 독립 allocation 목록에는 없다. 완전한 alias/dependency graph 검증은 이 JSON만으로 불가능하다.
- gate packed allocation은 v4 aggregate주소50331648, v5 67108864로 다르지만 physical_pages는 둘 다[2,3], size29491200, lifetime4302→38189다. activation 준비 buffer들은 page0에 있다. page가 다르다고 bank conflict가 없다는 뜻도 아니고, 다른 aggregate 주소가 원인이라는 뜻도 아니다.

원자료 발췌와 SHA256는 [기계 판독 근거](m06-timeout-mechanism-20260927.json)에 보존했다.

## 5. Bank starvation 가설의 현재 상태

공식 문서: https://developer.furiosa.ai/furiosa-opt/book/moving-tensors/memory-performance.html 의 Bank Starvation 절. Main Fetch/Main Commit/Sub Fetch/Sub Commit/DMA 우선순위와 같은 bank 64회 연속 점유 제약, 4096-cycle NoC timeout을 설명한다.

이번 보존물에서는 **동시 DMA 예약**은 보이지만, 다음이 없다:

1. lowering된 Fetch/Commit/DMA sequencer의 실제 per-slice address 순서와 arbitration 상태.
2. 같은 bank 64회 연속 접근 또는 4096-cycle 무응답을 특정한 장치 기록.
3. timeout 전에 완료된 마지막 task와 대기중인 first consumer.

Source만 보면 local H480 처리들은 연속 payload를 순회한다. BF16 fetch는 H480/16=30 logical packets, F32 collect는60이며, scalar scale/Max stream의 Time은1이다. 이를 실제 hardware request 개수나 consecutive-bank 상한과 동일시할 수 없지만 **소스에 '같은 scalar DM을 64회 이상 반복 fetch'하는 명백한 패턴도 없다.** scale의 반복 사용은 VRF operand다. 따라서 padding/bank 변경을 즉시 수리책으로 선택할 근거가 없다.

## 6. 재개 판단과 다음에 필요한 증거

현재는 M06 최적화 재개를 보류한다. 이번 감사에서 성능 후보 재시작을 정당화하는 특정 수리는 찾지 못했다. 다음 중 하나가 생기면 판단을 갱신할 수 있다.

- **동일 frozen image를 유지한 task-level failure 관측:** timeout에서도 firmware task progress/pe_log 또는 partial trace가 남도록 수집한다. host instrumentation은 target calculation task bytes가 같음을 확인해야 한다. 같은 rank가 표시되는지보다 마지막 완료 task가 어느 producer/consumer 경계인지가 중요하다.
- **global Max 경계 격리:** sharded RMS+amax/global Max/scale 출력을 관측 가능한 tensor로 소비하여 실제 값과 종료를 확인한다. 단독 성공은 full FFN과 weight DMA가 겹칠 때의 동작을 입증하지 않으므로, 그것만으로 승격하지 않는다. 이 작업은 추가 compile/Arena를 요구하는 미실행 진단이며 이번 감사에서는 수행하지 않았다.
- **실제 sequencer/firmware 증거가 same-bank starvation을 특정:** 해당 fetch/commit bank 반복 또는 동시 DMA 배치를 바꾸는 국소 수리를 설계한다. 단순 dtype/shape/context sweep으로 대체하지 않는다.

현재 가장 정확한 라벨: `runtime timeout, common input-preparation / global-Max / scalar-broadcast composition unresolved`. `bank starvation confirmed`, `direct packing bug`, `compiler issue confirmed`, `NPU regression` 중 어느 것도 아니다.
