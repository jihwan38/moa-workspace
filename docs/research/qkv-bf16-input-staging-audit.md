# QKV BF16 입력 staging: E54와 channel-scale fusion 후보

2026-09-21에 frozen source, 두 exact schedule/graph, 설치된 SDK0.8.1을 읽었다. **H3840 gather와 ring broadcast는 이미 한 번만 실행된다. 실제 중복은 같은 BF16 activation의 Q용/KV용 TRF 적재 두 번이다.** 기존 `Dummy2/Dummy256` 축을 쓰는 공통 TRF 한 개로 이 중복을 제거하는 경로는 API와 좌표 측면에서 설계할 수 있다. 다만 현재 TRF는 Q weight/lookup보다 먼저 준비되어 있으므로, 1,223-cycle 적재를 하나 없애도 전체 makespan이 줄어든다고 예측할 근거는 없다. 이번 작업은 읽기 전용 분석이며 새 source 후보, compile, NPU 실행은 없다.

## 조사 대상과 실제 호출 경로

- [Clean E54 source](../../moa-dev/experiments/artifacts/e54-bf16-qkv-clean-source/src): `ops.rs:38–86`의 `sliding_project_qkv`, `device/shared/rmsnorm.rs:203–281`, `device/sliding/projection.rs:7–173`.
- E54와 동일 QKV image `57a52d1a…`의 [exact schedule](../../moa-dev/experiments/artifacts/qkv-bf16-f32-scale-e53/candidate.json) / [graph](../../moa-dev/experiments/artifacts/qkv-bf16-f32-scale-e53/candidate-graph.json): makespan52,282.
- [Fused source](../../moa-dev/experiments/candidates/qkv-bf16-fused-channel-scale-e54/projection.rs), [schedule](../../moa-dev/experiments/artifacts/qkv-bf16-fused-channel-scale-e54/candidate.json), [graph](../../moa-dev/experiments/artifacts/qkv-bf16-fused-channel-scale-e54/candidate-graph.json): image`5bdab7ea…`, makespan51,293, 이 조사 시점 NPU 미검증.

실제 inference caller는 `src/host/runtime.rs:401–412`의 `Runtime::sliding` 경로이고, 공식 host 검사는 `src/bin/test_kernels.rs:481–513`에서 같은 public op를 launch한다. Private `sliding::projection::project_query`와 `project_key_value`를 직접 부르는 곳은 이 `sliding_project_qkv` 본문뿐이다. 이름이 같은 full-attention helper는 다른 모듈이며 조사 범위 밖이다.

Public 입력 `x:HbmTensor<bf16,H>`는 다음 순서로 흐른다.

```text
x HBM
→각 cluster에서 첫8개 slice에 H480×8 shard
→normalize_sharded_inter_slice: square/local sum÷3840
  →inter-slice sum→EPS AddF→sqrt
  →x÷rms×gamma→BF16 cast
→TDMA: 각32-slice ring의 leader8개에 H3840 전체를 조립·복제
→Main CustomBroadcast32: cluster마다256개 slice에 같은 BF16 H3840
→같은 DM의 Q mapping / KV mapping view
→Sub Q TRF 한 개 + Sub KV TRF 한 개
→Q dot, 같은 KV TRF를 공유하는 K dot/V dot
```

K와 V는 이미 `project_key_value` 내부의 `x_trf` 한 개를 공유한다. Q만 별도 `project_query` 안에서 같은 값을 다시 적재한다. Fused 후보의 입력 staging·RMS source는 E54와 byte-identical이다. 따라서 아래186-cycle 차이는 입력 산술 변경이 아니라 전체 schedule의 다른 DMA 배치에서 나온다.

## 정확한 구간과 비용

단위는 static cycles이며 `[begin,end)`다. 아래 표의 duration을 단순 합한 값은 elapsed runtime이 아니다. Gamma 준비가 RMS 계산과 겹친다.

| 단계 | E54 interval | Fused interval | 각 duration |
|---|---|---|---:|
| x HBM→H480 shard, DMA | 1,003–1,623 | 동일 | 620 |
| x²·local sum÷3840, Main | 1,623–2,024 | 동일 | 401 |
| gamma HBM→shard, DMA | 1,623–2,243 | 동일 | 620 |
| gamma BF16→F32 VRF, Sub | 2,243–2,626 | 동일 | 383 |
| inter-slice sum+EPS+sqrt, Main | 2,024–4,603 | 동일 | 2,579 |
| rms scalar→VRF, Sub | 4,603–4,870 | 동일 | 267 |
| x÷rms×gamma→BF16 shard, Main | 4,870–5,271 | 동일 | 401 |
| shard→8 ring leader H3840, TDMA | 5,457–8,109 | 5,271–7,923 | 2,652 |
| ring32 BF16 broadcast→DM, Main | 8,109–16,052 | 7,923–15,866 | 7,943 |
| KV DM→TRF, Sub | 16,052–17,275 | 15,866–17,089 | 1,223 |
| Q DM→TRF, Sub | 17,275–18,498 | 17,089–18,312 | 1,223 |

이 입력 경로의 resource duration 합은 Main11,324(3,381 RMS+7,943 broadcast), Sub3,096(650 RMS+2,446 TRF), DmaLoad1,240, DmaStos2,652다. 이것은 프로그램 전체 resource 합이 아니다.

**x DMA 시작부터 Q와 KV TRF가 모두 준비될 때까지의 elapsed 구간**은 E54에서17,495, fused에서17,309다. 두 버전 모두 pre-RMS의 순차 경로는4,268 cycles이다. E54는 BF16 shard 완료5,271 뒤 Q scale DMA가5,457까지 TDMA를 점유해186 기다린다. Fused는 그 대기 없이 gather한다. 즉 입력 값과 개별 stage 비용은 같고186-cycle scheduling gap만 사라졌다.

Graph의 broadcast `RenegadeCommand#O10`은 각 output slice `s`가 source anchor `32*floor(s/32)`를 읽는 bitmap을 명시한다. Leader에는 TDMA가 미리 모든 H3840을 쓴다. 원래8개의 H480 shard를 전체 H로 **reshape만 하는 것은 불가**하다. 실데이터 gather가 필요한 이유다. 마찬가지로 leader16개(두 cluster 합계)를512개로 reshape해도 실제 복제되지 않는다.

## DM·VRF·TRF lifetime과 용량

정규화 입력/출력 shard는 slice당480 BF16=960B다. 각 cluster의8개 slice만 유효해 실제 BF16 값은 두 cluster 합15,360B이지만, 공개 schedule은 전체512 physical slice를 적용해 각491,520B로 표시한다. Leader tensor는 유효16×7,680B=122,880B이고 schedule 예약은3,932,160B이다. 이 차이는 padding을 실제 데이터로 취급하라는 뜻이 아니다.

Broadcast 뒤 full-copy DM은512×7,680=3,932,160B이다. Schedule의 named tensor 하나만 보면 broadcast 완료에서 lifetime이 끝나지만, 같은 주소의 `reshape/view` 별칭을 함께 읽어야 한다. E54는 tensor59→62/63/64가 같은 주소16,777,216을 써서 전체 구간8,109–18,498까지 살아 있다. Fused는59→73/74/75가 주소33,554,432을 공유해7,923–18,312까지다. 별칭을 다른 DM 복사로 세지 않았다.

Pre-RMS gamma VRF는 slice당480 F32=1,920B, rms VRF는32B다. Mean-square의 stash 예약은 slice당1,024B이며1,623–2,024 동안만 살아 있다. Gamma VRF는2,243–5,271, rms VRF는4,603–5,271이어서 입력 RMS의 두 명시적 VRF가 겹칠 때는1,952B다. 프로그램의 다른 미리 적재한 scale/RoPE VRF는 별도이며 이 숫자는 전체 VRF peak가 아니다. Fused의 channel-scale VRF는 projection당256B/slice이고 이 분석의 공통-TRF 경로는 새 VRF를 필요로 하지 않는다.

| Activation TRF | E54 lifetime | Fused lifetime | Schedule 예약/slice |
|---|---|---|---:|
| KV 공용 | 16,052–44,629 | 15,866–45,704 | 32,768B |
| Q 별도 | 17,275–28,765 | 17,089–31,963 | 32,768B |

TRF 하나의 논리 데이터는 `Lane1×H3840×BF16=7,680B`다. Graph도 각각3,932,160B(512 slices)를 나타낸다. 하지만 schedule allocator가 각16,777,216B, 즉 **slice당32KiB를 예약**했으며 두 동시 lifetime에서는64KiB 전부가 예약된다. 이는 관측된 allocation이다. 임의로7,680B씩 수동 배치할 수 있는 API가 있다는 뜻은 아니다. SDK `TRF_CAPACITY_BYTES`는65,536이며 주소·bank 선택은 handle에 없다.

공통 TRF 한 개를 쓰면 제거 목표는 현재 Q용32KiB 예약과1,223-cycle `StoTrf` 한 개다. 공통 TRF lifetime은 기존 KV TRF처럼 세 dot이 모두 끝날 때까지 유지해야 한다. 현재 KV가 이미 첫 적재부터 마지막 dot까지 살아 있으므로, longest existing activation lifetime을 더 길게 만드는 설계는 아니다. 새 compile에서 실제 allocator 예약과 scheduling이 어떻게 바뀌는지는 미확인이다.

## 하나의 구체적인 경로: 공통 named-layout TRF

단순 `&q_trf`를 K/V에 전달할 수는 없다. SDK `engine/contraction/outer/mod.rs:124–127`의 `contract_outer`는 stream과 TRF의 **동일한 Chip/Cluster/Slice 타입**을 요구한다. Q는 `Cluster=Qs/2048, Slice=[Qs/64%32,Qs%8]`, K/V는 `Ps/1024,[Ps/32%32,Ps%8]`로 서로 다르다. SDK `tensor/memory.rs:1391–1437`에는 TRF `zero/view/view_mut`만 있고 공개 `reshape`나 `Clone`이 없다. `TensorView::reshape`는 generic view를 반환하므로 `&TrfTensor` 인자를 대신하지 못한다.

따라서 TRF를 다시 해석하는 대신 **세 DPE stream을 처음부터 하나의 공통 named-layout으로 맞춘다.** 새 축을 만들지 않고 기존 `Dummy2`, `Dummy256`을 쓴다.

```rust
type InputClusters = m![Dummy2];
type InputSlices = m![Dummy256];
// 현재 full-copy BF16 DM의 두 cluster/256 slices는 모두 실제 동일값을 갖는다.
let x_view: DmTensorView<'_, bf16, Chip, InputClusters, InputSlices, m![H]> =
    unsafe { x.view().reshape() };
let shared_x_trf: TrfTensor<bf16, Chip, InputClusters, InputSlices, m![1], m![H]> =
    device.sub.begin(x_view)
        .fetch::<m![H / 16], m![H % 16]>()
        .collect::<m![H / 16], m![H % 16]>()
        .to_trf();
// private Q/K/V projection helpers borrow this same handle.
```

이는 **설계 예시이며 compile한 코드가 아니다.** 이후 필요한 변환은 다음 하나의 연결된 경로다.

1. Weight HBM→DM은 현재 검증된 Q/KV typed mapping을 유지한다. DM view의 physical cluster/slice만 `Dummy2/Dummy256`으로 이름을 바꾼다. Q local `Qs/8%8`, K/V local `Ps/8%4`와 H 원소 순서는 그대로다.
2. E55 fused scale Sub는 현재 검증된 inverse InterTranspose를 그대로 수행한다. 생성된 VRF를 SDK의 공개 `VrfTensor::reshape`로 같은 named cluster/slice에 **move**한다. Local row/packet padding은 그대로 둔다.
3. 공통 TRF로 기존 BF16×BF16 DPE, 기존 FP32 reduction, 기존 channel scale, 첫 BF16 cast를 수행한다. Raw 결과 DM의 cluster/slice 이름을 원래 QueryRows/KvRows로 즉시 복원한 뒤 현재 BF16 output InterTranspose/head gather를 그대로 사용한다.

Public op/harness는 바꿀 필요가 없고 private helper 인자와 그 caller만 연결한다. Fused 후보가 아닌 E54에 적용하면 scale은 원래 canonical 결과 뒤에 남으므로 scale VRF relabel 단계도 필요 없다.

이 relabel의 안전조건은 “축 이름이 달라도 크기만 같으면 된다”가 아니다. Q 행 `q=2048*c+64*g+8*r+s`, KV 행 `p=1024*c+32*g+8*r+s`에 대해 원래 `(cluster=c,slice=8*g+s,local=r,h)`와 neutral `(Dummy2=c,Dummy256=8*g+s,local=r,h)`가 같은 물리 위치를 가리켜야 한다. 복원도 같은 tuple에 적용한다. Q4096/K2048/V2048행의 이 양방향 대응을 Python 정수 전수 검사로 확인했다. H는 순서를 바꾸지 않으며 K/V의 valid4와 output Switch의4#8 padding도 그대로다. Neutral cluster/slice는 **named 축**이어야 한다. Weight가 서로 다른 slice/cluster 값을 가진 상태에서 `m![2]`나 anonymous broadcast32로 바꾸면 동일-copy 의미를 잘못 주장하게 된다.

공개 API 근거는 SDK0.8.1 `tensor/memory.rs:1165–1193`의 같은 wire-order `DmTensorView::reshape`, `memory.rs:967–999`의 소비형 `DmTensor::reshape`, `memory.rs:1483–1525`의 `VrfTensor::reshape`, `engine/collect.rs:74–86`의 `collect→to_trf`다. Lower verifier `furiosa-opt-lower-0.8.1/src/verify/engine/collect.rs:179–242`는 Lane1과 `[H/16,H%16]→H`, BF167,680B의 capacity 검사를 제공한다. 이 API 근거와 좌표 증명은 새로운 조합의 backend compile·NPU 통과를 대신하지 않는다.

## 실제 성능 기대의 한계

Q 입력 TRF 완료는 E54 18,498 / fused18,312다. Q weight DMA 완료는21,492 /24,690이며 lookup-table 준비가 추가되어 Q dot은23,702 /26,900에 시작한다. 따라서 TRF 준비 뒤 각각 **5,204 /8,588 cycles의 여유**가 있다. 기존 KV TRF 역시 첫 KV dot보다 E54 12,116 /fused15,405 cycles 먼저 준비된다. 중복 적재 하나를 지워도 그1,223이 현재 dot critical path에서 그대로 사라지는 구조가 아니다.

현재 확인 가능한 가치는 Sub 작업 한 개와 동시에 예약되는 TRF 한 개를 줄이는 것이다. Whole-makespan 개선 여부는 아직 모른다. H3840 gather2,652와 ring broadcast7,943은 중복이 아니므로 이 공통-TRF 설계에서 제거했다고 계산하지 않는다. `Switch→collect→to_trf` 자체는 generic SDK typestate가 노출하지만, 현재 named output/공통 TRF와 결합한 CustomBroadcast의 lowered bitmap·allocation은 검증하지 않았으므로 이를 또 다른 즉시 실행 후보로 확대하지 않았다. 새 compile 없이 “broadcast도 무료로 합칠 수 있다”고 결론내릴 근거는 없다.

이번 조사에서 쓴 SDK root는 Docker `/home/moa/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/{furiosa-opt-std,furiosa-opt-lower}-0.8.1`이다. 문서 한 개만 작성했으며 source/artifact/live/Git/NPU는 변경하지 않았다.

이 읽기 전용 감사 이후 [공통 TRF 후보](../../moa-dev/experiments/candidates/qkv-bf16-shared-trf-e55/README.md)를 별도 scratch에서 exact compile했다. 첫 시도는 통과했지만 전체51,293→52,562 cycles(+2.47%)로 회귀해 미채택했다. 의도한 Sub1,223 감소와 TRF 예약64→32KiB/slice는 확인됐으나 K lookup table 준비가 늦어져 이후 경로가1,269 cycles 밀렸다. 이 후속 결과는 장치 검증이나 전체 속도 개선을 뜻하지 않는다.
