# M02 — O projection의 K-half cluster 소유권

작업 branch `opt/m02-cluster-k`, 실제 출발 commit
`ecdeaca8187504e833a415731f626a3fa1ef3d5e`. M01 기록을 main/origin에 통합한 뒤 시작한다.
성능 비교는 `baseline/e104.json`; 실행 source는 새 `project_output` 전체 경로다.
QKV·FFN·공개 입력/출력·정확도 기준·fixture/host는 유지한다.

**완료: 두 완성 경로 모두 compile·기본 NPU 정확도 PASS, 성능 회귀로 미채택.**
Arena95914에서 E10451010/50901 대비 v3 78981(+55.00%), v4 89203(+75.06%) cycles.
실행본은 E104 유지, 공식 제출 없음. [최종 판정](decision.json).
아래 첫 설계와 수정 이력은 최초 가설과 실제 결과를 구분해 보존한다.

## 사전 설계와 중복 확인

기존 E104의 cluster H-half / slice16행×16열 / local120×256을 제거한다.
cluster `c<2`는 K 반쪽, slice `s<256`은 interleaved H 행을 소유한다:
`h=256*t+s`, `k=2048*c+u`, `t<15`, `u<2048`.
HBM FP8 offset=`4096*h+2048*c+u`. 각 weight는 정확히 한 번 포함된다.

15×2048 tile 자체는 과거 `output-column-sweep/projection.2.rs`에도 있었다.
이번 차이는 **K-owned clusters + H interleave + leader 입력 방송 + F32 cluster 합산**이다.
과거 두열 구현은 H-owned clusters와 두 slice 열, BF16-before-scale을 사용했고
compile-only였다. 그때 weight 적재 정적 시간은 줄지 않았다. 큰 segment만으로
빨라진다는 주장은 하지 않는다. M01의120×256 배치도 계승하지 않는다.

```text
x HBM -> 각 cluster의 ring32 leader DM -> Switch 방송 -> BF16 TRF
W HBM -> 각 slice FP8 15×2048 DM -> BF16 lookup+dot
      -> F32 15개, 각각8B slot -> cluster 간 H480×8 gather, padding 유지
      -> TU로 두 F32씩 compact -> 두 H partial DM
      -> 첫 partial stream + 둘째 partial VRF -> F32 AddF -> F32 channel MulF
      -> 첫 BF16 -> 기존 post-RMS/residual -> HBM
```

| 값 | 소유권·배치·논리 공간 | lifetime / 소비자 |
|---|---|---|
| x leader | cluster당8개 ring32 leader, 각각2048BF16; 총65,536B | 방송 종료 / Switch |
| x replica | 각 cluster256 slices, slice당4096B DM/TRF; 총2MiB | 15행 dot 종료 / DPE |
| W | slice당30,720B FP8 DM; 총15MiB; source row segment2048B | dot 종료 / Main lookup+DPE |
| padded partial | slice당15×8B; 유효 F32 합계30,720B, 저장61,440B | gather 완료 / TDMA |
| gathered partial | cluster0의8 epilogue slices에각2×480×8B | compaction 완료 / Main TU |
| compact partial | slice당2×480F32=3840B; 총30,720B | 최종 합·scale / Main+Sub |
| 둘째 partial+scale | 각각480F32 VRF, slice당합계3840B | 최종 VE 종료 |
| scaled output | BF16 H480×8; 기존 반환 mapping | post-RMS/residual |

HBM의 K-high bit를 Dummy2로 이름만 바꾸는 reshape는 wire order를 보존한다.
방송 후 Dummy256→H%256 reshape도 물리 복제를 완료한 후 owner 이름만 바꾼다.
Partial gather와 compaction은 실제 DMA/TU 작업이다. 패딩을 reshape로 없애지 않는다.
각2048B weight 구간은256B 정렬, BF16 input 구간은4096B, F32 scalar commit은8B slot이다.
실제 DMA packet 제한·주소 조건과 TRF/VRF allocation은 compile에서 확인한다.

16열 stream 합산을 제거하지만 E104는 원래 partial 전체를 DM에 쓰지 않았다.
새 partial write/gather/compaction/read와 cluster 이동은 **추가 작업**이다.
입력 replica payload는262,144→2,097,152B(8배), leader 적재 이후 방송도 유료다.
Weight segment는61,440×256B→7,680×2048B이며 DMA 명령/실제 transaction 수가 아니다.
두 F32 partial을 AddF 후 MulF로 처리해 합산 결과의 별도 DM write/read를 피한다.
SDK stage 순서상 intra-slice reduce 후 FP Mul은 같은 stream에서 할 수 없으므로
대신 두 partial의 binary AddF를 사용한다. TU transpose 이후 VE/VRF 연결도 지원되는
position이 아니므로 compaction은 DM 경계를 갖는다. 같은 slice VRF의 두 operand
수명과 Sub의 준비 작업도 스케줄 비용에 포함한다.

## 확인할 경로와 종료 기준

`src/ops.rs`의 `sliding_attention_output` → `project_output` → 기존 post-RMS·residual.
기본3입력의 전체 O 정확도와 시간, QKV/FFN image 불변을 검증한다.
16×256에서2×2048로 F32 합산 순서가 달라져 기존 정확도 검증을 상속하지 않는다.
전체 O가 실행되면 같은 Arena allocation에서 E104 전/후 control과 비교한다.
회귀 또는5% 미만 개선이면 이 prototype을 닫고 작은 타일 sweep은 하지 않는다.
5% 이상 개선은 승격 기준이 아니며 독립 반복·확장 정확도를 통과한 뒤 판단한다.
구체적인 source padding/axis/API 오류는 원인과 수정 snapshot을 남겨 고친다.

## 근거

- [전체 설계 조사](../../../../docs/research/rngd-dataflow-candidates-20260925.md)
- [DMA 정렬·cross-cluster mapping](https://developer.furiosa.ai/furiosa-opt/book/moving-tensors/dma-engine.html)
- [Switch의 ring 방송과 setup](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/switch-engine.html)
- [BF16 outer contraction](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/contraction-engine/outer.html)
- 설치 SDK0.8.1 `src/engine/mod.rs`: `CanApplyToVrf`, `CanApplyVectorInit`, `CanApplyTranspose`.
- 설치 SDK0.8.1 `src/engine/vector/stage/markers.rs`: FP → intra-slice reduce 순서.
- 실제 재사용 source: `output-column-sweep/projection.2.rs`,
  `c14-ffn-l-pipeline/bf16-stage-compact-v4/mlp.rs::finish_down`,
  `qkv-fp8-split/projection.rs`의 AddF→MulF, `device/layout.rs`의 CustomBroadcast.

빌드·NPU 결과는 아래에 후속 기록한다. 이 문단까지는 최초 compile 전 설계다.

## 구현 수정과 NPU 전 추가 진단

1. `khalf-v1`: Switch bitmap과 output slice 좌표 불일치. 새로 방송되는32개 lane을
   `Dummy256`의 빠진 좌표로 표시한 source 오류다. 기존 ring32 구현처럼
   `Dummy256/32, 32`로 ring ID를 보존하고 anonymous 복제 lane을 사용해 v2에서 수정했다.
2. `khalf-v2`: H%256 slice/H/256 local에서 H/480 slice/H%480 local로 직접 DMA할 때
   `cannot factorize common axis H`. 256과480 경계가 나눠떨어지지 않는 표현 문제다.
3. `khalf-v3`: 각 K cluster 안에서 padded full-H를 모아 압축한 뒤, compact H를
   cross-cluster H480×8로 분배한다. **완성 O compile PASS, 정적37413**.
   padded gather4658 → 한 slice/cluster에서 compaction4105 → compact gather801이
   직렬로 이어지는 새 비용을 확인했다. padding 압축 이전 cluster별 공간은30720B,
   이후15360B. 전체로는 각각61440/30720B다. v1/v2와 달리 추가 compact DMA가 있다.
4. `khalf-v4`: 위 비용을 확인하고 **NPU 전**에 producer와 consumer 경계를 맞춘다.
   `s=32*g+r`, `h=480*g+32*t+r`, `g<8,r<32,t<15`.
   `RowSlices=m![H/480,H%32]`, local row=`H/32%15`.
   K cluster 반쪽·15×2048 tile·BF16/F32 순서는 유지한다. source와 destination에
   공통 H/480이 있어 padded partial을 바로 epilogue로 모을 수 있게 설계했다.
   압축은 cluster0의8 slices에480행씩 분산한다. 추가 full-H 집결·DMA를 제거한다.

v3와v4는 독립 architecture 두 개가 아니다. M02의 경계 변환을 구현하는 두 경로다.
**제한된 진단 질문:** v3의 한 slice 집결/압축을 v4의 출력 묶음별 배치로 없애면
완성 O에서 이 비용을 회수하는가? 같은 K ownership·tile·dtype·입력·후처리에서
E104 전/후 control 및 두 완성 경로를 한 allocation에서 검사한다.
v4가 실행되면 그 결과로 다음 설계 판단을 하며, 회귀/5% 미만은 더 sweep하지 않는다.
정적 interval은 겹칠 수 있어 합을 전체 runtime 절감으로 사용하지 않는다.

## 정적 일정과 메모리 근거

v3/v4 모두 exact compile과 host package를 만들었다. QKV/FFN image는 source origin의
E104 image와 같고, O image는 exact compile과 host에 포함된 image가 같음을 검사했다.
`structural_build.py --plan` 결과, recipe, 각 시도 source와 실패/성공 로그를 보존했다.
[재현 가능한 분석](analyze.py), [원본 일정과 image hash 대조](static-cost.json),
[좌표·source 범위·build/package 검증](verification.json).

| 정적 구간 cycles | E104 | v3 full-H 집결 | v4 H480 묶음 |
|---|---:|---:|---:|
| Weight HBM→DM | 13,383 | 13,512 | 13,512 |
| 입력 leader/replica HBM→DM | 933 | 646 | 646 |
| 입력 방송 | 없음 | 4,359 | 4,359 |
| 입력→TRF | 327 | 775 | 775 |
| Dot 및 같은 stream의 연산 | 5,063 | 5,063 | 5,063 |
| Dot 후 첫 DMA | 470 | 4,658 | 3,314 |
| 별도 partial 압축 | 없음 | 4,105 | 1,225 |
| 압축 후 추가 DMA | 없음 | 801 | 없음 |
| 둘째 partial→VRF | 없음 | 503 | 503 |
| Channel scale / M02 sum+scale | 401 | 401 | 401 |
| 전체 makespan | 27,239 | 37,413 | 32,382 |

Weight source 구간은8배 길어졌지만, 이 compiler 일정의 weight 적재는 줄지 않았다.
E104의16열 합산을 없애도 해당 dot stream은5063으로 같다. 원래 stream과 겹쳐
처리되던 합산을 빼는 것만으로 enclosing interval이 짧아지지 않은 것이다.
v4는 v3의 집결/압축 병목을 줄여 전체 일정이13.45% 짧아졌지만 E104 대비+18.88%다.
방송은 weight 적재와 겹치므로4359를 별도 runtime 손해로 더하지 않는다.
이는 compiler schedule 관측이며 NPU stall 분해나 실측 속도 비율이 아니다.

위 설계의61,440B 등의 값은 **활성 slice에 있는 유효값/slot 양**이다.
Sparse cluster/slice mapping의 실제 compiler allocation extent와 다르다.
예를 들어 v4 gathered partial extent는3,932,160B, scale/second VRF는각983,040B이고,
TRF extent는E104/v3/v4 모두16,777,216B다. v3 full-H compact partial extent는7,864,320B다.
이 값을 더해 peak 사용량 또는 실제 전송량이라고 하지 않는다. Shape·주소·수명은
`static-cost.json`의 `compiler_tensor_allocations`와 원본 schedule에 함께 보존했다.

## 실제 NPU 결과와 종료

Arena **95914**, 한 allocation에서 별도 process4개를 순차 실행했다.
공식 기본 host/fixture는 변경하지 않았고, 업로드된 archive를 다시 내려받아
실제 실행한 package의 모든 파일 hash를 로컬 검증 package와 대조했다.

| 경로 | O 중앙 cycles | 앞뒤 E104 중앙값 대비 |
|---|---:|---:|
| E104 before | 51,010 | 비교군 |
| E114 / v3 full-H 집결 | 78,981 | **+55.00% 느림** |
| E115 / v4 H480 묶음 | 89,203 | **+75.06% 느림** |
| E104 after | 50,901 | 비교군 |

네 process의 총60비교 모두 PASS. 각 후보 O는3입력·11520출력 원소를 검사했다.
두 후보 모두 O run1의 출력에 로그상 최대절대오차0.00391이 있고 기존 tolerance를
모두 통과했다. Run0/2는 로그 출력 정밀도에서0이다. E104와 bitwise 같다는 뜻은 아니다.
확장 cancellation/boundary 입력과 독립 allocation 반복은 회귀로 중단해 실행하지 않았다.
[E114 원본 보고](../../../../docs/experiments/e114-m02-full-h-gather.json),
[E115 원본 보고](../../../../docs/experiments/e115-m02-grouped-interleave.json),
[원본 NPU 로그](evidence/arena-95914.log).

**정적 순서와 NPU 순서가 뒤집혔다.** v4는 v3보다 compiler makespan이13.45% 짧지만,
NPU에서는12.94% 길었다. v4는 H row mapping과 partial 전달 경로를 함께 바꾸었으므로
whole-O 측정만으로 어느 bank/channel·통신·동기화가 원인인지 분리할 수 없다.
따라서 “병목을 해결했다”거나 “NPU의 특정 stall 때문”이라고 결론 내리지 않는다.
확인된 것은 두 완성 경로의 큰 회귀, 그리고 짧아지지 않은 compiler weight/dot 구간과
추가 partial 의존성이다. NPU 회귀의 정확한 비용 분해는 미측정이다.

규칙에 따라 두 경로를 닫고 타일·ring size sweep은 하지 않는다. 이 결과는
모든 K-owned cluster 설계의 실패나 E104의 최적성을 증명하지 않는다.
재시도하려면 이 전체 경로의 손해를 없애는 구체적인 배치/전송 변화와,
정적/NPU 역전을 구분하는 제한된 진단 근거가 있어야 한다. 다음 미실행 후보는 M03이다.

검증 명령(저장소 root):

```bash
python3 moa-dev/experiments/candidates/m02-cluster-k/verify.py --check
python3 moa-dev/experiments/candidates/m02-cluster-k/analyze.py --check
python3 moa-dev/experiment_index.py --check --check-live
```

Source snapshot4개, 서로 다른 source의 exact compile4회와 package 재compile2회,
host build2회, Arena allocation1회다. 새 독립 아키텍처는1개이며 완성 경로는2개다.
