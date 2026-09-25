# M01 — column-major 적재에서 계산하고 F32 partial만 재배치

브랜치 `opt/m01-partial-remap`, 시작 commit `8109a27`.
사용자가 2026-09-25에 M01을 먼저 실행하도록 지정했다. 기존 조사 우선순위 M02/M03보다
이번 명시적 실행 요청을 우선한다. 조사 기록은 main에 fast-forward 통합·push했다.
공개 입력/정확도 계약과 채택 E104 실행 소스는 유지하며 격리 snapshot으로 작업한다.

**전체 compile·기본 NPU 정확도 PASS. 두 전송 경로 모두 실측 회귀로 미채택·종료했다.**
[사전 설계](../../../../docs/research/rngd-dataflow-candidates-20260925.md),
[논리 좌표·비용](../../../../docs/research/rngd-dataflow-accounting-20260925.json).

## 실제 결과

Arena **95636**, 2026-09-25. 동일 allocation의 네 process:
E104 before → DMA → Switch → E104 after. 각 process는 변경 없는 3입력 benchmark다.

| 경로 | O cycles(3입력 중앙값) | 앞뒤 E104 중앙값51140.5 대비 | 기본 정확도 |
|---|---:|---:|---|
| E104 before |51007|비교군|PASS|
| M01 DMA `dma-v2` / E112 |55608|+8.74% 느림|PASS|
| M01 Switch `switch-v2` / E113 |60677|+18.65% 느림|PASS|
| E104 after |51274|비교군|PASS|

총60개 출력 비교가 통과했다. 후보별 O3입력/11520출력도 원래 기준 그대로 PASS다.
독립 반복이나 확대 정확도 검사를 한 것은 아니다. 두 후보의 모든 O timing sample은
이번 양쪽 control sample보다 컸고, 5% 개선 screen을 통과하지 못해 추가 튜닝·반복을 하지 않았다.
실행 source와 채택본은 E104이며 공식 leaderboard 제출은 없다.

[E112 실측](../../../../docs/experiments/e112-m01-partial-dma.json),
[E113 실측](../../../../docs/experiments/e113-m01-partial-switch.json),
[전체 allocation](m01-remap-screen-results.json), [판정](decision.json),
[원문 로그](evidence/arena-95636.log).
Collector가 실제 업로드 archive를 다시 내려받아 package 전체 해시를 대조했다.
두 candidate host의 QKV/FFN image는 E104와 같고, O image는 각 exact compile과 같다.

## 줄이지 못한 비용과 새로 추가된 의존성

[정적 일정·source 연결](static-cost.json), [재현 스크립트](analyze.py).

| 정적 구간 | E104 | M01 DMA | M01 Switch |
|---|---:|---:|---:|
| activation 적재 |933|1022|1022|
| weight 적재 |13383|13435|13435|
| dot 포함 producer stream |5063|5063|5063|
| 새 partial 재배치 |없음|2028|합산과 결합4615|
| 새 별도 partial 합산 |없음|416|위 구간에 포함|
| 전체 makespan |27239|29824(+9.49%)|31995(+17.46%)|

E104 producer는 dot과16열 합산을 같은 stream에서 끝낸다. M01은 이 합산을 분리해도
producer 시간이 줄지 않았다. DMA 경로는 dot 종료21361 이후 partial 이동21361…23389,
합산23389…23805가 직렬로 이어진다. Switch는 dot 뒤21361…25976에서 재배치/합산을
수행한다. 두 경로 모두 이 작업이 끝난 뒤에야 epilogue gather가 시작된다.

E64의 full-weight 재배치17112 modeled cycles를 작은 partial 재배치로 바꾸는 데는
성공했다. 그러나 E104에는 그 full-weight 복사가 없었고, 이번 적재·dot도 더 짧아지지 않았다.
그래서 비교군에 없던 dependency와 transfer를 추가한 조합이 됐다.
이 설명은 실제 source와 compiler 일정으로 확인한 경로다. NPU에서 어느 자원이 몇 cycle
stall했는지 분리 측정한 결과는 아니며, 정적 수치를 hardware 기여율로 바꾸지 않는다.

다시 검토할 조건은 ingress/compute를 실제로 단축하거나 partial 경계를 없애는 **다른
구체적 경로**다. 같은 배치의 padding·tile 수만 바꾸며 M01을 계속 탐색하지 않는다.
이번 결과로 모든 column-major/ingress-oriented 설계가 느리다고 결론 내리지 않는다.

## 검증할 가설

E64는 column-major 적재 뒤 15MiB weight 전체를 원래 배치로 복사해 느려졌다.
M01은 입력 TRF도 column-major slice에 맞추고 그 자리에서 BF16 dot을 계산한다.
합산 전 F32 partial만 옮긴다. 이것은 E64의 재배치 비용을 바꾸는 같은 가족의 시도다.
E104에는 이 weight 복사가 원래 없었으므로 E64 대비 절감을 E104 대비 이득으로 세지 않는다.
E64의 적재 정적 시간13383→13435도 긍정적 근거가 아니다.

## 전체 producer/transfer/consumer

`HBM x/W → slice=(K열,H행그룹) DM → BF16 input TRF + FP8 lookup/BF16 dot
→ compact F32 partial → slice=(H행그룹,K열) → F32 16열 합산
→ H480×8 gather → F32 channel scale → 첫 BF16 → 기존 RMS/residual`.

`c<2,k<16,r<16,t<120,u<256`: slice=`16*k+r`, H행=`1920*c+120*r+t`,
K열=`256*k+u`. 재배치 후 slice=`16*r+k`. Slice당 weight FP8 30720B,
BF16 input TRF 유효512B, F32 partial 유효480B. Weight·input은 dot까지,
partial은 reducer까지 유지한다. Pair commit은8B, consumer packet은8 F32/32B다.
Channel scale/epilogue helper와 QKV/FFN source는 바꾸지 않는다.

이번에 바꾼 것은 slice 축 순서와 부분합 이동/합산 경로다. **120×256 tile, H-owned
cluster, BF16 DPE는 유지**했다. 따라서 다른 tile/cluster 소유권 조합까지 시험한 결과로
확대하지 않는다. 후속 타일링은 HBM source 구간·입력 복제·부분합·레지스터 수명을 함께
바꾸는 조합으로 평가해야 한다. M02의15×2048 tile도 과거 사용됐으므로, 새 검증 대상은
tile 숫자 자체보다 cluster K 소유권과 입력/합산 경로의 결합이다.

두 전송 경로를 사전에 한정한다. tile sweep이나 별도 아키텍처로 세지 않는다.

- `dma-v1`: partial write→DMA read/write→reducer read. 추가 유효 DM edge983040B.
- `switch-v1`: partial write→fetch→Switch Transpose(16,16)→reducer를 한 consumer
  stream에 연결. 추가 DM write/read491520B + ring256 비용. 중간 partial B 저장은 없음.

두 경로 모두 완성된 O를 측정하므로 weight 준비·input 전달·partial boundary·epilogue가
포함된다. 논리량은 실제 bus bytes가 아니며, compile schedule도 NPU stall 분해가 아니다.
첫 결과에서 회귀/5% 미만 개선이면 구체적 예외 근거 없이 긴 튜닝을 이어가지 않는다.
동일 candidate의 두 경로를 E104 전후 control과 함께 측정하는 단회 screen을 사전
계획했으며, 위 Arena95636으로 실행했다.

## 구현 오류 수정과 보존

첫 `dma-v1`은 `vector_final().commit()`의 typestate 연결이 잘못돼 Rust frontend에서
실패했다. `commit_trim::<m![H % 8]>()` adapter를 넣는 것으로 해결했고 Packet8 F32와
산술은 그대로다. 같은 오류가 있던 `switch-v1`은 불필요한 compile 없이 선제 수정했다.
[수정 근거](repair-v2.json), 최초 snapshot·실패 로그는 보존했다.

서로 다른 compiled source는 실패dma-v1·성공dma-v2·성공switch-v2의3개다.
전체 exact compile5회(성공4회 중2회는 host package 재현), host build2회, Arena job1회다.
4개 source snapshot을 두 전송 대안/한 M01 가족과 혼동하지 않는다.

[검증 결과](verification.json)는 바뀌지 않은 함수 영역, partial의 F32 유지,
3840행 packet/transpose/reduce 좌표, frozen source/log와 exact/host image 해시를 확인한다.
좌표 모델과 실제 NPU 정확도는 별도 근거다.

```sh
python3 moa-dev/experiments/candidates/m01-partial-remap/verify.py --check
python3 moa-dev/experiments/candidates/m01-partial-remap/analyze.py --check
python3 moa-dev/experiment_index.py --check --check-live
```

## 문서·SDK 근거

- [Switch](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/switch-engine.html):
  `Transpose`는 slice 두 축을 바꾸며 Time/Packet 값은 보존한다. ring 크기는16×16이다.
- [DMA](https://developer.furiosa.ai/furiosa-opt/book/moving-tensors/dma-engine.html):
  typed tensor 이동으로 물리 배치를 바꾼다. `reshape`를 복사 대용으로 사용하지 않는다.
- 설치 std0.8.1 `src/engine/vector/tensor/vector_tensor.rs`의
  `VectorInitTensor<f32>::vector_inter_slice_reduce`: Packet을 유지하며 slice 축을 합산한다.
  eight-row packet 전체에 같은 K열 합산을 적용하는 설계이며 전체 lowering/NPU 검증이 필요하다.
- `prepare.py`의 실제 source 조합과 각 variant `recipe.json`을 보관한다.
  `structural_build.py --plan`으로 비교군과 source origin을 각각 검사한 뒤 컴파일한다.

source/축 오류는 구체적인 경계 문제를 고쳐 계속 검증한다. 실패 로그와 최초 snapshot을
남기며, API 하나의 존재만으로 전체 조합을 지원한다고 주장하지 않는다.
