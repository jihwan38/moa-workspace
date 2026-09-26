# O weight ingress — 연구 후보의 NPU 실행 결과, 2026-09-27

**M08 행 배치는 느려졌고, 동결 C09의 두 stage 공급은 단회2.50% 빨라졌다.**
두 후보의 전송 계약과 기본 정확도를 확인했지만, 10위권 격차를 메울 큰 개선은 얻지 못했다.
Live runtime은 E122로 유지한다. 두 후보 모두 추가 튜닝·채택·공식 제출은 하지 않았다.

[이전 문헌·구현 종합 분석](o-weight-ingress-20260927.md)의 두 우선 질문을 실제 실행했다.
M08 기록을 main/origin `617aaef`에 통합한 뒤 별도 branch에서 C09 최초 NPU를 진행했다.
C09는 새 tile 변형이 아니라 기존 동결 source와 **같은 exact O image**다.
성능 비교군은 모두 동시 E122이며, QKV/FFN도 actual E122 image를 사용했다.

| 후보 / Arena job | E122 O before | 후보 O | E122 O after | 앞뒤 중앙값 대비 | 판정 |
|---|---:|---:|---:|---:|---|
| [M08 / E125 / 105767](../../moa-dev/experiments/candidates/m08-output-pair4-ingress/README.md) | 51418 | 58095 | 50906 | **+13.551%** | 회귀 종료 |
| [동결 C09 / E126 / 105820](../../moa-dev/experiments/candidates/c09-output-first-npu/README.md) | 51237 | 49923 | 51165 | **−2.496%** | 작은 positive option으로 보존, 탐색 종료 |

각각 한 allocation, 원본3입력이다. 각 job의7stage에서57개 출력 비교가 통과했다.
Control을 포함한57개의 반복 비교가 서로 다른57개 입력이라는 뜻은 아니다.
표는 표준 전체 host의 O cycles이며, 별도 diagnostic info/trace 수치는 섞지 않았다.
Actual upload tar·host/device image·원본 로그·입력/runner 해시까지 검증했다.

## M08: HBM 주소 간격을 줄인 효과보다 gather 분할 비용이 컸다

원본 row-major FP8 weight15MiB를 추가 복사 없이 pair-preserving4-way 배치로 읽었다.
BF16 x TRF, inline FP8→BF16 lookup/dot, K16 F32 합산, scale 뒤 첫 BF16은 유지했다.
원본 byte를 정확히 한 번 읽고512개 slice에 맞는 weight/x를 공급함을 실제 graph로 검사했다.

그 결과 HBM traversal은 바뀌었지만 weight marker 중앙값은26769→27190으로 줄지 않았다.
Dot+reduce는12019→12011로 거의 같고, F32 gather는**1201→8222**로 늘었다.
같은15360B를 옮겨도32×480B에서1920×8B로 분할한 비용이 크게 나타났다.
모든 관측에서 gather가 길어졌고 전체 O도 회귀했으므로 이번 조합은 종료한다.
행 interleave 전체를 기각하거나 물리 HBM bank conflict가 없다고 증명한 결과는 아니다.

**설계 조건:** 원본 HBM 순서만 바꾸고 downstream에 복원 비용을 남기면 안 된다.
다음 배치가 있다면 weight·dot·F32 결과 소유권·scale/RMS·최종 store까지 함께 정의해야 한다.
이번처럼 weight 이득 자체가 관측되지 않은 배치에서 gather만 미세 조정하는 우선순위는 낮다.

## C09: 첫 dot 지연은 줄였지만 전체 이득은 작았다

각 slice의120행 weight를60행씩 별도 compact FP8 DM에 읽고, 동일한 BF16 x TRF를
두 dot에서 재사용했다. 두 F32 commit은 동일 result의 서로 겹치지 않는60행 영역이며,
최종 gather는 기존32×480B를 유지한다. 두 HBM source view의0/60행 offset까지 따라가
15MiB 일회 coverage와 output bounds를 확인했다. 새 compile의 O image/schedule도 원본과 같다.

| 같은 launch의 Task 기준 상대 시각/구간, 각6관측의 중앙값 | E122 | C09 |
|---|---:|---:|
| 첫 dot 시작 | 28945 | 17148 |
| 마지막 dot 종료 | 40953 | 39542.5 |
| 첫 weight 시작→마지막 weight 종료 | 26660 | 30073 |
| 첫 weight 시작→마지막 dot 종료 | 38679 | 36802 |
| 마지막 dot 종료→gather 시작 | 2 | 2 |
| Gather duration | 1199 | 1228 |

첫 dot은 절반 분량이므로 control의120행 dot과 그 duration을 직접 비교해 배속으로 쓰지 않는다.
중앙값끼리 차이를 낸 시각 이동과 same-run/cluster index별 차이의 중앙값도 구분한다.
위 첫/마지막 dot 시각 차이는 각각−11797/−1410.5이고, index별 차이 중앙값은−11638/−1566이다.
각 interval은 대기를 포함할 수 있으며 overlap을 중복 합산해 전체 절감량을 역산하지 않는다.

- dot0와 weight1의 실제 overlap: **7392.5**, 범위7153–7468cycles.
- weight0 종료→weight1 시작: **2154.5**. 그 사이의 table1 DMA 구간은2044이며,
  table DMA 종료→weight1 시작은98이다. 관측된 순서와 공백이며, source 재배열로 전부 제거할 수 있다는 뜻은 아니다.
- 같은60행 작업의 dot0 interval9564 vs dot1 6724. Table1 StoTab interval도7489다.
  동시실행 조건에서 marker 구간이 달라졌다. Counter/분리실험 없이 원인을 table 충돌·DM bank 충돌로 확정하지 않는다.
- 정적 일정에서 마지막dot→gather는0→1206이었으나 **NPU는 양쪽6관측 모두2**였다.
  정적 handoff를 이번 실측 병목이나 회수 가능한1206cycles로 설명하면 틀린다.

따라서 staging이 표현되지 않거나 overlap이 전혀 안 된다는 가설은 이 prototype에 맞지 않는다.
실제로 첫 dot은 앞당겨지고 overlap도 생겼다. 다만 두 stage의 공급 envelope와 반복 준비/
작은 작업/동시실행 비용을 모두 지불하면 전체 이득은2.50%에 그친다.
정적+0.71%를 하드웨어 회귀로 보았던 판정도 사용할 수 없다. 기존 정적 종료 기록은
그대로 두고, 이번 최초 NPU 결과를 별도로 연결했다.

## 다음 설계의 조건

이번 실행의 기본5% 기준은 탐색 예산 조건이다. C09는 작은 이득의 조합 선택지로 보존하되
독립 반복·확장 정확도가 없으므로 채택하지 않는다. 과거 결과나 단회57 PASS를 채택 검증으로 전용하지 않는다.

다음 주력은 **weight 완료 dependency를 분할하되, stage 경계의 새 비용을 줄이는 완성 경로**다.
구체적으로는 C09의 다음 weight 시작 전 공백, 길어진 weight 공급 envelope,
동시실행 시 첫 dot interval 증가 중 무엇을 바꾸는지 설명하는 후보가 필요하다.
4/8-stage 숫자만 늘리는 sweep은 반복 setup/작은 task를 더 만들 수 있어 근거가 약하다.
기존 primitive/API로 가능한지 확인한 뒤 한 변경이 어느 실제 window를 줄이는지 검증해야 한다.
공개 SDK에서 확인되지 않은 reusable LUT handle이나 수동 DMA priority를 가능하다고 가정하지 않는다.

[문헌 비교](o-weight-ingress-20260927/literature.md)의 staged supply·local conversion 원칙은 여전히 참고가 된다.
그러나 현재 O는 이미 FP8 compact 저장과 소비 직전 BF16 변환을 한다. 전체 BF16 materialization,
무료 offline packing, 낮은 activation 정밀도는 이번 관측에서 정당화되지 않았다.
입력·합산·scale 수치 경계를 유지하면서 **최초 dot 시각과 전체 종료 시각을 함께** 평가해야 한다.

M08과 C09는 local dataflow 검사다. 이를 전체-projection architecture를 광범위하게 탐색했다거나
랭커 수준의 구조가 확보됐다는 근거로 세지 않는다. 이번 단계의 성과는 두 병목 가설을
실제 hardware evidence로 좁히고, 다음 후보가 반드시 제거해야 할 비용을 특정한 것이다.

재검증은 두 후보 README의 descriptor/profile/results `--check`, 독립 package 검토,
`python3 moa-dev/experiment_index.py --check --check-live`에 연결된다.
