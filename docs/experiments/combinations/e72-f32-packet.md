# E72: F32 packet scale의 up/gate와 down 조합

**미채택, E55 유지.** C02의 A=up/gate, B=down, A+B를 각각 실행했다.
기준은 `baseline-e55`, commit `873f64ed94229c88cd18d2731bdcca7eea0d6365`다.
[구현·진단·빌드·업로드 감사](../../../moa-dev/experiments/candidates/c02-c03-e55/README.md).

| 구성요소 | A = E70 up/gate | B = E71 down |
|---|---|---|
| 유지 | Q1–Q6, O1–O3, R1, F1 cluster/row 분할, F2 전체 decode, F5 후단; down 전체 | 같은 기본 구성; up/gate 전체 |
| F3 변경 | 두 16원소 block의 F32 dot → 각각 F32 scale → compact F32 DM → F32 합산 → BF16 | 같은 경로 뒤 column-slice F32 합산 → BF16 |
| F4 변경 | 8행×4 pass → 4행×8 pass | 12행×5 pass → 4행×15 pass |
| 추가 비용 | 연속 F32 scale 생성·sequencer padding·partial 저장/재읽기 | 같은 종류의 pass와 row 반복 증가 |
| VRF / row tile | 7,680B | 3,840B |
| 수치 경계 | 추가 BF16 partial 없음; F32 합산 배치는 변경 | 추가 BF16 partial 없음; F32 합산 배치는 변경 |

성능은 job73881의 별도 프로세스로 측정했다. 기본 host/fixture/runner는 동결 E55와 같고,
바깥 batch script가 실행 순서를 정한다. 기준 C는 같은 job의 E55 앞뒤 FFN 중앙값
379,238 / 379,598의 중앙값 379,418이다. 서로 다른 stage를 독립 job으로 세지 않는다.

| 구성 | FFN cycles | C 대비 | 기본 비교 | 판정 |
|---|---:|---:|---|---|
| C+A, [E70](../e70-c02-upgate.json) | 432,912 | +14.10% | 15 PASS | 단회 회귀 |
| C+B, [E71](../e71-c02-down.json) | 415,542 | +9.52% | 15 PASS | 단회 회귀 |
| C+A+B, [E72](../e72-c02-combined.json) | 456,372 | +20.28% | 15 PASS | 단회 회귀 |

기본 FP4 weight decode와 O/QKV image는 세 후보에서 변하지 않았다. 후보에 따른
O/QKV cycle 차이는 FFN 최적화 효과가 아니다. 큰 회귀가 있어 C02의 추가 6입력,
독립 성능 반복, 전체 모델 검증은 진행하지 않았다. 별도 4-case packet 검사에서
F32 scale 3,840개와 출력 16개는 exact PASS다.

C03은 Q6 head 전달 배치를 바꾼 별도 QKV 후보다. [E68/E69](../e68-c03-cluster-heads.json)
반복에서 +1.56%로 회귀했고 C02와의 전체 결합은 실행하지 않았다. C03의 추가
30개 출력 비교와 비대상 cache 25,141,248개 BF16 원소 검사는 통과했다.
어느 후보도 baseline pointer/tag/live crate에 적용하거나 공식 제출하지 않았다.
