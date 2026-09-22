# E67: up/gate와 down scale-DPE 조합

**미채택, E55 유지.** C01의 up/gate A와 down B를 따로 실행한 뒤 A+B도 실행했다.
A의 작은 감소는 반복 범위가 겹치고, B와 A+B는 첫 screen에서 E55보다 느렸다.
실행한 전체 소스와 진단은 [C01 기록](../../../moa-dev/experiments/candidates/c01-scale-dpe-e55/README.md)에 있다.

기준 C는 [E55 recipe](../../../baseline/e55.json), commit
`873f64ed94229c88cd18d2731bdcca7eea0d6365`다. 비교 시점이 같은 대조 job 3회를 사용한다.

| 구성요소 | A = E65 up/gate | B = E66 down |
|---|---|---|
| 유지 | Q1–Q6, O1–O3, R1, F1의 cluster/row 분할, F5 epilogue; down 전체 | Q1–Q6, O1–O3, R1, F1의 cluster/column 분할, F2 decode, F5 epilogue; up/gate 전체 |
| F2 대체 | 전체 BF16 weight DM 저장을 없애고 FP8 lookup→첫 dot 연결 | 전체 BF16 weight DM 유지 |
| F3 대체 | F32 partial→BF16 DM→scale-DPE | F32 partial→BF16 DM→128칸 staging/tail clear→scale-DPE→F32 column 합 |
| F4 변경 | 8행×4 pass → 32행×1 pass | 12행×5 pass → 20행×3 pass |
| 추가 비용 | partial 저장/읽기, scale BF16 변환과 TRF | 위 비용과 compact120→128 복사 6개, 실제 tail zero-write |
| Gather 변경 | 기존 그대로 | E33 계열의 cluster compact 후 epilogue gather; 직접 gather 대비 정적 +451 cycles |
| 수치 경계 | block partial BF16 반올림 추가 | block partial BF16 반올림 추가 |

이 표의 A/B는 관련 배치·tile을 포함하는 구현 단위다. F2/F3/F4를 독립적인 on/off
스위치나 각각 증명된 가속 요소로 해석하지 않는다. 추가 BF16 rounding의 상쇄 반례는
C01 D01에서 따로 확인했다. 최종 FFN의 보편적인 오차 허용을 증명한 것은 아니다.

| 조합 | FFN cycles | C 대비 | job 수 | 기본 비교 | 결과 |
|---|---:|---:|---:|---|---|
| C: E55 | 378,270 | — | 3 | 45 PASS | [대조](../e55-c01-control-repeats.json) |
| C+A: E65 | 367,967 | −2.72% | 3 | 45 PASS | [반복](../e65-c01-upgate-scale-dpe-repeats.json), 범위 겹침 |
| C+B: E66 | 413,913 | +9.42% | 1 | 15 PASS | [단회 screen](../e66-c01-down-scale-dpe.json) 후 중단 |
| C+A+B: E67 | 392,118 | +3.66% | 1 | 15 PASS | [단회 screen](../e67-c01-combined-scale-dpe.json) 후 중단 |

E65는 추가 run3..8의 30개 비교도 PASS했다. E66/E67의 추가 입력·반복 성능·전체 모델
검사는 수행하지 않았다. 세 후보의 QKV/O device image는 E55와 같으며 그 cycle 변동은
FFN 최적화 효과로 세지 않는다. B/조합의 단회 값으로 상호작용 크기를 확정하지 않는다.

최종 source는 [E65](../../../moa-dev/experiments/candidates/c01-scale-dpe-e55/mlp.upgate-streamed.rs),
[E66](../../../moa-dev/experiments/candidates/c01-scale-dpe-e55/mlp.down.rs),
[E67](../../../moa-dev/experiments/candidates/c01-scale-dpe-e55/mlp.combined.rs)로 별도 보존했다.
[verify.py](../../../moa-dev/experiments/candidates/c01-scale-dpe-e55/verify.py)는 각 source의
46개 compile 입력과 실제 image/host/upload/측정을 연결한다. 현재 baseline pointer,
baseline tag, 실행 crate는 E55다. 공식 점수·순위 변경은 없다.
