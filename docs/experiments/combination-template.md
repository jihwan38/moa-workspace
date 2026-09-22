# 조합 실험 기록 양식

[현재 baseline](../../baseline/README.md) · [방법 목록](../../baseline/methods.md) ·
[실제 E60 예시](combinations/e60-down-padding.md)

새 조합마다 `docs/experiments/combinations/` 아래 문서를 만들고 아래 항목을 채운다.
실행하지 않은 칸은 `미측정`, 판단할 수 없는 원인은 `미확인`으로 남긴다.
실험 ID와 현재 적용 방법 ID는 구별한다. 예: E60은 실험이고 F1은 baseline 구성요소다.

## 식별과 재현

| 항목 | 기록 |
|---|---|
| 실험 ID / 가설 / 변경 대상 커널 | |
| 기준 recipe / 고정 Git commit | |
| 실제 source 부모 commit 또는 snapshot manifest | |
| 후보 source / 재현 스크립트 / 기준 대비 patch | |
| compiler·SDK·Rust toolchain / Cargo.lock | |
| 실제 compile 입력 해시 manifest | |
| exact kernel image·host·fixture·runner·upload 해시 근거 | |
| 비교 대조군의 측정 JSON | |

## 조합 구성

Baseline recipe의 적용 ID 전체를 출발점으로 적고, 새 방법은 새 ID와 동작 설명을 붙인다.
ID가 같아도 설정 또는 산술 순서가 다르면 변경으로 기록한다.

| 구분 | 방법 ID | 이전 → 이후 / 이유 |
|---|---|---|
| 유지 | | |
| 추가 | | |
| 제거 | | |
| 파라미터 변경 | | |
| 대체 구현 | | |

함께 바뀌는 layout·helper·호출자, dtype·반올림 경계, 유효 원소·padding의 대응을 적는다.
여러 후보가 같은 파일을 수정하면 각 변경이 최종 source에 함께 존재하는지 확인한다.
`patch 적용 성공`과 `의도한 모든 방법이 최종 image에 들어 있음`은 별도로 검증한다.

## 측정

변화율은 `(후보 cycles / 같은 비교 기준의 cycles − 1) × 100`이며 음수가 빠르다.
기본 입력·허용오차는 고정한다. 정확도 실패 시 cycles 대신 실패로 표시한다.

| 조합 | 고정 source/구성 | 대상 커널 cycles | 기준 대비 | 독립 job 수·변동 범위 | 기본 정확도 | 추가·경계 검사 | 측정 JSON |
|---|---|---:|---:|---|---|---|---|
| C | 기준 | | — | | | | |
| C+A | A 단독 | | | | | | |
| C+B | B 단독 | | | | | | |
| C+A+B | 조합 | | | | | | |

한 job 내부의 서로 다른 공식 입력 3개는 반복 job 3회가 아니다. Job 중앙값 범위와
원본값을 함께 확인한다. 1회 측정에는 실행 간 변동을 `미측정`으로 적는다. 여러
방법이 동시에 바뀐 과거 기록은 개별 기여를 분리한 실험으로 해석하지 않는다.

## 판단과 다음 비교

- **성능 판정:** 빨라짐 / 차이 미확인 / 느려짐 / 유효 성능 없음. 대조군과 반복 근거를 적는다.
- **정확도·빌드 판정:** 기본 PASS/FAIL, 추가 검사 범위, compile 실패나 timeout 여부를 구분한다.
- **채택:** 채택 / 미채택 / 추가 검증 대기. 현재 baseline에 반영했는지도 명시한다.
- **원인:** source·schedule·graph로 확인한 사실과 추정을 구별한다.
- **후속:** 유지할 방법, 되돌릴 방법, 다음 비교를 적는다. 새 조합은 별도 ID로 남긴다.

원본 결과를 `measure.py`로 수집한 뒤 [catalog.json](catalog.json)에 대상 커널별 판정과
비교 JSON을 연결하고 `python3 moa-dev/experiment_index.py`로 결과 분류를 갱신한다.
정적 분석만 한 후보와 정확도 실패 후보도 근거를 남긴다. 생성 표를 직접 편집하지 않는다.
