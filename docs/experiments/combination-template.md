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
| 선택한 구조·과거 방법 조합 / 충돌해서 제거한 방법 | |
| 같은 방법을 배치할 대안 구조와 기존 실패와의 차이 | |
| 후보 source / 재현 스크립트 / 기준 대비 patch | |
| compiler·SDK·Rust toolchain / Cargo.lock | |
| 실제 compile 입력 해시 manifest | |
| exact kernel image·host·fixture·runner·upload 해시 근거 | |
| 비교 대조군의 측정 JSON | |

## 조합 구성

비교군의 recipe와 후보의 실제 출발점·설계를 구분한다. E55/E74의 구조를 유지하거나
그 recipe 전체에서 출발할 필요는 없다. 비교군에서 유지·제거·대체한 구성과 새 방법의
동작을 적는다. ID가 같아도 설정 또는 산술 순서가 다르면 변경으로 기록한다.

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

### JSON을 실제 소스 선택에 연결하기

`moa-dev/experiments/structural_build.py`는 후보의 `candidate.json`을 읽는다.
`<variant>/recipe.json`이 있으면 그 완전한 recipe를 대신 사용한다. Variant마다
다른 source 부모와 과거 구현 파일을 선택할 수 있다. 예를 들어 아래는 **E79에서
출발하고 E104와 성능을 비교**하는 source 선택 형식이다. 조합의 성능 예시는 아니다.

```json
{
  "base_commit": "2735b5899f41195dedeab1e076c82317e24efcb2",
  "comparison_recipe": "baseline/e104.json",
  "target_kernel": "decoder_feedforward",
  "changed_sources": {
    "mlp.rs": "src/device/shared/mlp.rs",
    "ops.rs": "src/ops.rs"
  }
}
```

`changed_sources`의 key는 variant 폴더 기준 파일 경로다. 이미 보존한 다른 후보의
snapshot도 명시적으로 가리킬 수 있다. Value는 조립할 crate 안의 destination이다.
같은 destination을 두 번 지정하면 앞선 방법이 사라질 수 있으므로 빌더가 거부한다.
동일 파일의 두 방법은 호환되는 하나의 구현으로 직접 조합하고 위 구성표에 출처와
변경한 연결을 적는다. 모든 과거 방법을 자동 on/off하는 시스템을 구현한 것은 아니다.

`comparison_recipe`는 빌드 기록에 남기는 독립 비교 계획이다. 이 도구가 Arena batch를
자동 구성하지는 않는다. 측정할 때 실제 control package가 그 recipe와 일치하는지
별도로 확인한다. 기존 C08 `screen.py`의 control은 E104로 고정되어 있다.

먼저 `--plan`으로 source commit·파일별 hash·비교 recipe를 확인한다. 실제 조립한
compile 입력이 이 선택과 다르면 Docker/compile 전에 중단한다.

```bash
python3 moa-dev/experiments/structural_build.py CANDIDATE VARIANT --plan
python3 moa-dev/experiments/structural_build.py CANDIDATE VARIANT --label NEW-BUILD
```

선택적으로 `source_reference_build`에 출발 source의 과거 build JSON을 지정하면
변경하지 않은 kernel의 image 동일성을 검사한다. 이때 reference의 전체 compile
입력이 선택한 source commit과 일치해야 한다. 성능 비교군 E104의 image를 다른
source 부모의 불변 조건으로 사용하지 않는다. 여러 kernel을 함께 바꾸면
`unchanged_kernels`에 실제로 보존할 kernel만 지정한다. Reference가 없으면 자동
동일성 승계를 하지 않고 생성 image의 hash와 host 포함 여부를 기록한다.

Recipe의 `methods`와 `invariant`는 해당 구현을 설명한 기록이다. 새 조합에서 기존
내부 구조를 모두 보존해야 한다는 뜻이 아니다. 대회 입력·공개 인터페이스와 정확도
기준은 유지하고, 바뀐 조합 전체의 정확성을 새로 검증한다.

## 비용 가설과 판별 근거

최신 비교군의 실제 호출·producer/transfer/consumer 경로에서 출발한다. 과거 채택본의
병목 설명을 그대로 승계하지 않는다. 논문의 원리와 SDK에서 구현된 경로를 연결하고,
큰 변경이 줄이는 대기·저장과 새로 만드는 작업을 함께 적는다.

| 항목 | 기록 |
|---|---|
| 현재 실행 완료를 늦추는 의존성 또는 공유 자원 / source·schedule·NPU 근거 | |
| 없애거나 겹칠 작업 / 그대로 남는 대기 | |
| 추가되는 변환·양자화·복제·partial 저장/재읽기·scale·재배치·준비 비용 | |
| compiler 예상·논리 byte 수·실측 시간을 구별한 비용 추정과 불확실성 | |
| 미확인 원인이 설계 선택을 바꾸는가 / 그렇다면 구분할 가설 | |
| 필요한 최소 진단 / 고정할 입력·배치 / 측정량·새 경계 비용 / 결과별 다음 결정 | |

기존 증거로 판단할 수 있으면 새 진단을 의무적으로 추가하지 않는다. 진단용 사전
변환이나 부분 실행은 전체 커널 최적화·공식 성능으로 보고하지 않는다. 분리 실행이
바꾼 중첩·캐시·배치·출력 비용도 기록하며, 개별 시간을 더하거나 전체 시간에서 빼서
원래 커널의 병목 기여율을 단정하지 않는다. 결합 설계의 중간 단계마다 단독 개선을
요구하지 않고, 완성된 경로를 동시기 대조군과 NPU에서 검증한다.

## 측정

변화율은 `(후보 cycles / 같은 비교 기준의 cycles − 1) × 100`이며 음수가 빠르다.
기본 입력·허용오차는 고정한다. 정확도 실패 시 cycles 대신 실패로 표시한다.

| 조합 | 고정 source/구성 | 대상 커널 cycles | 기준 대비 | 독립 job 수·변동 범위 | 기본 정확도 | 추가·경계 검사 | 측정 JSON |
|---|---|---:|---:|---|---|---|---|
| C | 기준 | | — | | | | |
| C+A | A 단독 | | | | | | |
| C+B | B 단독 | | | | | | |
| C+A+B | 조합 | | | | | | |

위 표는 변경을 독립적으로 분리할 수 있을 때의 예시다. 결합 설계를 직접 시험하거나
완성된 후보에서 하나씩 제거해 비교할 수 있다. 중간 단계의 단독 개선을 요구하지 않는다.

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
