# 현재 구조 실험

작업 폴더: `/home/jihwan38/moa-workspace`. Private GitHub: `jihwan38/moa-workspace`.
조사 기준: `research/structural-candidates-20260923` / `cd1ad4f`.
채택 runtime: [baseline/current.json](../../baseline/current.json)의 E104.

사용자가 승인한 실행 순서는 **C09 → C11 → C10**이다.
[설계·과거 시도·종료 기준](../research/structural-candidates-20260923.md).

| 후보 | 브랜치 | 상태 |
|---|---|---|
| C09 | `opt/c09-output-pipeline` | [종료](../../moa-dev/experiments/candidates/c09-output-pipeline/README.md): compile PASS, 정적 +0.71%, Arena 미실행 |
| C11 | `opt/c11-ffn-partial-vrf` | [종료](../../moa-dev/experiments/candidates/c11-ffn-partial-vrf/README.md): compile PASS, 정적 +3.55%, Arena 미실행 |
| C10 | `opt/c10-direct-weight-trf` | [종료](../../moa-dev/experiments/candidates/c10-direct-weight-trf/README.md): StoTrf lowering에서 현재 target 설정 제약으로 compile 실패 |

각 후보는 E104 소스에서 별도로 출발한다. 다음 브랜치는 이전 실험의 기록을 이어받으며,
채택하지 않은 kernel 구현은 후보 snapshot에 보존한다. Main 병합·공식 제출 상태와
Arena 실험 상태를 구분한다.

이번 요청의 세 후보는 각각 **최초 compile 1회**로 종료했다. C09/C11은 E104 control도
각 1회 compile했으며 exact image hash가 채택본과 동일했다. Arena 실행·공식 제출은
없다. 성능표의 +0.71%/+3.55%는 compiler schedule이며 NPU cycle 변화가 아니다.

현재 작업 브랜치는 `opt/c10-direct-weight-trf`이며 C09 → C11 → C10 순서의 기록을
모두 포함한다. 세 후보 브랜치는 private origin에 각각 push한다. E104 실행 소스는
바꾸지 않았고 `experiment_index.py --check --check-live`로 확인한다.
후속 미구현 후보는 C12(Main/Sub 분리), C13(INT8 DPE), C14(FFN 전체 구간 pipeline)다.
