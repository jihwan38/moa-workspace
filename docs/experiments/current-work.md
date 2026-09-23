# 현재 구조 실험

작업 폴더: `/home/jihwan38/moa-workspace`. Private GitHub: `jihwan38/moa-workspace`.
조사 기준: `research/structural-candidates-20260923` / `cd1ad4f`.
채택 runtime: [baseline/current.json](../../baseline/current.json)의 E104.

사용자가 승인한 실행 순서는 **C09 → C11 → C10**이다.
[설계·과거 시도·종료 기준](../research/structural-candidates-20260923.md).

| 후보 | 브랜치 | 상태 |
|---|---|---|
| C09 | `opt/c09-output-pipeline` | [종료](../../moa-dev/experiments/candidates/c09-output-pipeline/README.md): compile PASS, 정적 +0.71%, Arena 미실행 |
| C11 | 예정: `opt/c11-ffn-partial-vrf` | 대기 |
| C10 | 예정: `opt/c10-direct-weight-trf` | 대기 |

각 후보는 E104 소스에서 별도로 출발한다. 다음 브랜치는 이전 실험의 기록을 이어받으며,
채택하지 않은 kernel 구현은 후보 snapshot에 보존한다. Main 병합·공식 제출 상태와
Arena 실험 상태를 구분한다.
