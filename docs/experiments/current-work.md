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

## 사용자 지적 이후 변경 범위 재검토

실제 candidate diff와 호출 경로를 다시 읽었다. C09/C11을 C10과 묶어서 세 개의 큰
아키텍처를 충분히 시험한 것처럼 설명한 것은 범위를 과장했다. 이번 실험의 정확한
분류는 다음과 같다. 기존 소스·빌드·판정 기록은 그대로 보존한다.

| 후보 | 그대로 남긴 구조 | 실제 변경 범위 |
|---|---|---|
| C09 | 두 cluster·16-column BF16 dot, F32 합산·scale, 후단 RMS/residual | `project_output`의 행별 전송/계산 의존성을 두 구간으로 분리한 국소 dataflow 변경 |
| C11 | E5M2 세 항 입력, FP4→FP8 weight DM, up/gate→GeGLU→down 전체 단계 의존성, down 구현 | `project_rows_with_scale_contraction`의 partial DM을 VRF로 바꾼 국소 dataflow 변경 |
| C10 | Q/K/V 개별 projection, 입력 전체 복제, 기존 head 후처리 | Weight/activation의 TRF·stream 역할과 row/lane 배치를 교체한 projection 재설계 시도. Lowering 실패로 성능 미평가 |

C07 native FP8/scale 계산 조합과 C08 QKV 통합·열 분할은 더 큰 변경을 실제로
시도한 이력이 있다. 그러나 build·parameter 변형 수를 독립 아키텍처 탐색 수로
해석할 수 없다. 이력만으로 현재 구조가 최적에 가깝다거나 상위권의 구현 방식을
안다고 결론 내리지 않는다.

최근 결과가 주는 정보는 제한적이다. C09는 추가 준비/전송이 overlap을 상쇄했고,
C11은 dot 2→30개로 실행 비용이 늘었다. 이들은 그 구현의 compiler schedule 관측이다.
NPU에서 해당 비율로 느려진다고 확정하거나 전체 설계 계열을 폐기하는 근거가 아니다.

다음 큰 설계는 전체 projection 또는 FFN 실행 graph를 먼저 쓰고, 상속한 데이터
소유권·weight 중간 저장·scale 합산·stage 완료 대기 중 무엇을 없앨지 명시한다.
기존 함수 내부 한 곳의 수정으로 설계 범위를 축소하지 않는다. 작은 지원 여부 검사는
그 큰 설계의 일부를 검증하는 용도로 사용한다. 빠른 종료 규칙은 유지한다.
C12도 현재 제안 범위는 국소 자원 배치 변경이므로 자동으로 다음 대형 설계로 세지 않는다.
C14의 FFN 단계 간 의존성 재설계와 C13의 계산 표현 교체처럼 넓은 가설을 우선 검토하되,
실제 개선 여부는 아직 미검증이다. 이번 재검토에서 새 kernel을 구현·실행하지 않았다.
