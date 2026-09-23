# 현재 구조 실험

작업 폴더: `/home/jihwan38/moa-workspace`. Private GitHub: `jihwan38/moa-workspace`.
조사 기준: `research/structural-candidates-20260923` / `cd1ad4f`.
채택 runtime: [baseline/current.json](../../baseline/current.json)의 E104.

## 최신 작업 — 세 구조 실행

진행 순서는 **C14-B → C15 → C12-B**다.
[설계 범위·비용 가설·과거 시도 대조](structural-round-20260923.md).
첫 `opt/c14-stage-compact`의 C14-B는 전체 compile·기본 NPU 정확도 PASS 후,
FFN397348 vs 동시 E104336341/339553(+17.58%)로 종료했다.
[source·측정·비용 분석](../../moa-dev/experiments/candidates/c14-ffn-l-pipeline/stage-compact.md).
C15도 up/gate column 소유권을 교체해 compile·기본 NPU PASS 후 FFN385621 vs
E104335618/335432(+14.93%)로 종료했다. [전송 비용·수정 기록](../../moa-dev/experiments/candidates/c15-upgate-columns/README.md).
실행 소스는 E104다. 다음 C12-B는 O의 두 구간 Main/Sub pipeline을 검증한다.

## 이전 작업 — main 통합과 E104 비용 진단

**main과 origin/main을 `873f64e`에서 `f9f905b`로 fast-forward해 push했다.**
`opt/c13-int8-ffn`의 `2c51fe2`까지 26개 commit과 작업 규칙·안내 정리 commit을
포함한다. 확인한 모든 기존 local/origin 실험 브랜치 tip이 이 이력에 포함되어,
별도로 병합할 기존 실험 변경은 없다. [통합 확인 기록](integration-20260923.json).

실행 소스는 채택 E104이며, C13/C14는 후보 snapshot에만 남는다. AGENTS.md와
조합 양식에 최신 비교군의 비용 근거, 제거·추가 작업, 실제 의존성/자원 경쟁,
설계 선택을 바꾸는 최소 진단과 그 한계를 기록하도록 보완했다. Root README의
현재 채택본 표시도 E79에서 E104로 수정했다.

당시 브랜치는 **`diag/e104-cost-attribution`**였다. E104의 기존 schedule 293개
instruction을 source와 대조하고 긴 node 18개·context 중첩을 재현 가능한 JSON으로
남겼다. [첫 정적 분석·다음 판별 질문](../../moa-dev/experiments/diagnostics/e104-cost-attribution/README.md).
Weight 준비와 activation 분배 중 무엇을 먼저 크게 바꿀지 결정하는 최소 NPU
진단은 아직 미구현·미실행이다. 이것은 새 최적화 가족이나 E104 구조에 탐색을 고정하는
규칙이 아니다. C14 compact 전송 수정은 아직 미구현인 후속 선택지이며, C12도 미구현이다.

## 이전 작업 — C13 재개 후 NPU 검증 완료

작업 브랜치는 `opt/c13-int8-ffn`이었다. **C13 v7은 전체 compile과 변경 없는 NPU
기본 정확도 검사를 통과했지만, 동시 E104보다 FFN이 18.58% 느려 미채택**했다.
[전체 구현·초기 실패·수정·실측](../../moa-dev/experiments/candidates/c13-int8-ffn/README.md).

초기 v2를 compile 횟수 한도로 닫은 뒤, 사용자의 지적을 받아 구체적인 구현 오류를
수정했다. 8B scale packing과 명시적 입력 복제로 v6을 compile했다. 첫 Arena
실행 E107은 non-finite FFN 출력으로 실패했다. 분리 진단에서 긴 K=3840 integer
unzip/zip 뒤에 scale 곱셈을 붙이는 조합의 block 대응 오류를 확인했다. v7은 정확한
정수 결과를 F32 DM에 저장한 뒤 별도 tag-zero stream에서 scale을 적용한다.

Arena job 78730: **E104 before 334969 → C13 397318 → E104 after 335156 cycles**.
각 process의 기본 15비교가 모두 통과했으며 FFN 3입력/11520출력도 기존 기준 그대로
PASS다. 단회 screen이며 공식 점수·독립 반복 결과가 아니다. 성능 회귀가 커서 더
튜닝하지 않고 종료했다. E104 실행 소스 유지, 공식 제출 없음.

E74를 출발로 up/gate/down 모두 INT8 두 항/I32 누산으로 구현한 같은 C13 구조다.
7개 source variant와 분리 진단을 여러 독립 구조로 세지 않는다. 0 입력의 raw
restoration scale은 CPU subnormal 대신 0으로 나오는 차이가 있어 진단 전체를
PASS로 처리하지 않았으며, 일반 BF16 입력 동치나 확대 검증은 주장하지 않는다.

AGENTS.md에는 수정 가능한 source/layout 오류를 compile 횟수만으로 포기하지
않도록 보완했다. C14의 기존 padding 전송 실패도 불가능성이나 성능 탈락의 근거는
아니다. [C14 후속 검토](../../moa-dev/experiments/candidates/c13-int8-ffn/c14-assessment.md)의
stage 내부 compact 경로는 아직 미구현이다. C12 역시 미구현이며 기존 제안 범위는
국소 Main/Sub dataflow 변경이다.

## 이전 작업 — C14-A

C14 작업 브랜치는 `opt/c14-ffn-l-pipeline`이다. **C14-A는 producer→down DMA의
layout lowering 실패로 종료**했다. [구현·세 버전·종료 근거](../../moa-dev/experiments/candidates/c14-ffn-l-pipeline/README.md).

E74 `9e2058ec0947498d7cb693fbe98e7b5f0b5c9ea7`에서 출발해 BF16 stream 계산,
분산 GeGLU, L-owned down cluster, 두 L 구간의 F32 partial 합산을 한 경로로
구현했다. E104는 비교 계획에만 사용했다. 기존 전체-stage 대기와 H-owned down
배치를 함께 변경한 whole-FFN 재설계 시도이며, 완성된 image는 생성되지 않았다.

Compile 3회: 최초 tiled-view reshape 실패 → HBM 축 분할 순서 수정 후 factorize
실패 → 별도 cluster 축 표현으로 수정 후 padded producer→down column DMA 실패.
기본 수정 횟수를 한 번 넘긴 구체적 근거와 총 3회 제한은 v3 recipe에 사전 기록했다.
논리 좌표/pitch/GeGLU 연산 순서 검사는 통과했지만 NPU 정확도·성능은 미평가다.
Arena·공식 제출 없음, E104 실행 소스 유지. C14 전체가 느리다는 판정은 아니다.
당시 다음 우선순위였던 C13의 실행 결과는 위 최신 작업에 기록했다. C12는 미구현이다.

## 이전 작업 — C09/C11/C10

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

세 실험의 마지막 브랜치는 `opt/c10-direct-weight-trf`이며 C09 → C11 → C10 순서의 기록을
모두 포함한다. 세 후보 브랜치는 private origin에 각각 push했다. E104 실행 소스는
바꾸지 않았고 `experiment_index.py --check --check-live`로 확인한다.
당시 후속 미구현 후보는 C12(Main/Sub 분리), C13(INT8 DPE), C14(FFN 전체 구간
pipeline)였다. C14-A의 후속 실행 결과는 위 최신 작업에 기록했다.

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

## 소스 선택의 E104 고정 제거

소스 선택 수정 브랜치는 `fix/recipe-driven-experiments`다. 후속 지적을 받고 실제 빌더를
확인한 결과, `structural_build.py`는 JSON의 `base_commit`을 사용하지 않고 E104 상수를
`common.BASE`에 대입하고 있었다. 변경하지 않은 image의 검사도 E104로 고정돼 있었다.
이 고정은 내가 만든 구현 선택이며, 기록만으로 조합 탐색이 수행됐다고 볼 수 없다.

빌더가 recipe의 출발 commit과 variant별 파일 선택을 실제로 사용하도록 수정했다.
성능 비교군은 `comparison_recipe`로 분리했다. 과거 build를 image 기준으로 쓸 때는
선택한 출발 source와 일치하는지 검사한다. 같은 destination의 중복 덮어쓰기는 거부한다.
[실제 입력 형식과 조립 범위](combination-template.md)를 따른다.

검증: E55/E79/E104 source 선택과 E104 비교군의 독립성, variant별 source 선택,
기준 image의 잘못된 승계 거부, 중복 destination 거부, 기존 세 recipe 호환성,
실제 Git archive→변경 파일 복사 결과까지 **6개 테스트 PASS**다. 실제 조립 검사는
첫 Docker 호출 직전에 멈추도록 만든 테스트다. 새 compiler/NPU/공식 제출 실행은 없다.
기존 실험의 frozen source/log/build JSON과 채택 E104는 변경하지 않았다.

이 수정은 선택한 조합을 빌더가 따르게 만든 것이다. 과거 method ID를 자동으로
서로 호환되는 코드로 합치는 기능은 아니다. 다음 실험은 출발 구조·과거 방법·제거할
충돌 요소의 조합부터 정하고, 그 조합을 담은 실제 코드와 JSON이 일치하는지 확인한다.
