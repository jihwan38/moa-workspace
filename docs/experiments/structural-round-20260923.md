# 세 구조의 비용 검증

사용자 요청: AGENTS.md를 따라 가능성 있는 세 구조를 넓게 탐색하고, 늘어난 비용의
원인을 확인하며 실제로 시도한다. 비교군은 E104이며 source 출발점을 제한하지 않는다.
시작 전 `diag/e104-cost-attribution` 기록까지 main `8793514`에 통합했다.

| 순서 | 후보·범위 | 제거하려는 제약 | 추가 비용과 첫 판별 |
|---|---|---|---|
| 1 | C14-B, whole FFN | 전체 L 완료 대기, H-owned down의 cross-cluster 입력 복제 | E74에서 L-stage/local GeGLU/L-owned down을 함께 사용. Stage 내부 compact DMA로 기존 lowering 오류를 수정하고 stage별 load·실행·F32 partial 비용을 확인 |
| 2 | C15, whole up/gate projection | 모든 slice가 전체 H와 세 FP8 입력 항을 보유하는 배치 | E104 산술과 C04 ring leader 방송을 조합하되 row/column 소유권을 공동 교체. 4-column H 분할·120 local rows로 입력 payload를 줄이는 대신 strided weight DMA와 F32 column reduction이 추가됨 |
| 3 | C12-B, O projection의 producer/consumer pipeline | Main이 dot과 column reduction·후처리를 함께 끝내야 다음 구간을 처리하는 구조 | C09 두 구간 staging과 Main dot/Sub F32 reduction을 결합. Partial 저장·재읽기, shared VE 점유와 lookup 반복 비용이 중첩 이득을 상쇄하는지 확인 |

C14-B는 새로운 가족이 아니라 미실행 C14의 구체적 전송 수정이다. C15는 기존 down
column sweep을 up/gate의 새 계약으로 옮겨 input/TRF, weight/scale, partial reduction과
최종 row 복원까지 함께 바꾸는 projection 설계다. C12-B는 단순 engine 이름 교체의
개선을 가정하지 않고 producer와 consumer를 나눈 O 전체 경로를 검증한다.

기존 자료: [현재 비용 진단](../../moa-dev/experiments/diagnostics/e104-cost-attribution/README.md),
[C14 전송 실패](../../moa-dev/experiments/candidates/c13-int8-ffn/c14-assessment.md),
[C04](../../moa-dev/experiments/candidates/c04-down-transfer/README.md),
[C07](../../moa-dev/experiments/candidates/c07-native-scale-dpe/README.md),
[C09](../../moa-dev/experiments/candidates/c09-output-pipeline/README.md),
[C05 Sub 실패](../../moa-dev/experiments/candidates/c05-streamed-decode/README.md).
이번 사전 대조에서는 기존 candidate README 목록 전체를 확인하고 관련 구현을 읽었다.

SDK 공식 [스케줄 조정](https://developer.furiosa.ai/furiosa-opt/book/scheduling/tuning.html),
[register files](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/register-files.html),
[메모리 성능](https://developer.furiosa.ai/furiosa-opt/book/moving-tensors/memory-performance.html)을
9월 23일 확인했다. Buffer 수나 작은 tensor만으로 중첩·대역폭 향상을 추정하지 않는다.
설계의 각 producer/consumer를 실제 SDK 0.8.1 소스·compile 결과에 대응시킨다.

각 후보의 완성 경로와 기본 정확도/NPU 비교가 비용 가설의 판별 실험이다. 별도 지원
검사나 원인 분리가 필요한 경우에만 작은 진단을 추가한다. 빌드 횟수만으로 수정 가능한
source 오류를 포기하지 않되, 손실 원인이 확인된 같은 매개변수 sweep은 하지 않는다.
정적 결과는 실제 NPU 차이와 구분하고, 유효 후보는 E104→후보→E104 같은 할당에서
검사한다. 회귀와 5% 미만 개선은 기본적으로 닫고, 유망한 후보만 독립 반복과 확대
정확도 검증을 진행한다. 세 후보의 source·로그·hash·결과와 종료 이유를 각 브랜치에 남긴다.

현재 상태: C14-B 전체 compile·기본 NPU PASS, 동시 E104 대비 FFN+17.58%로 종료.
[기록](../../moa-dev/experiments/candidates/c14-ffn-l-pipeline/stage-compact.md).
C15는4-column source 오류를2-column 정렬 배치로 수정해 compile·기본 NPU PASS,
동시 E104 대비 FFN+14.93%로 종료했다. [기록](../../moa-dev/experiments/candidates/c15-upgate-columns/README.md).
C12-B는 다음 실행 대상이다.
