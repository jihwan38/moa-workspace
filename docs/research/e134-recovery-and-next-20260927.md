# E134: 실패 원인 수리 후 회수한 개선과 다음 판단

현재 source는 E134 `baseline-e134` / `4f8fe7f`다. 성능 비교군은 E128이며,
E133 v2가 정밀도 수리의 source origin이다. [채택 구성](../../baseline/e134.json)과
[전체 실험](../../moa-dev/experiments/candidates/ffn-native-down-weighted-residual/README.md)을 기준으로 한다.

## 이번 결과가 답한 질문

큰 저장을 제거한 후보가 느리거나 부정확했다고 그 구조 전체를 폐기할 필요는 없다.
다만 남은 비용이나 오차를 실제로 바꾸는 수리와, 같은 구조의 타일 반복을 구별해야 한다.

| 시도 | 확인한 원인·제약 | 후속 판단 |
| --- | --- | --- |
| E131 joint direct FFN | 입력 HBM이 up-scale 전송 뒤에 시작하고 첫 native도 지연됨. 뒤에서 일부 시간을 회수하므로 marker 차이를 더할 수 없음 | row sweep 종료. 발행·소비 의존성을 바꿀 구체적 경로가 있을 때 재개 |
| E132 O K4 | gather 절감보다 입력 준비·첫 weight 지연이 큼. source 생성 순서를 바꾼 v3는 바이너리도 동일 | 무효한 source 순서 변경을 NPU에서 반복하지 않음 |
| E133 native down | 전체 FFN 성능은 개선. 같은 scale의 E5 잔차가 작은 값을 소실 | positive를 보존하고 표현과 복원을 함께 수리 |
| E134 weighted native down | 잔차 확대와 F32 계수 복원 후 알려진 raw-down·full-FFN 반례 통과, 추가 비용을 포함해 전체 성능 개선 | 고정 v3 채택, 파라미터 탐색 종료 |

E134는 down의 전체 decoded-weight DM 쓰기·읽기를 각각56.25MiB 없앤다.
대신 실제 GeGLU 출력의 amax·3항 양자화·방송·계수 준비·native F32 term 복원이 들어간다.
Down F32 block partial 쓰기·읽기 각14.0625MiB와 up/gate FP8 weight 저장은 남는다.
관측한 whole FFN 감소율 중앙값4.451846%는 이 모든 변화를 포함하며,
제거한 논리적 바이트만의 독립 효과나 측정된 메모리 버스 트래픽 감소율이 아니다.

표준9입력·실제down9case34560출력·구성full-FFN1입력을 검증했다.
원래 허용오차를 유지한 full-FFN 비교에서 E133은1920개 실패,
E128/E134는3840개 모두 BF16 정답과 일치했다. 이 유한한 검증은 임의입력이나
전체모델 정확성 증명이 아니다. 원본 pending 기록은 보존하고 별도 채택 결정을 남겼다.

## 다음 선택의 범위

FFN4.451846% 감소만 있고 QKV/O가 같으면 기하평균 점수 증가는 약1.5296%다.
이 산술을 수상권 진입의 증거로 사용할 수 없다. 새 대형 QKV/O 구조가 지원되고
빠르다는 근거도 아직 확보하지 못했다. 큰 구조 연구와 작은 수리의 우선순위를 섞지 않는다.

조건부로 남은 국소 질문은 E132 K4의 반복 입력 HBM 공급을 compact seed와 Switch 복제로
교체하는 것이다. `two-stage-v2/projection.rs::project_output`의 입력 복제를
`[ring8,1#8,K4]` seed → `CustomBroadcast32` → `[ring8,8,K4]` DM → 기존 Sub TRF로 바꾼다.
E134 `broadcast_down_quantized`의 anonymous8/K4 형태와 기존 BF16 방송 경로는 참고가 되지만,
정확한 O 조합은 미구현·미검증이다.

논리 HBM 입력은1MiB→128KiB로 줄지만 seed DM읽기128KiB, Switch 출력DM쓰기1MiB,
bitmap과 Main 작업이 추가된다. 기존 ingress 대비 DM쓰기 순증은128KiB이고
weight15MiB와 TRF 적재는 유지된다. 따라서 새 아키텍처나 큰 개선으로 이름 붙이지 않는다.
재개한다면 emitted bitmap source `32*(slice//32)+slice%4`, HBM128KiB와 K1024 보존을
먼저 확인하고, 첫weight·최종dot·whole O를 함께 측정한다. 입력 DMA만 짧아지고
whole O가 빨라지지 않으면 그 수리는 채택하지 않는다.

이 문서는 후속 후보를 실행했다는 기록이 아니다. 새 실험은 E134 채택·제출 결과를 통합한
main에서 시작하고, 실제 source recipe와 성능 control을 다시 명시해야 한다.

공식 제출892178e8까지 완료해51/69위가 됐다. 현재20위·10위와의 수치상 격차는
[공식 결과 기반 계산](e134-rank-gap-20260927.json)에 별도로 남겼다.
