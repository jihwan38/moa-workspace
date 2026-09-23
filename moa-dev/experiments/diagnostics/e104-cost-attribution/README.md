# E104 비용 진단

작업 브랜치: `diag/e104-cost-attribution`. Main 통합 commit `f9f905b`에서 시작했다.
비교 대상 source는 `baseline-e104` / `7f891f9c48e670fe014b2ac5bd2c92bae3bac433`다.
**기존 정적 schedule을 source에 대응시키는 첫 분석을 완료했다. 새 compile·NPU 진단은
아직 실행하지 않았다.** 최적화 후보 수나 NPU 측정 횟수에 이 작업을 추가하지 않는다.

## 확인한 실행 경로

[`ops::decoder_feedforward`](../../../../furiosa-opt-gemma4-12B/src/ops.rs#L212)는
pre-RMS → `feedforward_sharded` → post-RMS → residual/layer scale → HBM 저장을
수행한다. [`mlp.rs`](../../../../furiosa-opt-gemma4-12B/src/device/shared/mlp.rs)의
`prepare_ffn_fp8_split`이 세 E5M2 항을 만들고 `broadcast_ffn_quantized`로 복제한다.
`project_up_and_gate`는 FP4 weight를 E5M2 DM에 펼친 뒤 native dot/F32 partial,
block scale·합산을 수행한다. GeGLU 뒤 down은 FP4→FP8 DM→BF16 lookup/dot,
F32 partial의 scale·합산 경로를 쓴다.

기존 C07 build의 [원본 schedule](e104-ffn-schedule.json)을 hash 그대로 복사했다.
[분석기](analyze.py)는 채택 commit의 MLP source와 build source hash를 대조하고,
293개 instruction의 context interval 합집합과 긴 node 18개의 source·tensor 연결을
[static-audit.json](static-audit.json)에 남긴다. Source 선언 순서와 실제 schedule 순서는
다르며, 예를 들어 gate weight가 up보다 먼저 처리된다.

| E104 compiler 관측 | 정적 cycles |
|---|---:|
| 전체 makespan | 157045 |
| DMA context interval 합집합 | 126267 |
| Main context interval 합집합 | 94944 |
| Main과 DMA가 겹치는 구간 | 70771 |
| Up/gate weight load 각각 | 24612 |
| 세 weight의 FP4→FP8 materialization 각각 | 9263 |
| 세 항 입력의 ring32 방송 | 11783 |
| Up/gate native dot·input scale·partial 저장 각각 | 5684 |
| Down FP8→BF16 lookup·dot·partial 저장 | 18263 |

위 행들은 서로 겹치거나 포함 관계다. 합산한 값을 전체 latency 또는 절감 가능량으로
사용하지 않는다. 긴 node가 곧 critical path라는 뜻도 아니다. 기존 hardware control
[job 78730의 E104 before](../../../../docs/experiments/c13-v7-screen-e104-before.json)는
FFN 334969 cycles지만, 정적 시간과의 차이를 특정 stall 원인의 기여율로 해석하지 않는다.

## 다음 판별 질문과 범위

첫 질문은 **up/gate에서 weight 변환을 줄이는 설계와 입력 표현·분배 및 weight 전송을
함께 바꾸는 설계 중 어느 쪽을 먼저 전체 구조로 구현할 근거가 있는가**다. 기존 C07의
전체 성능 비교에는 표현·배치·reload·materialization 변화가 함께 있어 개별 비용을
분리한 증거로 승계하지 않는다. C13의 추가 INT8 변환도 같은 경로의 단순 개선이 아니다.

1. 위 긴 node의 producer/consumer를 view·alias와 자원 의존성까지 추적한다.
   특히 gate weight load가 28201에 끝난 뒤 materialization이 35834에 시작하는 간격,
   입력 방송 45097–56880 뒤 첫 native dot이 58586에 시작하는 간격의 선행 작업을
   확인한다. 시간 차이만으로 HBM stall이나 입력 대기로 이름 붙이지 않는다.
2. 정적 근거로 판별되지 않는 부분에 대해 E104의 **실제 크기·자료형·배치**를 복사한
   최소 NPU probe를 설계한다. Weight 준비 경로와 activation 분배 경로의 입력,
   출력 소비, 수치 검사를 명시하고 추가한 launch·출력·동기화 비용을 기록한다.
   지원 API만 확인하는 작은 shape의 속도로 full shape를 대신하지 않는다.
3. Weight 준비 비용이 큰 경우에는 materialization/transfer를 함께 줄이는 조합을,
   입력 분배 비용이 큰 경우에는 표현·복제·operand 소유권을 함께 바꾸는 조합을
   다음 전체 후보로 검토한다. 고립된 두 시간이 비슷하거나 전체와 대응되지 않으면
   제한을 남기고, 관련 두 경로를 결합한 한 번의 확인으로 범위를 제한한다.

분리 실행은 원래의 overlap·DM 배치·캐시 상태를 바꾸므로 비용 기여율의 직접 측정이
아니다. 진단용 사전 변환으로 빠른 수치를 얻어도 제출 가능한 최적화로 세지 않는다.
선택한 전체 후보의 변환·통신·정확도와 동시기 control 비교가 최종 판단 근거다.
새 구조는 E74/E79나 다른 과거 조합에서 출발할 수 있으며 E104는 비교 대상으로 둔다.
C14 stage-local compact 전송 수정은 여전히 미구현인 선택지다.

## 재현

저장소 루트에서 아래 명령을 실행한다. Git에 저장된 source와 schedule만 읽으며,
Docker·compiler·Arena를 호출하지 않는다.

```bash
python3 moa-dev/experiments/diagnostics/e104-cost-attribution/analyze.py --check
```

분석기를 변경한 뒤 JSON을 재생성할 때만 `--check`를 생략한다. 기존 source/schedule
hash와 build의 makespan 검사를 제거해서 다른 artifact를 같은 E104로 해석하지 않는다.
