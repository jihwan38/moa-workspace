# C13 — FFN의 INT8 곱셈과 I32 누산

**v7은 전체 컴파일·기존 NPU 기본 정확도 검사를 통과했다. FFN은 동시 E104보다
18.58% 느려 미채택으로 종료했다.** 브랜치 `opt/c13-int8-ffn`, E104 runtime 유지.
[최종 판정](decision.json) · [최종 소스](integer-v7/mlp.rs) ·
[실측 E108](../../../../docs/experiments/e108-c13-integer-v7.json).

| 같은 Arena job 78730의 실행 순서 | FFN median cycles | 기본 정확도 |
|---|---:|---|
| E104 before | 334,969 | 15비교 PASS |
| C13 v7 | 397,318 | 15비교 PASS |
| E104 after | 335,156 | 15비교 PASS |

각 process는 변경 없는 fixture의 3개 입력을 실행했다. FFN은 11,520개 출력 모두
기존 `atol=0.01 + rtol=0.01*abs(expected)`를 만족했다. +18.58%는 두 control
median의 중앙값 대비 변화이며, before 대비는 +18.61%다. 독립 반복이나 공식 점수
주장이 아닌 단회 screen이다. 개선 후보가 아니므로 추가 튜닝·확대 검증·공식 제출은
하지 않았다. [원본 로그](evidence/c13-v7-screen/arena-78730.log).

## 컴파일을 재개해 고친 부분

처음 v2에서 scale DMA 정렬 실패를 이유로 중단했다. 사용자는 수정 횟수만 세어
구현 오류를 방치한 판단을 지적했다. [초기 판정](decision-initial-v2.json)은 보존하며,
AGENTS.md에는 수정 가능한 구현 오류와 실제 SDK 제약·성능 탈락을 구분하도록
보완했다. [재개 근거](reopening.json). 아래는 같은 C13 구조의 수정 과정이다.

| 버전 | 결과와 수정 |
|---|---|
| v1 | `reduce<H/L>`가 유지할 block 축까지 없앰 |
| v2 | 16개 내부 원소만 별도 축으로 reduce; 다음 scale gather의 4B stride가 8B DMA 정렬에 걸림 |
| v3 | scale을 4개씩 pack하려 했으나 F32 transpose는 2행까지만 지원 |
| v4 | scale 두 개를 8B로 pack해 정렬 해결; 60개 H quant slice의 padding을 잃는 fetch-sequencer 내부 오류 발생 |
| v5 | anonymous high padding을 명시해도 같은 오류 |
| v6 | 네 개의 이름 있는 동일 입력 복제로 256-slice 배치를 유지해 전체 compile 성공; NPU FFN 출력은 non-finite |
| v7 | integer unzip/zip 뒤의 F32 scale 복원을 별도 pipeline으로 분리; 전체 NPU 기본 검사 PASS |

v6은 host·fixture·기준을 바꾸지 않고 실행한 첫 전체 image다. 앞뒤 E104는
정상이었지만 FFN 3입력 모두 실패했다. [E107](../../../../docs/experiments/e107-c13-integer-v6.json)의
실패 시간은 유효 성능 집계에서 제외한다. 이를 컴파일 성공만으로 검증됐다고 보지 않았다.

원인을 구분하려고 실제 quantizer body를 복사한 별도 진단을 실행했다.
FP4 16개 code 변환과 일반 H/L 입력의 정수 항·scale은 일치했다. K=3840의
정수 dot만 실행하면 960개 결과 모두 맞지만, 같은 pipeline에서 narrow와 per-block
VRF scale 곱셈을 붙이면 956개가 틀렸다. F32 부분합을 DM에 저장하고 새 tag-zero
stream에서 scale을 곱하자 진단의 960개가 모두 맞았고 전체 fixture도 통과했다.
[결합 실패](c13-diagnostic-v6c-results.json) · [정수 dot 분리](c13-diagnostic-v6d-results.json) ·
[수정한 결합](c13-diagnostic-v7-results.json).
정확한 compiler/ISA 내부 원인까지 확정한 것은 아니다.

진단 전체를 PASS로 표시하지 않았다. 모든 0 입력에서 하드웨어는 수학적 복원 scale
`2^-134`를 0으로 내보냈고, zero digits는 맞았다. 각 진단에 이 subnormal 차이
1,200개를 그대로 기록했다. 일반 입력/정수 dot 검사 통과와 이 raw-scale bit 차이를
분리한다. 전체 BF16 입력의 동치나 확대 정확도 검증을 주장하지 않는다.

## 구현한 계산과 비용

출발 source는 E74 `9e2058ec0947498d7cb693fbe98e7b5f0b5c9ea7`, 성능 비교군은
E104다. E74의 F32 block 합산과 BF16 비선형 경계, C07의 lane/unzip 기법,
C04의 down ring32를 조합했다. E104의 global input scale과 FP8 세 항은 쓰지 않는다.
행 소유권과 전체 GeGLU 완성 후 down을 시작하는 의존성은 남겼다.

실제 경로는 `feedforward_sharded → quantize_h → project_up_and_gate → geglu →
project_down_sharded → quantize_l`다. up/gate/down 세 projection 모두를 바꾸며,
공개 decoder 인터페이스와 RMS/residual/layer scale 및 host 입력 형식은 유지한다.

각 16개 BF16 입력에서 `s = 2^floor(log2(max(amax,2^-120))) / 64`를 구하고,
`hi = clamp(round(x/s),-127,127)`, `lo = round((x/s-hi)*128)`로 표현한다.
반올림은 SDK의 half-away-from-zero다. FP4 weight는 정확한 정수 `2w`로 변환한다.
I32에서 `128*dot_hi + dot_lo`를 계산하고 F32로 바꿔 DM에 저장한다. 다음 F32
단계가 `s/256`을 곱하고, 기존 weight block scale과 합산을 거쳐 BF16 projection을
만든다. 기존 GeGLU의 BF16 결과를 다시 같은 방식으로 양자화해 down에 넣는다.

블록 정수 상한 `16*12*(128*127+64)=3,133,440`은 I32와 F32 정확 정수 범위 안이다.
Activation 근사와 F32 합산 오차는 별개다. [CPU 계산식 screen](numeric-screen.json)은
기존 11,520개 출력을 통과했지만 NPU mapping 검증을 대신하지 못했다.

v7 정적 schedule은 187,947 cycles다. Weight는 FP4→FP8→F32→INT8로 변환하며
이 비용은 모두 timed kernel에 들어간다. 정적 schedule의 weight materialization
세 연산은 각각 17,559/17,559/16,479 cycles, E104는 각각 9,263 cycles다.
[비용 기록](weight-conversion-cost.json). 배치 차이와 연산 중첩이 있으므로 이 값을
합산해 NPU 소요 시간이나 정확한 병목 기여율로 해석하지 않는다. 재시도하려면
변환·양자화 비용을 바꾸는 구체적인 설계가 필요하며, INT8 계열 전체를 폐기하는
결론은 아니다.

## 기록과 재현

한 구조의 source variant 7개, exact compile 호출 9회(같은 v4의 backtrace 재실행과
v6 package용 재컴파일 포함), 전체 host package 2개를 기록했다. 별도 진단 host
build는 6회이며 2회는 Rust host 작성 오류, 4회는 실행 가능한 진단이다.
진단/지원 검사 수를 독립 아키텍처 수로 세지 않는다.

`build-*.json`은 source origin·실제 container 입력·image·package hash를 기록한다.
`evidence/`에는 각 실패 source/log, 성공 schedule, Arena 로그를 보존한다.
E104 control package가 선언한 baseline source manifest와 같고, C13의 변경 없는
QKV/O image도 E104와 동일함을 확인했다. SDK/CLI 0.8.1과 pinned Cargo.lock을 사용했다.

```bash
python3 moa-dev/experiments/structural_build.py c13-int8-ffn integer-v7 --plan
python3 moa-dev/experiments/structural_build.py c13-int8-ffn integer-v7 --label NEW-LABEL --package
python3 moa-dev/experiment_index.py --check --check-live
```

C14도 이전 lowering 실패만으로 불가능하다고 판단하지 않는다. 실제 v3 코드와
E74의 compact/gather를 대조한 [후속 검토](c14-assessment.md)를 남겼다.
C14 수정본을 이번 작업에서 컴파일·실행한 것은 아니다.
