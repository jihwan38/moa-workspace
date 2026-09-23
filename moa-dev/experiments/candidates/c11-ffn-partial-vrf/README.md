# C11 — Native FP8 dot의 F32 partial을 VRF에 유지

**컴파일 성공, 정적 일정 +3.55%로 종료. 미채택·Arena 미실행.** 후보 1회와
E104 control 1회를 compile했다. [판정](decision.json) · [정적 근거](static-screen.json).

[two-row-v1/mlp.rs](two-row-v1/mlp.rs)의 `project_rows_with_scale_contraction`만 E104에서
바꿨다. FP4→E5M2 weight DM 변환은 기존처럼 tile 밖에서 수행한다. 각 두 행에 대해
native DPE→입력 scale 복원→`to_vrf(&mut device.sub)`로 F32 partial을 유지한 다음,
FP8 block scale을 F32로 읽어 partial VRF와 곱하고 H축을 합산한다. 새로운 BF16 partial
rounding은 없다. Up/gate gather·GeGLU·down·후단은 기존 구현이다.

조사 당시 3행은 padded VRF 용량 상한 계산이었다. 첫 빌드 전 출력 F32 commit의
8바이트 정렬을 고려해 **2행/3,840B**로 정했다. 3행 결과를 담는 추가 padded DM은
만들지 않았다. 이 작업에서 tile 크기를 반복 탐색하지 않았다.

| 정적 항목 | E104 | C11 |
|---|---:|---:|
| 전체 makespan | 157,045 | 162,624 |
| instruction 수 | 293 | 490 |
| Up/gate native DPE 실행 수 | 2 | 30 |
| 해당 DPE duration 합 | 11,368 | 19,290 |
| Block scale/reduce 실행 수 | 10 | 30 |
| 해당 scale/reduce duration 합 | 6,400 | 12,060 |
| Main 점유 interval union | 94,944 | 108,526 |
| Sub 점유 interval union | 19,827 | 34,420 |

스케줄에서 두 기존 dot의 output은 SRAM이고, 후보의 30개 output은 모두 VRF다.
원한 중간 저장 제거 경로는 구현됐다. 그러나 작은 dot/scale 실행이 반복되고 Main→VRF
쓰기 노드가 **Main·Sub·VE를 동시에 점유**한다. 이 구현의 전체 일정은 더 길어져
추가 수정·host package·Arena를 진행하지 않는다. 표의 자원별 시간을 더해 latency나
제거한 비용으로 해석하지 않는다. VRF 방식 전체가 불가능하다는 판정도 아니다.

Control exact image가 E104 FFN image와 동일함을 확인했다. NPU 정확성은 검증하지
않았으므로 이 구현을 실행 소스로 채택하지 않는다. 원래 scale과 partial의 대응을
설계상 보존했지만 compiler 성공이 좌표·정밀도 검증을 대신하지 않는다.

```bash
python3 moa-dev/experiments/structural_build.py c11-ffn-partial-vrf two-row-v1 --label NEW-CANDIDATE
python3 moa-dev/experiments/structural_build.py c11-ffn-partial-vrf control-e104 --label NEW-CONTROL
python3 moa-dev/experiments/candidates/c11-ffn-partial-vrf/analyze.py
```

각 build의 소스·로그는 `evidence/`, 전체 입력·image·schedule hash는 build JSON에 있다.
나머지 입력은 E104 commit과 lock에서 재구성한다. 다음 후보는 C10이다.
