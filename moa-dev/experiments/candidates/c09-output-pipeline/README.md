# C09 — O의 두 행 구간 전송·계산 pipeline

**컴파일 성공, 정적 일정 +0.71%로 종료. 채택하지 않는다.** 후보 compile 1회와
E104 control compile 1회만 수행했다. Arena 정확도·성능은 측정하지 않았다.
[판정](decision.json) · [정적 근거](static-screen.json) · [사전 가설](candidate.json).

E104 `7f891f9`의 `project_output`을 [two-stage-v1/projection.rs](two-stage-v1/projection.rs)에서
변경했다. H3840을 두 cluster·16 row group·120 local row로 해석하고, 각 group의
0–59/60–119행을 별도 FP8 DM allocation으로 전송한다. `project_output_stage`가
동일한 BF16 lookup→dot→16-column F32 합산을 수행해 기존 result의 대응 구간에 쓴다.
후속 F32 channel scale→BF16→RMS→residual은 원본과 같다. 논리 row 전수 coverage는
[logical-layout.json](logical-layout.json)에 있으며 NPU correctness 증명은 아니다.

| 정적 항목 | E104 | C09 |
|---|---:|---:|
| 전체 makespan | 27,239 | 27,432 |
| instruction 수 | 32 | 49 |
| 전체 HBM load 명령 수 | 6 | 8 |
| weight load duration 합 | 13,383 | 13,932 |
| lookup/dot duration 합 | 5,063 | 5,326 |

첫 lookup/dot은 9740–12403, 두 번째 weight load는 10578–17544로 **1825 cycles의
중첩이 실제 compiler schedule에 존재한다**. 그러나 weight load와 lookup/setup 반복
비용이 늘어 전체 일정은 짧아지지 않는다. 표의 자원별 duration은 겹치므로 합산해서
latency나 기대 절감량으로 해석하지 않는다. 이 구현에서 큰 이득의 근거가 없어
host package·Arena·tile sweep을 생략하고 C11로 이동한다. 모든 가능한 staging
구조가 느리다는 결론은 아니다.

Control exact image의 SHA-256이 채택 E104 O image와 같다. 두 build 기록은 전체
compile 입력 해시와 container 복사 검증, exact image/schedule hash를 보존한다.
`evidence/`에는 실제 변경 소스와 compile 로그를 동결했다. 나머지 소스는 기록한
E104 commit과 lock에서 재구성한다. Raw schedule/image는 ignored artifacts에 있다.

```bash
python3 moa-dev/experiments/structural_build.py c09-output-pipeline two-stage-v1 --label NEW-CANDIDATE
python3 moa-dev/experiments/structural_build.py c09-output-pipeline control-e104 --label NEW-CONTROL
# 기록된 원래 두 build의 schedule을 다시 분석:
python3 moa-dev/experiments/candidates/c09-output-pipeline/analyze.py
```

기존 build label은 덮어쓰지 않는다. Live runtime과 공식 제출은 E104 그대로다.
