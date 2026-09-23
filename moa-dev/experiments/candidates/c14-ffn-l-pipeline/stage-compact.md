# C14-B: stage 내부 compact 후 전체 실행

브랜치 `opt/c14-stage-compact`. E74 source에서 출발한 C14-A v3의 전체 graph를
유지하고, down에 전달하기 전에 각 stage·cluster의 유효3840개 BF16 값만 compact했다.
이 수정으로 기존3840/4096 padding lowering 오류를 해결했고 전체 compile이 통과했다.
전역 L15360개를 모으는 단계로 되돌리지 않았다.

**미채택: 기본 NPU 정확도는 통과했지만 FFN이 동시 E104보다17.58% 느렸다.**
[정확한 source/recipe](bf16-stage-compact-v4/recipe.json) ·
[소스](bf16-stage-compact-v4/mlp.rs) · [판정](decision-stage-compact.json).

| Arena job 79238 실행 순서 | FFN median cycles | 기본 정확도 |
|---|---:|---|
| E104 before | 336341 | 15비교 PASS |
| C14-B | 397348 | 15비교 PASS |
| E104 after | 339553 | 15비교 PASS |

FFN 3입력/11520출력이 기존 tolerance를 통과했다. 새 source variant1개, exact compile
2회(동일 소스 package용 포함), host build1회, Arena allocation1회다. 후보·control
프로세스들은 독립 job 반복이 아니다. 회귀가 커서 추가 parameter sweep·확대 검사·
공식 제출을 하지 않았다. [결과 JSON](../../../../docs/experiments/e109-c14-stage-compact.json),
[원본 로그](evidence/c14-compact-screen/arena-79238.log).

## 커진 비용과 해석 범위

Compiler 일정에서 첫 stage down dot은77001–86264에 수행됐고, 둘째 stage의
up/gate dot은124253에야 완료됐다. 조기 소비 가설 자체는 표현됐다. 하지만 stage를
나누면서 matrix decode stream6개, stage별 입력 분배2개와 F32 부분합 처리가 생겼다.
이 구현은 E74의 BF16 up/gate dot을 사용하므로 E104 native FP8 계산과도 다르다.

[정적 비용 비교](stage-compact-cost.json): makespan157045→160081(+1.93%),
Main 점유 합집합94944→121095, DMA126267→139308, Sub19827→30929다.
여러 엔진의 시간은 겹치므로 더해서 지연을 계산하지 않는다. 정적 변화와 실측
+17.58%는 다르며, 정확한 하드웨어 stall 기여율까지 분리한 것은 아니다.
동시 E74 대조를 실행하지 않았으므로 +17.58%를 pipeline 하나의 효과라고 말하지 않는다.

재시도하려면 stage별 lookup·전송·계산·부분합 중 비용을 줄이는 구체적인 새 조합이
필요하다. C14 전체 계열이 불가능하거나 느리다고 판정하지 않는다. 현재 runtime은
E104이며 C13 등 다른 후보 구현을 runtime으로 옮기지 않았다.

## 재현과 근거

```bash
python3 moa-dev/experiments/structural_build.py c14-ffn-l-pipeline bf16-stage-compact-v4 --plan
python3 moa-dev/experiments/structural_build.py c14-ffn-l-pipeline bf16-stage-compact-v4 --label NEW-LABEL --package
python3 moa-dev/experiments/structural_screen.py c14-ffn-l-pipeline prepare NEW-SCREEN NEW-LABEL
python3 moa-dev/experiments/structural_screen.py c14-ffn-l-pipeline submit NEW-SCREEN
```

Terminal status 이후 `collect NEW-SCREEN`을 실행한다. Shared runner는 실제 E104
control source manifest, host/fixture/runner, candidate recipe와 내려받은 업로드
archive를 검사한다. 이번 QKV/O image는 E104와 byte-identical하다. 변경 없는
공식 benchmark와 fixture를 사용했다. Source/log/schedule은 `evidence/`에 보존했다.
