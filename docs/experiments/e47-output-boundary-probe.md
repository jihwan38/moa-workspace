# E47 O 경계 검사: cancellation 반례 재현

Arena **job70785**는 8-case를 모두 완료하고 수치 비교 실패로 exit1을 반환했다. timeout/launch 실패가 아니다. case0–6은 통과했으며, case7의 **3,840개 출력 전부**가 expected `0.609375` 대신 `0.5`여서 허용오차를 넘었다. nonfinite는 전체 0개다.

| case | 목적 | 결과 | 최대 절대오차 |
| --- | --- | --- | ---: |
| 0 | 교대 +0/-0 | PASS | 0 |
| 1 | 256-column 및 output row 경계 | PASS | 0 |
| 2 | amax가 2의 거듭제곱 바로 아래 | PASS | 0 |
| 3 | amax가 정확히 2의 거듭제곱 | PASS | 0 |
| 4 | amax가 바로 위 BF16 | PASS | 0 |
| 5 | 작은 성분 복원/underflow 경계 | PASS | 0 |
| 6 | amax<2^-119 floor | PASS | 1.015627119e-35 |
| 7 | 작은 성분 누적과 cancellation/RMS | **FAIL 3840/3840** | **0.109375** |

Case7은 각 256개 activation이 `[+1,-1,254×2^-22]`, 모든 weight row가 `[1,1,254×240]`의 반복이다. channel scale=`2^-10`, RMS gamma=residual=`0.5`다. 각 값은 공식 generator의 개별 표현 가능 범위 안이지만, 실제 PRNG seed의 출력으로 찾은 배열은 아니다. 의도적으로 상관시킨 진단 입력이다.

원래 CPU F32 matmul은 `0.2325439453125`, scale 후 BF16은 `0.0002269744873046875`, RMS/residual 후 expected는 `0.609375`다. 현재 FP8 분해 모델은 작은 성분을 소실시켜 최종 `0.5`를 예측하며 실제 NPU와 일치했다. 이 결과로 해당 입력의 오차를 확인했으며 전체 backend 원인을 단정하지 않는다. 원본 upstream의 동일 fixture 대조 job70799는 8-case 모두PASS했고 case7도3840개exact였다([비교](output-boundary-comparison.md)).

검증된 업로드 host는 E47 O image `00d8a969edf26d5a34fbf9bc5430cdd889f60fa58f122ebcbd7aa2208f584353`를 정확히 한 번 포함한다. 다운로드한 `test_runtime`, runner, fixture의 SHA256이 동결 manifest와 모두 같다. `job.yaml`은 원래 runner/no args/no env이며 host_only=false를 실행 로그에서 확인했다. 입력 checksum 40개, launch/result 8개, 120행 단위 실패 histogram, 대표 BF16 bits와 첫 32개 mismatch를 payload oracle에 연결해 검증했다.

Case6의 기대값은 atol .05보다 훨씬 작아서 모두 zero여도 PASS할 수 있다. 따라서 floor case PASS는 tiny 값 보존 증명이 아니다. Case4의 1.0078125는 공식 활성값 범위 밖이지만 유한 BF16이다. 이 suite는 공식 성능·순위 측정과 분리된 관측이며 모든 입력 정확도를 증명하지 않는다.

원자료·전체 수치·해시는 [실측 JSON](e47-output-boundary-probe.json), 재현은 [후보 README](../../moa-dev/experiments/candidates/output-boundary-probe-e47/README.md), 수학 검토는 [design audit](../research/output-boundary-design-audit.md)에 있다. collector가 새 NPU job을 만들거나 source/binary를 변경하지 않았다.
