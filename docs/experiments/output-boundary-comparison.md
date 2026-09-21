# 같은 입력에서 원본 O·E47·E53 비교

원본 upstream O와 BF16 activation을 유지하는 E53은 8개 경계 검사를 모두 통과했다. E47은 case7의 3,840개 출력 모두 허용오차를 벗어났다. **동일 host source·runner·CPU reference payload**를 사용했으며, 각각의 업로드 해시와 동결 device image를 확인했다.

| case | 원본 job70799 최대오차 | E47 job70785 최대오차 | E53 job70810 최대오차 | 결과 |
| --- | ---: | ---: | ---: | --- |
| 0: ±0 | 0 | 0 | 0 | 모두 PASS |
| 1: column/row 경계 | 0.015625 | 0 | 0 | 모두 PASS |
| 2: amax power2 아래 | 0.0234375 | 0 | 0 | 모두 PASS |
| 3: amax power2 동일 | 0.033203125 | 0 | 0 | 모두 PASS |
| 4: amax power2 위 | 0.03515625 | 0 | 0 | 모두 PASS |
| 5: 작은 성분 경계 | 0.015625 | 0 | 0 | 모두 PASS |
| 6: tiny 입력 | 1.203706215e-35 | 1.015627119e-35 | 1.015627119e-35 | 모두 PASS; tiny 보존 증명 아님 |
| 7: correlated cancellation | **0** | **0.109375** | **0** | 원본/E53 PASS, E47 3840/3840 FAIL |

Case7의 expected는 모든 행 BF16 `0.609375(0x3f1c)`다. 원본과 E53은 3,840개 모두 정확히 일치했다. E47은 모두 `0.5(0x3f00)`였고, 오차 .109375가 허용값 .05609375보다 크다. 세 job 모두 8-case 전체가 끝났으며 각각 입력 checksum 40/40, nonfinite=0이다. E47 exit1은 수치 비교 실패이고 원본/E53은 exit0이다.

원본 device image는 commit `8559916051162e5fd682c29ec0fe3e9cc6e420cf`의 무변경 src43개에서 빌드했으며 초기 원본 실측 host의 O image와 byte-identical한 `434f864a…`다. E47은 공식 실측과 같은 `00d8a969…`, E53은 parent가 동결한 `35947381…`다. 세 standalone host 모두 해당 image가 정확히 한 번 들어 있다.

E53은 E47의 16-column layout/후단을 사용하며 activation을 BF16 TRF에 유지하고, FP8 weight lookup→BF16 contraction→FP32 column 합→FP32 channel scale 후 처음 BF16으로 cast한다. 따라서 FP8 activation quantization과 scale 이전 추가 BF16 경계를 함께 없앴다. 이 결과는 case7의 회복을 확인하지만 내부 단계 하나만 바꾼 인과 실험이라고 주장하지 않는다. E47 출력이 작은 성분의 FP8 소실을 계산한 독립 수학 모델과 일치한다는 증거도 함께 보존한다.

이는 공식 seeded score 측정이 아니다. case7은 각 값이 공식 generator의 표현 가능한 support 안인 상관 배열이며 실제 PRNG seed의 출력이라고 주장하지 않는다. Case4 일부 값은 공식 활성값 범위 밖이다. Case6은 output이 atol보다 극히 작아, 통과로 FTZ나 tiny bit 보존을 판단할 수 없다. 현재 경계 suite를 넘어 전체 입력 정확도를 증명하지 않는다.

원자료: [원본 결과](original-output-boundary-control.json), [E47 결과](e47-output-boundary-probe.json), [E53 결과](e53-output-boundary-probe.json). 재현: [원본 control](../../moa-dev/experiments/candidates/original-output-boundary-probe-control/README.md), [8-case 설계](../../moa-dev/experiments/candidates/output-boundary-probe-e47/README.md), [E53 probe](../../moa-dev/experiments/candidates/bf16-output-boundary-probe-e53/README.md), [수학 감사](../research/output-boundary-design-audit.md).
