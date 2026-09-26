# M08 — O 행쌍 보존 4-way ingress / 직접 F32 gather

2026-09-27. Branch `opt/m08-output-pair4-ingress`, parent `63a5c6d`.
Source는 E122 `c3553ab`, performance control은 별도로 `baseline/e122.json`이다.
[사전 연구](../../../../docs/research/o-weight-ingress-20260927.md)의 첫 후보를 실행한다.
기존 row-interleave family의 local producer/consumer replacement이며 새 architecture로 세지 않는다.

전체 graph·소유권·수명·추가 비용·선행 실패·종료 기준은 [recipe](pair4-v1/recipe.json)에 기록했다.
FP8 원본15MiB와 BF16/F32 numerical 경계를 유지하고 추가 weight/partial copy를 두지 않는다.
최종 gather의32×480B→예상1920×8B 분할 비용은 전체 O에서 반드시 포함한다.

첫 gate는 실제 emitted weight/input/F32 gather descriptor 전수 검사다. 통과하면 전체 O
marker로 weight-ready 및 gather를 같은 concurrent 실행 안에서 구분한다. 기존 marker가
충분하면 별도 isolated roundtrip microbench를 추가하지 않는다. 정보/trace 모드와 실제
전체 표준 host를 나눠 profiler 영향과 전체 cycles를 구분하며 앞뒤 control은 E122다.

**E125 / Arena105767: 기본 정확도 PASS, O +13.55% 회귀로 종료했다.** Live E122를 유지하며 추가 튜닝·채택·공식 제출은 없다.

## 첫 compile gate

`m08-pair4-v1` 첫 compile PASS. 정적 전체28259 vs E12227239(+3.7446%).
Weight13383 및 dot/reduce5063는 그대로이며 F32 gather가470→1490으로 증가했다.
실제 source address order가 바뀌었으므로 정적 weight 모델이 같은 값이라는 사실만으로
NPU ingress 효과를 판정하지 않는다. 이번 한정 진단은 그 질문과 gather 손실을 함께
관측하기 위한 예외다. 정적 회귀를 개선 후보라고 주장하지 않으며 NPU 회귀 또는5%미만
이득이면 추가 parameter sweep 없이 종료한다.

## NPU 판정

표준 host의 전체 O cycles는 E122 before51418 → M08 58095 → E122 after50906이다.
앞뒤 중앙값51162 대비 **+13.5511%** 회귀다. Candidate의3측정 모두6개 control측정보다 느렸다.
전체7stage가 완료됐고57비교 PASS(표준45 + O진단12)다. 서로 다른 입력은 기존3개뿐이며
같은 입력 반복을 확장 정확도나 독립 allocation으로 세지 않는다.

| O trace interval, 각6관측의 중앙값 | Control | M08 |
|---|---:|---:|
| Weight | 26769 | 27190 |
| Dot + K16 F32 reduce | 12019 | 12011 |
| F32 gather | 1201 | 8222 |
| Dot 완료→Task 완료 tail | 9384.5 | 16751.5 |

두 쪽의3runs×2clusters×22 engine marker와 Task max cycles를 검증했다.
Weight 단축은 관측되지 않았고, 같은15360B의 gather가32×480B에서1920×8B로
분할되면서 모든 관측에서 gather 구간이 크게 늘었다. 이는 이 producer/consumer
조합을 종료할 근거다. Marker에는 queue/dependency 대기가 포함될 수 있으며
각 구간의 중앙값 차이를 더해 전체 회귀나 물리 bandwidth를 역산하지 않는다.

[판정](decision.json), [표준 host 원본 집계](e125-m08-pair4-screen-job-105767-results.json),
[trace 분석](profile-analysis-m08-pair4-v1.json), [descriptor 전수검사](descriptor-analysis-m08-pair4-v1.json),
[독립 package·실제 upload 감사](independent-package-review.md),
[독립 NPU 해석](independent-npu-review.md)을 보존했다.
표준 host의 QKV/FFN은 E122 exact image다. O-only diagnostic control은 O image가
E122와 같은 과거 E104 profile host를 재사용한다. 실행하지 않는 FFN까지 E122라고
주장하지 않는다. 실제 upload tar와 준비 package의 각 member/hash 일치도 확인했다.

재검증:

```sh
python3 moa-dev/experiments/candidates/m08-output-pair4-ingress/verify_descriptors.py m08-pair4-v1 --check
python3 moa-dev/experiments/candidates/m08-output-pair4-ingress/analyze_profile.py m08-pair4-v1 --check
python3 moa-dev/experiments/candidates/m08-output-pair4-ingress/results.py e125-m08-pair4-screen 105767 --check
```

후속으로 [동결 C09 two-stage의 최초 NPU](../c09-output-first-npu/README.md)를 완료했다.
그 단회 −2.496% 결과는 별도 보존한다. M08 결과를 C09의 weight 완료 dependency
분할이나 row-interleave 전체 family의 기각으로 확대하지 않는다.
