# 실험 문서 상태·링크 확인

2026-09-21. 결과 collector JSON과 후보 source hash를 대조해 문서의 미검증 상태를 갱신했다. 새 build/NPU 실행, source 수정, Git mutation은 하지 않았다. 첫 검사에서는 `docs/OPTIMIZATION_PLAN.md`와 이 디렉터리 `README.md`를 읽기만 했다. 이후 원장 정리 요청에 따라 `docs/experiments/README.md`와 이 QA 문서만 추가 갱신했다. 루트 `README.md`, `docs/OPTIMIZATION_PLAN.md`, source 및 result JSON은 이번 정리에서 수정하지 않았다.

| 대상 | 확인한 상태와 범위 |
| --- | --- |
| E35 full256 | [70485](e35-full256-native.json) 공식15개 PASS, FFN811,219. packet4 자체가 불가능하다는 주장을 반박하지만 timeout의 단일 원인을 확정하지 않는다. 기존 문서 상태는 정확했다. |
| E36 FFN 후단 | [70498](e36-ffn-parallel-epilogue.json) 공식15개 PASS, FFN378,021과 [70506](e36-ffn-epilogue-extra-correctness.json) 추가30개 PASS를 후보 README/evidence에 반영했다. 후보 MLP·ops·residual이 frozen E36과 각각 byte-identical이다. |
| E37 kernel별 RMS | [70519](e37-kernel-specific-rms.json) 초기 Q94,094/O45,796/FFN376,040과 [3회 반복](e37-combined-repeats.json)의 median-of-job-medians Q94,195/O45,748/FFN379,741을 구분했다. [70544 추가30개](e37-combined-extra-correctness.json)도 PASS다. 후보 ops/RMS source가 frozen E37과 일치한다. |
| E38 weight+scale direct | [70525](e38-full256-directcommit.json) 공식15개 PASS, Q96,579/O45,101/FFN482,511. `mlp.direct-scales.rs` SHA `7c363a7f…`가 frozen E38 MLP와 일치한다. **같은 폴더의 weight-only `mlp.rs`에는 이 결과를 적용하지 않았다.** |
| E39 tail-only / E40 row2+tail | 첫 QA에서는 build/capture 상태였다. 후속 [E39 job70551](e39-native-tail-zero.json)은 공식15개 PASS, FFN415,457이며 [E40 job70552](e40-row2-tail-zero.json)는 FFN non-finite 실패다. 원장에는 각각 실측 상태를 유지하고 E40 cycles를 유효 성능으로 사용하지 않았다. |
| E41 row30 / E42 row30-row256 | [E41 job70562](e41-native-row30.json)은 공식15개 PASS, Q96,917/O46,477/FFN426,442인 단일 screen이다. [E42 job70576](e42-native-row30-row256.json)은 첫 확인 시 pending이었다. 후속 root 갱신에서 공식15개 PASS, Q98,728/O45,794/FFN456,849를 확인해 원장에 반영했다. |
| E27 / E29 반복 범위 | E27은 원래 package의 단일 screen과 별도 추가 입력 검증을 구분했다. [E29 control](e29-qkv-input-control-repeats.json)은70428/70482/70493 세 job PASS, median94,491/49,228/388,311이며 원장에 반복 행을 추가했다. |
| E31 cacheguard | [70507](e31-cache-preservation-correctness.json) 기존15개 출력·sentinel12개 PASS, 호출 후 비대상12,570,624개 BF16 exact 비교 mismatch0을 README에 반영했다. 고정 build manifest는 수정하지 않았다. |
| RMS Switch-gather helper | E26-base standalone package 미실행과 E31 통합 package의 공식·추가·반복 PASS를 구분했다. QKV에는 E29 inter-slice helper가 반복 실측에서 유리해 E37이 이를 선택한다는 후속 결론도 연결했다. |

첫 QA는 실험/candidate Markdown92개·로컬 링크183개·JSON291개를 확인했다. 원장 최종 갱신 후 재검사는 workspace의 `rg --files -g "*.md"`로 발견한 Markdown 전체와 `docs/**/*.json`, `moa-dev/experiments/candidates/**/*.json`으로 범위를 넓혔다. 최신 개수와 확인 시각은 아래에 기록한다.

검사한 Markdown은 inline link/image와 reference-style link의 **로컬 파일 존재 여부**다. URL은 decoding하고 `#anchor`와 파일 뒤 line-number suffix는 파일 검사에서 제외했다. 외부 URL 연결 상태, heading anchor의 정확성, HTML 속성, code fence 안의 예제 링크는 검사하지 않았다. Git-ignored artifact/target 내부 문서는 열거 대상에서 제외했다. 링크 목적지가 현재 workspace의 ignored artifact를 가리키면 존재 여부는 확인하되, 새 clone에도 그 artifact가 있다는 보증은 하지 않는다. JSON은 표준 parser 문법 검사이며 schema/수치 정합성 전체 검사가 아니다.

최종 재검사 시각 `2026-09-21T11:34:26.577459+00:00`: Markdown **111개**, 로컬 링크 **286개**, JSON **307개**에서 누락 파일·JSON syntax 오류는 **0건**이다. 외부 링크 152개는 위 범위대로 건너뛰었다. 검사 스크립트와 출력은 임시 `/tmp/moa-local-doc-qa.py`, `/tmp/moa-local-doc-qa-result.json`에 있으며 결과 JSON을 수정하지 않았다.

[공식 제출 상태 기록](official-submission-status-20260921.json)의 평가 기준 maintenance 안내와 공식 score/rank 미확정 범위를 유지했다. Arena PASS와 진단용 score 추정을 공식 순위로 바꾸지 않았다. 결과는 각 immutable package와 측정 입력에 한정하며, 새 후보의 보편적 정확성을 증명하지 않는다.
