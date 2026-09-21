# 실험 문서와 기록 검사

2026-09-21. 검사 대상은 현재 작업 공간의 Markdown 로컬 링크, 결과·후보 JSON 문법,
후보 source hash와 실측 결과의 귀속, 현재 원장의 채택·실패·미실행 상태다.

| 대상 | 확인한 상태와 범위 |
|---|---|
| E37 이전 조합 | [동일 패키지3회](e37-combined-repeats.json) 전부 PASS, 중앙값94,195/45,748/379,741. [별도 추가 입력6개](e37-combined-extra-correctness.json)도30개 비교 PASS. |
| E47 채택 조합 | [반복3회](e47-output-sixteen-columns-repeats.json) 모두15개 PASS, 중앙값94,662/44,588/377,564. [추가 입력6개](e47-output-sixteen-columns-extra-correctness.json)30개 PASS. 동일 source 복구 빌드와 extra host 내 device images도 검증. |
| E44–E50 후속 | E44/E45/E46/E48 성능 이득 없음, E49 정적 회귀·NPU 미검사. E50은15개 PASS/376,184 단일 screen이며 E51 결합 검증 전 미채택. |
| FP4 진단 | Small/fullshape-tagged 검사는 각각25,165,824/117,964,800 BF16 값 bit-exact PASS. 전체 FFN·모든 입력의 보장으로 확장하지 않음. |
| E37 cache 보존 | [70589](e37-cache-preservation-correctness.json)는 E37과 동일한 device image를 쓴 별도 host harness. 기존15개 비교와 cache12개 검사 PASS, 비대상12,570,624개 bit-exact 보존. 진단 cycles는 성능 집계에서 제외. |
| E37 소스·재현성 | [수정 범위 검사](e37-source-contract-audit.json), [복구 빌드](e37-final-restore-audit.json), [빈 target 빌드](e37-clean-build-audit.md)를 구분. 공식 harness와 공개 함수 선언 보존. |
| E27/E29 반복 | E27은 원래 패키지의 단일 screen. E29는 [70428/70482/70493 반복](e29-qkv-input-control-repeats.json) PASS. 추가 입력과 같은 패키지 반복을 혼동하지 않음. |
| E31/E36 | E31 반복·추가 입력·cacheguard를 E37 검사로 대신하지 않음. E36 후단은 공식15개·추가30개 PASS이며 E37 FFN device image와 동일. |
| E35/E38 | E35 native full256은 통과했지만 느림. E38 PASS는 `mlp.direct-scales.rs` 변형이며, 같은 폴더의 weight-only `mlp.rs`에는 적용하지 않음. |
| E39/E40 | E39는15개 PASS/FFN415,457. E40은 FFN 세 입력 모두 non-finite로 기각. 실패 cycles를 유효 개선으로 표시하지 않음. |
| E41/E42 | 각각15개 PASS, FFN426,442/456,849. E37보다 느려 미채택. 각 후보의 정적 예측을 실측과 구분. |

Markdown은 `rg --files -g '*.md'`로 열거하고 inline/reference-style 로컬 링크 목적지의
파일 존재를 검사한다. URL decoding 후 anchor와 line-number suffix는 제거한다.
외부 URL 연결, heading anchor 정확성, HTML 속성, code fence 안 예제 링크는 검사하지 않는다.
Git-ignored artifact/target 내부 문서는 열거에서 제외한다. 일부 링크는 현재 workspace의
ignored artifact에 연결되므로 새 clone에서 해당 artifact가 존재한다는 보증은 하지 않는다.

JSON 범위는 `docs/**/*.json`과 `moa-dev/experiments/candidates/**/*.json`이다.
표준 parser의 문법 검사이며 모든 JSON schema나 수치 관계를 검증한 것은 아니다.
실측 수치는 별도로 collector 결과 및 업로드 파일 hash와 대조했다.

Git whitespace 검사는 raw patch/log를 제외했다. 변경하지 않은 frozen cache-checker에는
파일 끝 빈 줄이 있어 `core.whitespace=-blank-at-eof`로 이 항목만 제외하고 검사했다.
실험 source의 hash를 보존하기 위해 해당 파일을 서식 목적으로 변경하지 않았다.

최종 검사 `2026-09-21T12:26:12.380545+00:00`: Markdown 129개, 로컬 링크 423개,
JSON 394개에서 누락 파일·JSON 문법 오류0건. 외부 링크154개는 검사 범위에서 제외했다.
검사 스크립트와 출력은 `/tmp/moa-local-doc-qa.py`, `/tmp/moa-local-doc-qa-result.json`에 있다.

공식 score/rank는 아직 없다. [공식 상태](official-submission-status-20260921.json)의
평가 기준 maintenance와 별도 제출기 인증 미완료를 유지한다. Arena PASS는 기록된
입력·artifact에 한정하며 전체 모델 또는 모든 가능한 입력의 정확성 증명으로 확장하지 않는다.
