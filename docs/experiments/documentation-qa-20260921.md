# 실험 문서와 기록 검사

2026-09-21. 검사 대상은 현재 작업 공간의 Markdown 로컬 링크, 결과·후보 JSON 문법,
후보 source hash와 실측 결과의 귀속, 현재 원장의 채택·실패·미실행 상태다.

| 대상 | 확인한 상태와 범위 |
|---|---|
| E37 이전 조합 | [동일 패키지3회](e37-combined-repeats.json) 전부 PASS, 중앙값94,195/45,748/379,741. [별도 추가 입력6개](e37-combined-extra-correctness.json)도30개 비교 PASS. |
| E47 이전 조합 | [반복3회](e47-output-sixteen-columns-repeats.json) 모두15개 PASS, 중앙값94,662/44,588/377,564. [추가 입력6개](e47-output-sixteen-columns-extra-correctness.json)30개 PASS. 동일 source 복구 빌드와 extra host 내 device images도 검증. |
| E53 이전 채택 | [동일 패키지3회](e53-bf16-output-repeats.json) 각각15개 PASS, 중앙값95,621/50,882/377,880. 추가30개 및 O 경계8개 PASS. 이후 별도 QKV 상쇄 입력에서8,072개 실패하여 E54로 교체. |
| E54 이전 정확도 기준 | 공식15개·추가30개 PASS. 원본/E53/E54의 동일 QKV 경계2개 대조에서 E53만 실패, E54는 비대상 cache8,380,416개도 보존. 정리한 source의 전체 host byte-identical, 별도 extra host의3개 image 일치. [반복3회](e54-bf16-qkv-repeats.json) 각각15개 PASS, 중앙값108,700/50,970/382,268. |
| E52 및 QKV BF16 대안 | E52 공식15개 PASS이나 속도 이득 없어 미채택. BF16 QKV 대안은 exact compile만 확인했고 장치 미검증이며 E53 O를 덮어쓰지 않도록 통합 주의 기록. |
| E55 현재 채택 | [동일 패키지3회](e55-fused-qkv-repeats.json) 각각15개 PASS, Q/O/FFN 중앙값102,878/51,249/380,339. [추가30개](e55-fused-qkv-extra-correctness.json)와 [QKV 경계2개](e55-qkv-boundary-probe.json) PASS. QKV는 E54보다5.36% 낮다. 같은source의 isolated/live full host byte-identical, extra/probe image와 업로드 package 검증. |
| E55 공식 CLI 패키징 | [로컬 검사](e55-official-submit-preflight.json)의 세 POST body를 독립 decode하여 동결 E55 소스23개/160,829B와 일치 확인. 세 body byte-identical, 잘못된 cwd의 음성 검사에서는 HTTP 요청0개. 합성 토큰·loopback으로 검사했고 공식 서버의 인증·계약·점수는 미검증. |
| E55 전체 모델 host 빌드 | [격리 검사](e55-model-host-build.json)에서 `gemma4` fresh build와 `server` incremental build 성공. 두 ELF의14-entry registry를 끝까지 parse하여 compiler image 및 실측 Stage1의3개 image와 정확히 대조. 모델 자산 적재·NPU 실행·E2E 정확도/성능 검증은 아님. |
| Arena 실제 모델 자산 | [71164](e55-arena-model-assets.json)에서 모델·tokenizer·metadata/template의 regular/readable 상태 확인. Python 해시 시도[71191](e55-arena-model-fingerprint-python.json)는 원격 인터프리터 부재로 즉시 실패했고 자산 내용을 읽지 않음. Rust [71224](e55-arena-model-identity.json)는 전체 weight/metadata SHA·1,149 required header/dtype 검사 완료. partial은 optional JSON template 부재만 의미함. |
| 실제 모델 첫 실행 | CLI [71173](e55-arena-model-smoke.json)은70초 timeout. 별도 진단 [71198](e55-arena-model-progress.json)은 실제14-token prefix의262,144개 finite logits까지 정상 완료. 첫 timeout 원인은 미확정이며 한 번의 시간값을 cold-load/공식 성능으로 해석하지 않음. |
| 원본/E55 모델 비교 | [71207](e55-model-fixed-prefix-pair.json)의262,144개 전원소 비교에서 max abs1.5·mean abs0.194487·RMSE0.251039·top1 동일. 양쪽 host/업로드/47개 source/10개 image 귀속을 확인했으나 정확도 수용 판정은 null. 다른 입력·long context·멀티모달은 범위 밖. |
| 4-prefix 확장 | [71239](e55-model-prefix-suite.json)의82token·1,048,576개 logits가 모두 finite, raw top1 네 개 동일, max abs0.584–1.5. 같은 job의 전체 weight SHA·전후 stat/header·metadata SHA 일치 확인. 수집기는 실제양성1개와 변형음성20개 검사 통과. 정답률·공식성능·모든입력 정확도 PASS 아님. |
| 실제 모델 단일-kernel 대조 | [71275/71281/71282/71290](e55-model-single-kernel-controls.md)4건 모두 정상 완료. 대상1image=원본/다른9image=E55를실제업로드에서도확인. Full-O의1,048,576개logits는E55와bit-identical. 나머지3건은71239와동일referenceSHA, 원본과의오차가일관되게줄지않음. 수집기실제4양성+8변형음성검사통과;이전수집기도1양성+20음성유지. |
| 생성 토큰 진단 설계 | [후속설계](../research/model-continuation-diagnostic-design.md)는실제sampler·EOS·Workspace경로를검토. 원본토큰을고정해최대32checkpoint/110forward를비교하는설계이며아직구현·빌드·NPU없음. 추가logitshead/I/O때문에70초완료보장없음. |
| K/V 결합 후보 | [단일 stream 후보](../../moa-dev/experiments/candidates/qkv-bf16-interleaved-kv-e55/README.md)는 최초 exact compile의 declared/native 축 불일치로 동결. 소스 범위·의도한4096좌표 검사는 PASS이나 실제 lowering/정확성/성능으로 확대하지 않음. |
| SDK 후속 감사 | [mixed DPE](../research/mixed-dtype-dpe-audit.md)와 [lookup table 공유](../research/fp8-lookup-table-sharing-audit.md)를 설치 SDK0.8.1의 실제 caller·dtype bound·typestate 및 기존 graph/schedule과 대조. 공개 API의 제약이며 새 compile/NPU 실행 또는 하드웨어 전체 불가능 증명으로 확대하지 않음. |
| E54 후속 정적 후보 | BF16 QKV dot·scale 결합 exact compile 세 번째 성공, 전체52,282→51,293. 이후 별도 E55 NPU 검증으로 채택. 공통 TRF는51,293→52,562 회귀로 제외; broadcast→TRF는 첫 compile의 bitmap 검증 실패, 유효 성능 수치 없음. |
| E53 후속 정적 후보 | BF16 O8, FFN sharded pre-RMS 두 경로, F32 channel-scale fused O의 exact compile 성공. 네 변형 모두 전체 정적 일정이 회귀해 NPU 미실행·미채택. 실제 성능 또는 수치 정확성 통과로 표시하지 않음. |
| E44–E50 후속 | E44/E45/E46/E48 성능 이득 없음, E49 정적 회귀·NPU 미검사. E50은15개 PASS/376,184 단일 screen. E51 반복·추가 입력 모두 PASS지만 FFN381,286으로 미채택. |
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

최종 검사 `2026-09-21T16:05:52.393202+00:00`: Markdown 179개, 로컬 링크 1007개,
JSON 577개에서 누락 파일·JSON 문법 오류0건. 외부 링크165개는 검사 범위에서 제외했다.
검사 스크립트와 출력은 `/tmp/moa-local-doc-qa.py`, `/tmp/moa-local-doc-qa-result.json`에 있다.

공식 score/rank는 아직 없다. [공식 상태](official-submission-status-20260921.json)의
평가 기준 maintenance와 별도 제출기 인증 미완료를 유지한다. Arena PASS는 기록된
입력·artifact에 한정하며 전체 모델 또는 모든 가능한 입력의 정확성 증명으로 확장하지 않는다.
