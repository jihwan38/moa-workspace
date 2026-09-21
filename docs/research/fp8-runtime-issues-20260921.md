# opt 0.8.1 FP8 / pair 실행 실패와 공개 이슈 대조

확인 시각: 2026-09-21 10:39 UTC. 조사 대상은 Furiosa 공식 포럼, `furiosa-ai/furiosa-opt` 공개 issues / releases / v0.8.1 소스다. 외부 게시·문의, NPU 실행, SDK 변경은 하지 않았다.

현재 FFN native FP8의 launch timeout이나 QKV pair의 V 일부 좌표 오류와 **정확히 일치하는 공개 결함 보고는 확인하지 못했다.** 아래는 원인이 확정된 이전 사례와 현재 대조에 쓸 수 있는 제약이다. 이전 버전의 수정 진행 답변을 0.8.1의 미해결 결함으로 간주하지 않는다.

| 출처와 날짜 | 공개 자료에서 확인한 내용 | 현재 실험에 적용할 범위 |
| --- | --- | --- |
| [드라이버 호환성, yan 답변 2026-09-17](https://forums.furiosa.ai/t/furioa-opt-support-matrix-driver-version/464/4) | opt 0.8.1 실물 카드 실행에는 드라이버 2026.4.0이 필요하며, 당시 드라이버가 아직 출시되지 않아 이전 opt 사용을 안내했다. 질문자는 다음 날 opt 0.7.0 실행 성공을 보고했다. | Arena의 실제 드라이버/펌웨어 버전은 이 조사에서 확인하지 못했다. 이 글만으로 Arena 환경 불일치를 주장할 수 없다. 로컬 SDK를 무작정 내릴 근거도 아니다. |
| [TDMA/Switch broadcast correctness, jeongmin.park 답변 2026-08-03](https://forums.furiosa.ai/t/furiosa-opt-0-4-tdma-layout-broadcast-cycle-mapping-correctness/435/6) | opt 0.4 보고의 `m![A / 2] → m![A]` 전송에서 빠진 `A % 2`가 broadcast로 처리되지 않는 문제를 확인했다. 먼저 명시적인 별도 2축을 붙여 복제한 뒤 reshape하는 우회를 제시하고 수정을 진행 중이라고 답했다. | 컴파일 성공 뒤 특정 좌표가 틀릴 수 있었던 **다른 mapping의 과거 사례**다. 현재 0.8.1 잔존 여부와 V 오류와의 동일성은 미확인이다. 입력의 물리 복제와 축 reinterpret를 구별하는 대조에 유용하다. |
| [v0.8.1 `VectorTensorPair` 소스 632–647행](https://github.com/furiosa-ai/furiosa-opt/blob/v0.8.1/furiosa-opt-std/src/engine/vector/tensor/vector_tensor_pair.rs#L632), [879–882행](https://github.com/furiosa-ai/furiosa-opt/blob/v0.8.1/furiosa-opt-std/src/engine/vector/tensor/vector_tensor_pair.rs#L879) | pair에서 bypass `vector_narrow_trim` / `vector_widen_pad`를 허용하지 않는 이유가 적혀 있다. padding된 execution ID와 최종 group filter의 조합이 선언한 출력과 다른 shape를 만들 수 있어 buffered `vector_narrow_split` / `vector_widen_concat`를 사용하도록 한다. | 가장 직접적인 pair 자료지만 **이미 API가 막는 패턴**이다. 우리의 QKV pair와 native FFN은 buffered split/concat를 사용한다. 이 주석을 현재 실패의 원인으로 인용하면 안 된다. |
| [VRF scheduling, chaehyun.jeong 답변 2026-08-14](https://forums.furiosa.ai/t/furiosa-opt-v0-5-1-8-kib-slice-vrf-operand-vector-pipeline-scheduling/451/4) | opt 0.5.1의 8 KiB operand에 `vector_stash()`가 보수적으로 예약하는 1,024 B가 더해져 할당이 실패한 사례다. stash 제거 후 compile 성공을 확인했다. VRF는 slice당 총 8 KiB / 2 banks다. | 우리 native FFN 8-row variant의 compile-time VRF 실패를 검토할 때 참고할 자료다. compile에 성공한 4-row variant의 **실행 timeout을 설명하지는 않는다.** |
| [transpose emulation, jeongmin.park 답변 2026-09-15](https://forums.furiosa.ai/t/furiosa-opt-v0-6-0-v0-3-0-transpose-gelu-v0-6-0-emulation/458/2) | opt 0.6.0 transpose emulation이 sequencing 표현 한계로 실패할 수 있으나 실제 NPU 실행에는 영향을 주지 않는다고 답했다. stash에는 단일 소비와 쓰기/읽기 ALU way 일치 제약도 설명했다. | emulation 오류와 물리 NPU 오류를 분리해야 한다. 현재 Arena timeout과 같은 이슈라는 근거가 아니다. |
| [commit_view 축 제약, jeongmin.park 답변 2026-08-19](https://forums.furiosa.ai/t/furiosa-opt-v0-6-0-slice-dm-commit-view-streamunmatchedsegment/453/2) | commit stream의 packet과 destination element의 같은 크기 innermost 부분이 정확히 일치해야 한다. 원소 수만 같은 축 재배열은 허용되지 않으며 transpose로 맞추도록 답했다. | scratch mapping을 구현할 때 실제 packet 정렬을 확인하는 근거다. 해당 보고는 compile-time rejection이며 runtime DM corruption 보고가 아니다. |

GitHub 공개 issues 전체를 `state=all&per_page=100`으로 조회한 결과 1건이었다. [issue #1](https://github.com/furiosa-ai/furiosa-opt/issues/1)은 v0.4.0 CLI가 0.3.0으로 버전을 출력하던 문제이며 닫혀 있다. [v0.8.1 release](https://github.com/furiosa-ai/furiosa-opt/releases/tag/v0.8.1)는 2026-09-11 게시됐고 runtime package 정렬, production-signed bootloader, mapping diagnostic test 수정만 명시한다. 이 짧은 release note로 FP8 pair 결함의 존재·부재나 수정 여부를 추정하지 않는다.

## 로컬 실패와 구분

- [E22 원본 로그](../../moa-dev/logs/e22-native-ffn-upgate-safe-qkv-job-70321.log)의 QKV 9개와 O 3개 비교는 통과했다. FFN은 첫 실행에서 launch time limit으로 종료했고 출력 비교나 cycles를 얻지 못했다. 이는 수치 오차 초과를 측정한 결과가 아니다.
- [native FFN postreduce-div 후보](../../moa-dev/experiments/candidates/ffn-native-fp8-upgate/mlp.postreduce-div.rs)의 pair 부분은 `contract_lane(Sequential) → unzip → narrow_split → Mul0 → zip → Mul1 → widen_concat`다. 이후 별도 standalone block reduction에서 쓰는 `widen_pad`를 pair에 금지된 bypass 패턴과 혼동하지 않는다.
- [QKV pair 후보](../../moa-dev/experiments/candidates/qkv-fp8-split-two-lane/projection.rs)도 pair 부분에서 buffered split/concat를 사용한다. V 좌표별 진단 결과는 담당 agent의 자체 실험 근거이며 위 외부 글들이 그 원인을 증명하지 않는다.

다음 원인 분리의 정보 가치는 실제 실행 환경 버전 기록, 같은 입력을 쓴 producer-only 출력 확인, 명시적 broadcast 축과 현재 축의 비교 순이다. 이는 후속 실험 제안이며 본 조사에서 실행하지 않았다. 현재 통과한 구현은 유지한다.

검색 범위와 한계: 공식 포럼 최신 목록 및 관련 여섯 thread 전체, 공식 GitHub issues 전체 첫 페이지(총 1건), 최신 release 세 개, v0.8.1 pair 소스를 읽었다. 웹 검색에는 `FP8`, `Sequential`, `timeout`, `corruption`, `furiosa-opt`, `correctness`를 사용했다. 포럼 내부 search API 일부 요청은 HTTP 429로 거절되어 재요청하지 않았다. 검색 결과가 없다는 것은 SDK 전체에 결함이 없다는 뜻이 아니다. 조회 URL·응답 hash·검색 제약은 [evidence JSON](fp8-runtime-issues-20260921-evidence.json)에 기록했다.
