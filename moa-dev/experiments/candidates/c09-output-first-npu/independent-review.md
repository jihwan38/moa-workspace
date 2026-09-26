# C09 첫 NPU 패키지·실제 업로드·원시 로그 재검토 — 2026-09-27

검사 결과 **패키지 및 job105820 검증 PASS**, 차단 오류 없음. Packer 작성자가 수행한 별도 검사이며 다른 사람의 리뷰를 뜻하지 않는다. `package.py`, `screen.py`의 판정 함수를 import하거나 실행하지 않고 실제 파일의 해시, 별도로 읽은 FOPT ABI, host 전체 byte diff, 실제 다운로드 tar와 raw log를 대조했다. Compile/host build/Arena 호출은 하지 않았다. 검사 수치·해시는 [independent-review.json](independent-review.json)에 남겼다.

## 실제 source와 host

E122 recipe → source-build-audit → measured build/package를 직접 읽었다. E122의 45개 git source 입력과 별도 tracked Cargo.lock 한 개를 manifest 및 실제 control package와 비교했고, QKV/O/FFN exact image를 control host에서 각각 한 번 확인했다. E122 source commit은 `c3553ab87046fb4d59c10149a82517f8440bba27`이다.

Candidate의 46개 실제 crate 입력도 build/package manifest와 같다. E122와 다른 파일은 동결 원 C09의 `src/device/sliding/projection.rs` 하나다. 새 exact O image는 원 C09의 80,640B `fe90ed758c27fe40a11e78ff5132e7b98152fc8cd79208d49c22053672859495`와 byte-identical이다. Standard host 및 unlabelled diagnostic host에 각 image가 정확히 한 번 포함되고, 두 host의 세 device image가 모두 일치한다.

| Kernel | candidate image SHA256 | 계약 |
|---|---|---|
| QKV | `5bdab7eafec5ac0aca352101deee334180b550c87db23e33c70d008892edba9e` | E122 동일 |
| O | `fe90ed758c27fe40a11e78ff5132e7b98152fc8cd79208d49c22053672859495` | 원 C09 동일 |
| FFN | `16914f7736e5eb919c053cca3a82f02784380a5d0f983b9d77c0d98db7f070d2` | E122 동일 |

실제 descriptor gate JSON의 `passed:true`, package가 고정한 gate SHA 및 모든 `inputdigests`의 현재 bytes/SHA도 다시 대조했다. Descriptor 알고리즘 자체의 독립 검토를 재수행했다는 뜻은 아니다. 그 범위는 [descriptor analysis](descriptor-analysis-c09-first-npu-v1.json)와 담당 검토 기록을 따른다.

## Metadata만 바꾼 diagnostic

FOPT ABI의 original/probe header와 task 위치를 별도 parser로 읽었다. Candidate probe에서 바뀐454 bytes는 profile span name의 고정길이 영역에만 있고, control은315 bytes다. ABI의 marker ID, task bytes, constant/입출력/stack/slot/depth, image 길이와 host의 나머지 bytes가 같다. C09의 두 task는 원본과 같은35,616B 및35,040B이고 SHA도 기록했다.

원본 image에서 직접 센 native engine marker는 candidate26개, control22개다. Native kind 전체 순서, marker-map의 schedule node 내용, 실제 frozen schedule을 대조했다. 22개라는 기존 가정을 C09에 적용하지 않았다. Diagnostic source/build logs/metadata/marker-map의 frozen evidence 파일들도 ignored artifact와 같다.

Diagnostic host source는 기존 generic transform과 동일한 해시다. 변환 코드를 읽어 `DIAG_KERNEL` 선택과 span name 수집·출력만 추가하며 `Synth::new`, fixture checksum,3개 run 입력, numerical tolerance, compare/exit 경로를 유지함을 확인했다. Whole와 diagnostic의 fixture는 byte-identical이다. 이전 kernel 실행 여부, HBM 주소와 cache/queue 이력까지 동일하다는 주장은 하지 않는다.

Control diagnostic의 원본은 E104 generic host지만 O image만 E122와 동일함을 검증하여 O-only로 실행한다. 그 안의 E104 FFN은 실행하지 않는다. Complete before/after control은 별도의 실제 E122 full-three-kernel host다.

## 실제 업로드와 실행

실제 tar `job-105820.tar`는15,039,488B, SHA `85d1afb43402c4d33448b94927a9939f2e53ecb04b53b75b7cb226ee5158fbcd`다. Regular member8개에 이름 중복이 없고 `job.yaml` 제외 모든 bytes/SHA가 준비 package와 같다. Tar 안의 standard hosts, diagnostic probes, fixture 및 official runner도 감사한 로컬 파일과 직접 비교했다. 원본 log는53,144B, SHA `7981fa803fb8f114b9f18220b5301c7271ef24e3060f799185cc8f3218dae06c`로 결과 기록과 일치한다.

Raw log에서 아래7개 stage 순서, 각 exit0, SCREEN_COMPLETE1회 및57개의 중복 없는 정확한 output label/100% PASS를 독립 검사했다. Submission/result job ID는105820으로 일치하고 저장된 status는 SUCCEEDED/exit0이다.

| Stage | PASS 비교 | O cycles 3개 | O 중앙값 |
|---|---:|---|---:|
| control_before | 15 | 52,702 / 50,354 / 51,237 | 51,237 |
| control_info | 3 | 53,403 / 49,940 / 50,019 | 50,019 |
| control_trace | 3 | 53,414 / 51,151 / 50,513 | 51,151 |
| candidate_info | 3 | 51,620 / 49,819 / 48,371 | 49,819 |
| candidate_trace | 3 | 53,147 / 49,169 / 49,780 | 49,780 |
| candidate_whole | 15 | 50,813 / 48,496 / 49,923 | 49,923 |
| control_after | 15 | 52,336 / 49,853 / 51,165 | 51,165 |

Diagnostic raw span은 info 각각6개, control trace156개, candidate trace180개다. 모든3run×2cluster에서 Task 범위와 marker 중복 여부를 확인했고, trace에서 각26/22개 native operation의 존재·시작 순서 및 reported cycles와 max-cluster Task duration 일치를 검사했다. 세부 weight/dot/gather causal 해석은 별도 analyzer 범위다.

표준 host complete O만 비교하면 before51,237 / after51,165의 중앙값51,201 대비 candidate49,923으로 **−2.496045%**다. Diagnostic info/trace를 이 계산에 섞지 않았다. Standard host도 `PROFILE=info`이므로 unprofiled 결과라고 부르지 않는다. 한 allocation과 원본3입력에서 얻은 제한적 결과이며 독립 반복이나 확장 정확도 검증은 아니다. 기본5% 예산 기준 아래의 소폭 positive option으로 보존하고 추가 실험 없이 종료한다는 결정과 부합한다. 통계적 효과 확정, 채택 또는 공식 제출 근거로 승격하지 않는다.

## 비차단 수집 한계

완료된 stage 안의 `DIAG_SPAN` JSON 자체가 깨지면, `screen.py`의 예외 처리 list comprehension이 다시 예외를 내서 뒤 stage의 구조화 처리를 멈출 수 있다. 원본 log와 이미 처리한 stage가 남고 `valid=false`라 false PASS 경로는 아니다. 이번 job에는 해당 현상이 없었으며 모든 stage와 span이 정상 파싱되었다. 원본 freeze 이후 이 한계 때문에 코드를 바꾸거나 실행을 반복하지 않았다.

검사에 사용한 별도 parser/script는 [verify_package_independently.py](verify_package_independently.py)에 byte 그대로 보존했다. 로컬 artifact cache에 의존하는 기록용 verifier다.
