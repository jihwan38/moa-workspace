# Original / E55 continuation 진단 host

2026-09-22. [네 prefix 비교](../model-prefix-suite-e55/README.md)를 최대 8개 continuation checkpoint로 확장하는 별도 host다. 원본 모델의 공개 `sample` 함수가 고른 token을 E55에 그대로 공급하고, 같은 입력 history에서 매 checkpoint의 전체 262,144 logits를 비교한다. E55가 다른 token이나 EOS를 선택했어도 그것을 다음 입력으로 사용하지 않는다. **독립적인 두 답변 생성, 정답 평가, 정확도 합격 판정은 아니다.** Arena runner와 package는 root 소유의 `model-continuation-pair-e55`에 별도로 둔다. 이 후보의 빌드/CPU 검증은 모델이나 NPU를 실행하지 않는다.

Host 구현은 [probe_model_continuation.rs](probe_model_continuation.rs) 하나이며, readiness와 SHA는 [host-source.json](host-source.json)에 고정했다. 나머지 모델 source/dependency는 frozen original 또는 E55의 46개 입력 그대로다. 기존 `probe_model_prefix_suite.rs`는 복사하지 않으므로 각 crate는 정확히 **46개 base + 새 host 1개 = 47개**다.

**두 fullhost build, source47 검증, image10 identity, CPU guard11개 모두 PASS.** 두 target은 처음 비어 있었고 원본51.22초/E55 58.41초에 첫 빌드가 성공했다. Host source는 `e098585658c1533d5248eeff86574d60753e0bea5825cd86a9d1b463e19ce791` (38,207B)이며 준비 후 수정 없이 빌드했다.

| Host | Bytes | SHA256 |
|---|---:|---|
| [Original](../../artifacts/model-continuation-e55/original/probe_model_continuation) | 11,535,888 | `5fccf852472c61043dc9968b8823a17b8add3c02006e3840c5b7d1df03937dc9` |
| [E55](../../artifacts/model-continuation-e55/e55/probe_model_continuation) | 11,534,608 | `418f682a305c2ffb39b59aaf6bb939da236857f4f3ab7522476b41c882a3837b` |

[Evidence](evidence.json), [source 검사](source-verification.json), [실제 registry](registry-summary.json), [CPU 검사](cpu-test-evidence.json)에 최종 연결을 기록했다. 이 준비 단계에서 NPU를 실행하지 않았다.

## 실행 흐름과 검사 범위

```text
args / reference의 manifest·파일집합·모든 checkpoint 검사
→ 공식 tokenizer로 고정 네 prompt encode, token ID와 budget 확인
→ Device::new + load_model (host당 한 번)
→ prompt마다 fresh Workspace / Decode (prompt 사이 cache 공유 없음)
→ prompt prefill 한 번, 마지막 prompt 위치 logits가 step0
→ 선택된 원본 token을 하나씩 추가하며 step1..7 logits 비교
→ 원본 EOS 또는 8번째 선택에서 해당 case 종료
→ 모든 case 완료 후 최종 completion event
```

Prompt는 기존과 같은 `Hi`, `What is 2 + 2?`, `대한민국의 수도는?`, `Complete the Python expression: len([1, 2, 3])`다. `thinking=false`, `add_generation_prompt=true`를 유지한다. 공개 `sampling.rs::sample`을 `temperature=1, top_k=1, top_p=1`로 호출하고 EOI/EOA 두 token을 suppress한다. EOS 3개는 선택 가능하다. Sampler의 `total_cmp`/unstable tie 순서를 임의의 lowest-ID 선택으로 대체하지 않는다. 별도의 raw-top 진단 tie 규칙은 sampler 의미와 구분한다.

`--emit DIR`은 없는 directory를 새로 만들고 각 checkpoint를 `case-I-step-J.json`에 완전히 serialize/flush/sync한 뒤 이벤트를 출력한다. 모든 case가 끝나야 `manifest.json`을 publish한다. 기존 파일을 덮어쓰지 않는다. `--compare DIR`은 정확한 manifest/file set, regular file, finite/원소 수/좌표/선택 token 최대성, prompt IDs와 EOS 종료 규칙을 Device 생성 전에 검사한다. candidate가 reference와 다른 합법적인 greedy token을 고르면 첫 불일치 step과 reference token의 candidate rank를 기록한다. 이후에도 reference history를 재생한다.

동일 prompt 안에서는 Workspace/Decode를 checkpoint 사이에 유지한다. EOS 선택과 마지막 8번째 선택은 다시 feed하지 않는다. 관측된 prompt 합82라면 보수적 token guard는 `82 + 4×8 = 114 ≤128`, 실제 forward 상한은 `82 + 4×7 =110`이다. 최대32 checkpoints/8,388,608 logits이며 EOS에서 일찍 끝나면 줄어든다. 이 workload는 긴 context/page/ring 경계나 multimodal 입력을 검사하지 않는다.

새 `total_seconds`는 reference parse/validation을 포함하고 `reference_parse_seconds`도 따로 기록한다. `load_seconds`는 Device 생성·모델 로딩, `workspace_seconds`는 새 Workspace/Decode, `prefill_seconds`는 prompt 전체 forward(마지막 head 포함)다. checkpoint의 `decode_and_read_seconds`는 step0에서는 logits read, step>0에서는 추가 token forward+head+read다. Sampler/통계/JSON 비용은 해당 값 밖이지만 checkpoint/case/전체 시간에는 포함된다. 순차 단일 실행 시간이므로 공식 성능이나 kernel별 점유율로 해석하지 않는다. 모든 accuracy threshold/verdict는 null이다.

## Source / build / image provenance

[prepare_sources.py](prepare_sources.py)는 기존 prefix suite의 원본/E55 source47개와 compiled47개 manifest 및 실제 frozen 파일을 각각 대조한다. old probe를 제외한46개만 새 artifact crate에 복사하고 readiness SHA가 일치한 continuation host를 추가한다. 추가/누락 파일이 있으면 중단한다. [verify_source_inputs.py](verify_source_inputs.py)는 준비된47개, 실제 Docker 컴파일47개, 빌드 후 실제 Docker47개와 frozen host SHA를 다시 대조한다.

```sh
sh moa-dev/experiments/candidates/model-continuation-e55/build.sh
python3 moa-dev/experiments/candidates/model-continuation-e55/test_guards.py
```

[build.py](build.py)는 두 variant를 `/tmp/moa-model-continuation-e55/{original,e55}`에서 별도로 빌드한다. `CARGO_HOME=/home/moa/.cargo`, 각 scratch의 `target`, 각 scratch의 `target/furiosa-opt/kernel`을 명시하고 `cargo furiosa-opt build --release --locked --bin probe_model_continuation`을 사용한다. live crate와 `/workspace/target`은 사용하지 않는다. source가 없거나 readiness SHA와 다르면 placeholder 빌드 없이 중단한다.

각 [original source manifest](original-source-manifest.json) / [E55 source manifest](e55-source-manifest.json)는 입력47개 전체 hash와 크기를 포함한다. 대응 `*-compiled-source-manifest.json`에는 실제 Docker 입력과 첫 target 상태, `*-build-result.json`에는 명령·exit·시간·로그 SHA를 기록한다. [verify_registry.py](verify_registry.py)는 SHA로 고정한 기존 ELF/bincode parser로 실제 두 host registry를 파싱한다. 각 variant의 정확한10개 text entry와 empty key, standalone `.bin`, 기존 prefix-suite host의10개 image와 byte identity를 요구한다. 원본/E55 사이의 변경 image set은 기존대로 Stage1 세 개와 full-O 총4개여야 한다.

## CPU guard 검증

[test_guards.py](test_guards.py)는 host의 `#[tokio::main]` 이전 helper와 이후 test module을 원문에서 추출한다. SDK import만 제거하고 frozen `sampling.rs`를 그대로 넣으며, context/vocabulary 상수 E/W만 실제 frozen axes와 대조한 stub으로 제공한다. 기존 sampler를 재구현하지 않는다. 새 E55 build의 serde/serde_json/direct rand0.9 rlib을 링크해 `rustc --test`로 실행한다. NPU main은 포함하지 않는다.

소스에 포함된11개 test는 suppress/EOS, 공개 sampler tie 및 signed zero, EOS와8번째 token feed count,114 guard/110forward/overflow, manifest counter와 policy, checkpoint 좌표·최대성·finite·길이, tokenization 불일치, candidate EOS와 reference continuation의 분리, 전체 배열 마지막 원소·signed-zero 통계, unknown JSON field 거절을 확인한다. CPU 함수 검증은 실제 cache allocation/Device/model 실행 검증을 대신하지 않는다. 결과와 링크된 library hash는 `cpu-test-evidence.json` 및 artifact의 `guard-test-build.json`, `guard-tests.log`에 기록한다.

최종 source/build/image/test 연결과 executable SHA는 `evidence.json`에 동결한다. 원본은 비교 기준이며 독립적인 정확도 oracle로 간주하지 않는다. 이 준비 기록 자체로 NPU 결과·속도·순위를 주장하지 않는다.
