# Python prompt 40-checkpoint original / E55 host

2026-09-22. [기존 네 prompt × 최대8 checkpoint 진단](../model-continuation-e55/README.md)을 Python prompt 하나와 최대40 checkpoint로 바꾼 별도 host다. 원본 모델의 공개 `sample`이 고른 token을 E55에 공급하고 같은 history에서 전체262,144 logits를 비교한다. E55가 다른 token이나 EOS를 선택해도 이후 입력은 원본 선택을 따른다. 독립 답변 생성이나 정답 평가가 아니며 accuracy threshold/verdict는 null이다. 새 원격 package와 Arena 실행은 root 소유의 `model-continuation-python40-pair-e55`에서 별도로 관리한다.

**두 isolated fullhost build, 각47 source 입력, 각각10 kernel 이미지 identity, 실제 helper CPU test11개 모두 PASS.** 초기 target은 비어 있었고 첫 빌드에 원본46.03초/E55 57.70초가 걸렸다. 준비 단계에서 모델/NPU를 실행하지 않았다.

| Host | Bytes | SHA256 |
|---|---:|---|
| [Original](../../artifacts/model-continuation-python40-e55/original/probe_model_continuation) | 11,533,872 | `6adf3fc2ba27a2238ee444fb29667756d42a10c724c69c9bc113afe007de0c19` |
| [E55](../../artifacts/model-continuation-python40-e55/e55/probe_model_continuation) | 11,532,592 | `6e9d625971b47fa65859851e2753b2d3fdf3f97eba6424141630028826a29731` |

Host source [probe_model_continuation.rs](probe_model_continuation.rs)는38,682B, SHA256 `d23555ec84eb1b8c3045ff605b6d25862ad734ee605ea1ecd7278f774af7ca4c`다. [host-source.json](host-source.json)의 author readiness 이후 수정 없이 빌드했다. [source.patch](source.patch)와 [parent-host-delta.json](parent-host-delta.json)은 이전 host 대비 schema/상수, 두 동적 오류 문구, CPU test 변경을 기록한다. `#[tokio::main]`부터 test module 전까지 runtime main은 이전 frozen host와 byte-identical이다. 이는 같은 main이 사용하는 prompt/상수가 바뀌지 않았다는 뜻은 아니다.

## 실행 흐름과 범위

Prompt는 `Complete the Python expression: len([1, 2, 3])` 하나다. Schema는 `moa-model-continuation-python40-v1`, `MAX_NEW_TOKENS=40`, `MAX_TOTAL_TOKENS=128`이다. 기존 tokenizer의 `thinking=false`, `add_generation_prompt=true`, 실제 공개 `sample`의 `temperature=1, top_k=1, top_p=1`, EOI/EOA suppression과 EOS3개를 유지한다. 공개 sampler의 tie 순서를 임의의 lowest-ID 선택으로 교체하지 않는다.

```text
args / 모든 reference 파일·manifest·checkpoint 검사
→ 공식 tokenizer의 prompt IDs / budget 검사
→ Device 생성과 모델 로딩 한 번
→ fresh Workspace / Decode, prompt prefill 한 번
→ 마지막 prompt 위치 logits가 step0
→ 이전 원본 selected token을 feed하며 step1..39 비교
→ 원본 EOS 또는40번째 선택에서 종료
```

원본이 EOS를 선택하거나40번째 선택에 도달하면 그 token은 다시 feed하지 않는다. 이전 prefix 진단에서 관측한28-token prompt를 그대로 얻는 경우 보수적 guard는 `28+40=68≤128`, 실제 forward 상한은 `28+39=67`이다. token 수는 실행 시 실제 tokenizer 결과로 재검사한다. 최대40 checkpoints /10,485,760 logits이며 원본 EOS가 나오면 줄어든다. 긴 context/page/ring 경계와 multimodal 입력은 검사하지 않는다.

`--emit DIR`은 새 reference directory에 checkpoint마다 serialize/flush/sync하고 모든 case가 완료된 뒤 manifest를 publish한다. `--compare DIR`은 정확한 파일집합과 regular-file 여부, schema/정책/counter, 모든 logits의 길이·finite·좌표·선택 최대성, prompt IDs와 종료 규칙을 Device 생성 전에 검사한다. 기존8-checkpoint 완료 기록은 이40-checkpoint schema/완료 규칙을 만족하지 않는다. candidate가 reference와 다른 합법적 token을 고르면 최초 불일치와 reference token의 candidate rank를 기록하지만 계속 원본 history를 따른다.

`total_seconds`에는 reference parse/validation이 포함된다. `load_seconds`는 Device+model load, `workspace_seconds`는 Workspace/Decode 생성, `prefill_seconds`는 prompt forward와 마지막 head다. checkpoint `decode_and_read_seconds`는 step0에서는 read, 이후에는 추가 token forward+head+read다. sampler/통계/JSON은 해당 내부 timer 밖이고 checkpoint/case/전체 시간 안이다. 67-forward 상한만으로 원격 시간 제한 충족을 보장하지 않으며, 이 단일 diagnostic 시간은 공식 성능이나 kernel별 점유율이 아니다.

## 재현과 provenance

```sh
sh moa-dev/experiments/candidates/model-continuation-python40-e55/build.sh
python3 moa-dev/experiments/candidates/model-continuation-python40-e55/test_guards.py
```

[prepare_sources.py](prepare_sources.py)는 frozen `model-prefix-suite-e55`의 실제 original/E5547개 파일과 prepared/compiled manifests를 각각 대조한다. 이전 `probe_model_prefix_suite.rs`만 제외한46개 base 파일에 새 host 하나를 넣어 정확히47개 입력을 준비한다. 추가·누락·hash 불일치를 거절한다. [build.py](build.py)는 `/tmp/moa-model-continuation-python40-e55/{original,e55}`에서 각자 `CARGO_TARGET_DIR`와 `FURIOSA_OPT_OUT_DIR`를 사용한다. `CARGO_HOME=/home/moa/.cargo`, 명령은 `cargo furiosa-opt build --release --locked --bin probe_model_continuation`이다. live crate와 공유 `/workspace/target`을 사용하지 않는다.

[source-verification.json](source-verification.json)은 준비된47개와 실제 Docker 컴파일 전후47개를 확인한다. `*-source-manifest.json`, `*-compiled-source-manifest.json`, `*-build-result.json`에 모든 입력 SHA/bytes, 실제 target 상태, 명령/exit/시간/로그 SHA가 있다. [verify_registry.py](verify_registry.py)는 SHA로 고정한 ELF/bincode parser로 실제 두 host registry를 읽는다. 각 variant의 정확한10개 text entries, empty generic key, standalone `.bin`, 대응 frozen prefix host의10개 이미지와 byte identity를 요구한다. 원본/E55 간 변경 이미지 집합은 기존 Stage1 세 개와 full-O 네 개 그대로다. 결과는 [registry-summary.json](registry-summary.json)에 있다.

[test_guards.py](test_guards.py)는 실제 host helper와 test module을 추출하고 SDK import만 제거한다. frozen `sampling.rs` 본문을 그대로 사용하고 E/W 상수 stub은 실제 frozen axes와 대조한다. 새 E55 build의 serde/serde_json/direct rand0.9 rlib을 링크한 `rustc --test`로 실행한다. 원본 source의 `#[test]` 개수를 세어 실제 결과와 대조하고 ignored/filtered가0인지 확인하므로 이전 test count를 성공으로 가정하지 않는다. 11개 test는 sampler suppression/EOS/tie/signed zero,40번째/EOS의 no-feed,68guard/67forward/overflow, manifest policy와 잘못된8-step 완료, checkpoint/전체배열/finite/tokenization/unknown-field 검사를 포함한다.

[CPU 결과](cpu-test-evidence.json)는11/11 PASS이고 library/test ELF hash는 artifact의 `guard-test-build.json`에 있다. 실제 Device/cache/model을 실행하지 않은 CPU helper 검사이므로 NPU 실행이나 모델 정확도를 증명하지 않는다. [evidence.json](evidence.json)에 source/build/image/test/재현 script를 연결했다. 원본 모델도 비교 기준이며 독립적인 정답 oracle로 보지 않는다.
