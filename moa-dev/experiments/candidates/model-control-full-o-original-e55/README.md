# E55의 full-O만 원본으로 복원하는 모델 대조

Subsequent root-managed [Arena job71281](../../../../docs/experiments/e55-model-control-full-o.json) completed.
See the [combined model results](../../../../docs/experiments/e55-model-single-kernel-controls.md).
The build evidence below remains a preparation-time snapshot.

2026-09-22. [수치 차이 감사](../../../../docs/research/e55-model-numerical-differences.md)와 [shared caller 감사](../../../../docs/experiments/e47-shared-caller-audit.md)에 따른 단일 kernel 대조다. E55에서 `shared::residual::add`의 정적 8-tile 매크로만 원본 loop로 복원한다. 각 tile의 주소, F32 add, BF16 cast, write는 원본과 같다. 이 작업은 새 최적화나 모델 정확도 합격 판정이 아니다.

**Fresh build와 image 대조 PASS.** Full-O 1개는 원본, 나머지 9개는 E55 image와 정확히 일치했다. 첫 target은 비어 있었고 빌드는 127.22초에 성공했다. 실제 host는 [`artifacts/model-control-full-o-original-e55/probe_model_prefix_suite`](../../artifacts/model-control-full-o-original-e55/probe_model_prefix_suite), **11,406,512B / SHA256 `690695e1d097c6a7ba86f7e3c1a3f3c29c437e9fa1f1a75df0afde538d00a76d`**다. NPU는 이 준비 작업에서 실행하지 않았다.

`src/host/load.rs`는 48 layer 중 `layer % 6 == 5`인 8개를 full attention으로 로드한다. 매 token의 `Decode::full`은 full attention 결과와 softmax sum을 `ops::full_attention_output`에 넘기고, 그 kernel은 divide → 원본 full projection → generic RMS → residual add → HBM write를 수행한다. E55 source에서 generic `shared::residual::add(`의 실제 호출은 `src/ops.rs:207` 이 한 곳이다. Sliding O는 자기 sharded residual 경로, FFN은 `add_and_scale_sharded`를 사용한다. 따라서 source 변경 범위로는 full-O만 영향을 받을 것으로 예상했고, 최종 판정은 compiled image 비교로 한다.

[prepare_sources.py](prepare_sources.py)는 [기존 original/E55 prefix suite](../model-prefix-suite-e55/README.md)의 frozen 47개 입력을 양쪽 source/compiled manifest와 실제 파일로 먼저 대조한다. E55 47개를 복사한 뒤 `src/device/shared/residual.rs::add` 함수만 원본 source의 같은 함수로 치환한다. 함수 뒤의 E55 helper suffix는 byte-identical로 보존한다. [source.patch](source.patch)와 [residual.rs](residual.rs)가 유일한 source delta다. 원본 `ops` body, 전체 host/loader/runtime, dependency, public device signature는 바뀌지 않는다.

- 변경 residual SHA: `8be965295957defb41fb6c5c65d68b655726ba6053da72d51b93ce8e9d7b7860` (8,850B)
- 재사용 host SHA: `07cfaf4646da18a78e03bb386632ac455ccf545d98b858385bded281c3104da4` (13,434B)
- Lock SHA: `aaff8dd41f9c643dd8c9faa358b9bc780e88a42c09172c1a5d7a0726d07efe09`
- 준비 manifest: [source-manifest.json](source-manifest.json)
- 실제 Docker 입력 manifest: [compiled-source-manifest.json](compiled-source-manifest.json)

Host는 기존 네 prefix, 독립 Workspace/cache 초기화, tokenizer/template 설정, 최대 총 128 tokens, 262,144 logits×4, finite/ID/metadata guard, f64 오차 집계와 null accuracy verdict를 그대로 사용한다. Host를 새로 복제·수정한 진단 코드는 없고 frozen source 파일을 그대로 복사한다. 이미 검증된 같은 host의 CPU guard 테스트를 이 source 변경을 이유로 다시 만들거나 확대하지 않았다.

## 빌드와 image 판정

```sh
sh moa-dev/experiments/candidates/model-control-full-o-original-e55/build.sh
```

[build.py](build.py)는 `/tmp/moa-model-control-full-o-original-e55` 안에서 47개의 실제 컴파일 입력을 확인한 뒤 아래 명령에 해당하는 별도 target을 사용한다. live crate나 `/workspace/target`은 건드리지 않는다.

```sh
cd /tmp/moa-model-control-full-o-original-e55
CARGO_HOME=/home/moa/.cargo \
CARGO_TARGET_DIR=/tmp/moa-model-control-full-o-original-e55/target \
FURIOSA_OPT_OUT_DIR=/tmp/moa-model-control-full-o-original-e55/target/furiosa-opt/kernel \
cargo furiosa-opt build --release --locked --bin probe_model_prefix_suite
```

[verify_registry.py](verify_registry.py)는 SHA로 고정한 기존 suite의 ELF/bincode parser를 재사용한다. 원본/E55 두 reference host도 실제 registry를 다시 읽고 frozen host/image SHA와 대조한다. 대조 host에서 정확히 10개 text entry와 empty generic key, standalone `.bin` 일치, 완전하게 소비된 sorted registry를 요구한다. 추가로 **full-O는 원본 image와 byte-identical, 나머지 9개는 E55 image와 byte-identical**이어야만 통과한다. 정확히 하나의 E55 image가 달라지는 조건도 검사한다.

원본 full-O의 요구 SHA는 `00f4ef931a38f9223ef27563468e663fb7e58c080fa53112dc96911ab1773daf` (126,976B)다. E55 full-O는 `0e0fb0fc9e2733f236eed383a9af7ab1dd40bfb0af4f4f81e3afd376ae3d9e4f` (142,336B)다. 다른 kernel의 예상 image가 하나라도 달라지면 단일-kernel 대조로 사용하지 않는다.

최종 ELF/image/build 결과는 [registry-check.json](registry-check.json), [build-result.json](build-result.json), [evidence.json](evidence.json)에 연결한다. Arena 제출·실행과 결과 해석은 root가 별도로 수행한다. Full-O 대조는 root의 통합 package에서 E55와 직접 비교할 예정이며, 다른 세 kernel의 원본 대비 대조와 기준 host를 혼동하지 않는다. Registry 일치는 device image의 귀속만 증명하며 실제 모델 입력에서 수치적으로 bit-identical한지 또는 정확도가 개선됐는지는 증명하지 않는다.
