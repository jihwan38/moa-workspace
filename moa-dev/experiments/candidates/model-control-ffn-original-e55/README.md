E55의 `decoder_feedforward` 이미지 **하나만 원본 `8559916`으로 되돌린** 전체 모델 진단 대조군이다. 첫 격리 release build와 [ELF registry 검사](registry-summary.json)가 통과했다. FFN은 원본과 byte-identical이고, 다른 9개 text image는 모두 E55와 byte-identical이다. 이 준비 작업에서는 NPU를 실행하지 않았으며 모델 정확도·속도 결과나 채택 후보를 뜻하지 않는다.

Subsequent root-managed [Arena job71282](../../../../docs/experiments/e55-model-control-ffn.json) completed.
See the [combined model results](../../../../docs/experiments/e55-model-single-kernel-controls.md).
The build evidence below remains a preparation-time snapshot.

## 변경 범위

[원본/E55 수치 차이 감사](../../../../docs/research/e55-model-numerical-differences.md)의 FFN 가설을 분리하기 위한 제어 실험이다. 전체48개 decoder layer가 호출하는 FFN operation을 원본으로 복원하므로 MLP뿐 아니라 원래 FFN post-RMS·residual·layer scaling 경로도 함께 복원한다. 특정 block reduction만의 효과를 분리하는 실험은 아니다.

- [mlp.rs](mlp.rs)는 원본 `src/device/shared/mlp.rs` 전체와 동일하다. FP4→FP8 lookup, local scale를 곱한 BF16 weight, 원래 dot/GeGLU/down 경로를 사용한다.
- [ops.rs](ops.rs)는 `decoder_feedforward`만 원래 본문으로 복원한다. 공개 signature와 나머지9개 operation은 E55 그대로다. 원래 `shared::residual::add` 호출 이름 하나만 `add_original_for_feedforward`로 바꾼다.
- [residual.rs](residual.rs)는 E55 전체 파일 뒤에 원본 `add`의 동일 본문을 전용 이름으로 추가한다. 기존 E55 `add`와 full-O 호출을 유지한다. Generic RMS와 `scale_by_layer_gate`는 원본 코드와 동일함을 별도로 확인했다.

[source.patch](source.patch), [source-audit.json](source-audit.json)에 재구성 범위를 기록했다. 다른 agent의 독립 읽기 감사도 47개 파일·세 변경 파일·원본 함수 본문·기존 E55 residual 전체 prefix·나머지9개 ops의 일치를 확인했다. 최종 image 비교에서 전용 helper 이름 변경도 원본 FFN의 컴파일 결과를 바꾸지 않았다.

## 재현과 이미지 검증

```bash
# workspace root에서 실행; 모델이나 NPU를 실행하지 않는다.
python3 -B moa-dev/experiments/candidates/model-control-ffn-original-e55/build.py
python3 -B moa-dev/experiments/candidates/model-control-ffn-original-e55/verify_registry.py
```

[prepare_sources.py](prepare_sources.py)는 기존 [four-prefix suite](../model-prefix-suite-e55/README.md)의 원본/E55 동결 입력을 모두 크기와 SHA256으로 검사하고, E55에 위 세 변경만 적용한다. 기존46개 crate 입력과 **변경하지 않은** `probe_model_prefix_suite.rs` 하나, 총47개를 전용 [crate](../../artifacts/model-control-ffn-original-e55/crate)에 보존한다. 기존 경로에 다른 내용이 있으면 덮어쓰지 않는다. Live crate, 기존 원본/E55 후보, 공식 harness, loader/tokenizer/runtime은 수정하지 않는다.

[build.py](build.py)는 Docker `/tmp/moa-model-control-ffn-original-e55`의 독립 source·target에서 `cargo furiosa-opt build --release --locked --bin probe_model_prefix_suite`를 실행한다. [source-manifest.json](source-manifest.json)과 [compiled-source-manifest.json](compiled-source-manifest.json)은 동일47개 입력이며, 빌드 전후 실제 Docker 파일의 크기·SHA도 동일했다. 첫 target은 비어 있었고 build는 **113.44초**에 통과했다([build-result](../../artifacts/model-control-ffn-original-e55/build-result.json), [log](../../artifacts/model-control-ffn-original-e55/build.log)). 이 시간은 공유 CPU에서 관측한 빌드 시간이며 NPU 성능 수치가 아니다.

환경은 [build-environment.json](build-environment.json)의 기존 Docker image·SDK0.8.1·Rust nightly이며, Cargo.lock은 원본/E55의 `aaff8dd41f9c643dd8c9faa358b9bc780e88a42c09172c1a5d7a0726d07efe09`다. 이미지 재빌드나 dependency 변경은 수행하지 않았다.

[verify_registry.py](verify_registry.py)는 기존 검증된 suite의 ELF parser를 사용한다. 원본/E55 참조 host가 기존 감사된 hash와 같은지 먼저 검사한 뒤, `furiosa_kernels` section 전체를 파싱하여 정확히10개 text entry와 empty generic key를 확인한다. Embedded image와 standalone `.bin`도 각각 비교한다. **원본 FFN + E55의 나머지9개**가 모두 맞고 E55에서 달라진 이미지가 FFN 하나뿐일 때만 `EXACT_SINGLE_KERNEL_CONTROL_PASS`를 출력한다.

| 산출물 | Bytes | SHA256 |
|---|---:|---|
| [FFN-only original control host](../../artifacts/model-control-ffn-original-e55/probe_model_prefix_suite) | 11,361,200 | `e843f5119c2b0b729a310029f630b0beae72f7d013ed58a81aee704a9866fada` |
| 원본과 동일한 `decoder_feedforward` image | 143,360 | `345c1bd08bf2b475ad5a2a926a6614e8cd132ed4a397a396b8df17cc2b616395` |
| 변경하지 않은 four-prefix host source | 13,434 | `07cfaf4646da18a78e03bb386632ac455ccf545d98b858385bded281c3104da4` |

## 이후 실행의 해석

Host는 기존과 같은 `--emit reference.json` / `--compare reference.json` 인터페이스를 사용한다. 동일한 네 prefix·토큰 합128 이하·case마다 새 Workspace·전체262,144개 finite logits 비교를 유지하고 `accuracy_pass`와 정확도 threshold는 `null`이다. Reference는 원본 모델의 진단 결과이며 독립적으로 입증된 정확한 수학적 oracle이 아니다.

원격 패키징·제출·수집은 root가 별도로 수행한다. 같은 job에서 자산 identity와 reference hash, actual uploaded host·runner를 검증해야 한다. FFN 복원에 따른 최종 logits 변화는 이후 모든 layer·KV 상태를 거치므로 다른 control의 차이와 선형 합산할 수 없다. Image 귀속 PASS를 모델 정확도 PASS나 공식 Stage2 결과로 해석하지 않는다. 현재 증거는 [evidence.json](evidence.json)에 묶었다.
