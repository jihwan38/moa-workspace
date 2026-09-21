# 원본 O 커널 경계 대조군

**Arena job70799에서 8/8-case, 총30,720개 output이 모두 tolerance PASS했다.** E47 경계 검사 job70785의 case7 실패를 원본 커널과 대조하기 위한 실행 파일이다. 같은 8개 입력·원본 CPU oracle·host source·runner를 유지하고 device source만 upstream `8559916051162e5fd682c29ec0fe3e9cc6e420cf`의 원형으로 사용했다. 공식 score나 성능 측정용이 아니다.

제출 패키지: `../../artifacts/original-output-boundary-probe-control-package/`. 세 파일을 함께 제출하며 Root만 NPU를 실행한다.

| 파일 | 크기 | SHA256 |
| --- | ---: | --- |
| test_runtime | 2,374,792 B | `af263c63f4c5763b264a3c830aa50634e8e3e532f82addb946440841b0981c75` |
| boundary.safetensors | 379,768 B | `35df91a34ca00b077b35327c513bb36783355363855e214674ddaa04a47ec907` |
| remote_entrypoint.sh | 318 B | `be29bae1a4647066c647ce899e6eeba3e75aec1cacaf7a0bcb3f0742f4728da8` |

ELF registry에는 원본 `ops::sliding_attention_output` 한 개만 있다. image 134,400 B의 SHA256은 `434f864af2f46a75c06cf2c5f2f42053e705596ad75d397809a73f50b488ead4`이고 host 내 정확히 한 번 들어 있다. 이 image는 최초 원본 실측 host(`../../artifacts/original/test_kernels`, SHA256 `942c1958f8e643a4625e111e312e469263598e5da2efeebcaf74fb08f7a5336c`)에 들어 있는 O image와 byte-identical이다. `registry-summary.json`과 `inspect_registry.py`가 검증한다. 별도 복사한 kernel은 artifact `original-o.bin`이다.

## 고정한 입력과 변경 범위

`git ls-tree`와 `git show` 읽기만으로 upstream의 src 43개 파일, Cargo.toml, toolchain을 독립 scratch에 내보냈다. 원본 src는 수정하지 않았고, 추가 파일은 E47 진단과 byte-identical한 `src/bin/probe_output.rs` 하나다. `source-origin.json`은 upstream 파일별 SHA256을 기록한다. Cargo.lock은 원본 초기 빌드와 동일한 고정 파일 `moa-dev/locks/gemma4-Cargo.lock`이다. 이 작업은 checkout/branch/index/commit 또는 live source를 바꾸지 않는다.

Payload/runner를 재생성하지 않고 E47 동결 패키지에서 그대로 복사했다. CPU oracle은 [E47 probe](../output-boundary-probe-e47/README.md)의 원래 F32 matmul → channel scale/BF16 → RMS/BF16 → residual/BF16이다. 상대/절대 허용오차는 .01/.05이며 원본 장치에 맞춰 oracle을 바꾸지 않았다. 가중치도 같은 Rust 좌표식으로 만들어 upload package에 대형 weight 파일이 없다.

8-case host-only 실행에서 40개 입력 checksum과 expected payload가 통과했고 Device/NPU는 생성하지 않았다. 원본 O는 BF16 activation과 lookup으로 BF16에 확장한 FP8 weight를 contraction하는 경로다. E47의 native FP8 activation quantizer를 포함하지 않는다. O projection 이후 원래 shared RMS/residual도 upstream 그대로다. 따라서 이 대조는 **전체 원본 경로 대 현재 E47 경로**이며 내부 stage 하나만 바꾼 인과 실험이라고 주장하지 않는다.

E47 job70785는 case0–6 PASS, case7의 3,840개 전부 got=.5/want=.609375로 FAIL이었다. 이 대조군은 case7에서도 3,840개 모두 expected=.609375와 정확히 일치했다. 업로드 세 파일 SHA와 원본 O image1회 포함, 40/40 input checksum, nonfinite=0을 확인했다. [전체 JSON](../../../../docs/experiments/original-output-boundary-control.json), [E47 비교](../../../../docs/experiments/output-boundary-comparison.md)에 결과가 있다. 모든 수치 case를 계속 기록한 뒤 전체 실패가 있으면 exit1을 반환하므로 진단 FAIL을 실행 누락으로 처리하지 않아야 한다. floor case6은 기대값이 atol보다 극히 작아 zero라도 PASS하며, 원본 대조 역시 tiny bit 보존이나 FTZ를 독립적으로 증명하지 못한다. nonfinite 및 출력 bit sample은 그대로 기록한다.

## 재현

Scratch는 Docker `/tmp/moa-original-output-boundary-probe-control`, source/log snapshot은 `../../artifacts/original-output-boundary-probe-control/`이다. 설치된 compiler/std0.8.1로 독립 target/output을 사용해 host build **1회, 실패0회**였다. `manifest.json`에 전체 입력·package hash와 검증 상태가 있다.

```bash
cd /tmp/moa-original-output-boundary-probe-control
export CARGO_TARGET_DIR=$PWD/target
export FURIOSA_OPT_OUT_DIR=$PWD/target/furiosa-opt/kernel
cargo furiosa-opt build --release --locked --bin probe_output
target/release/probe_output --fixture /tmp/moa-output-boundary-probe-e47/suite/boundary.safetensors --host-check-only
```

정확도 fixture/설계 한계는 [E47 probe](../output-boundary-probe-e47/README.md), 기존 실측은 [job70785 결과](../../../../docs/experiments/e47-output-boundary-probe.json)에서 확인한다. Root가 job70799를 제출했고 agent는 해당 job만 읽어 수집했다. collector는 새 NPU 제출을 하지 않았다. 확인일 2026-09-21.
