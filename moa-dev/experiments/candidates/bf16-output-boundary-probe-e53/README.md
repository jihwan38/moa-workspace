# E53 BF16 activation O 경계 검사

**Arena job70810에서 8/8-case, 총30,720개 output이 모두 tolerance PASS했다.** E47 경계 검사 및 원본 대조와 동일한 host source·8-case payload·runner를 E53 BF16 O image에 연결했다. 공식 성능이나 순위 측정이 아니다.

제출 패키지: `../../artifacts/bf16-output-boundary-probe-e53-package/`.

| 파일 | 크기 | SHA256 |
| --- | ---: | --- |
| test_runtime | 2,311,560 B | `b0649dba5f36141e20ee7be1f34853120ef55dbfa69bb4660bf48fd395dc23eb` |
| boundary.safetensors | 379,768 B | `35df91a34ca00b077b35327c513bb36783355363855e214674ddaa04a47ec907` |
| remote_entrypoint.sh | 318 B | `be29bae1a4647066c647ce899e6eeba3e75aec1cacaf7a0bcb3f0742f4728da8` |

O image 71,168 B는 parent가 exact compile한 후보와 byte-identical한 `359473814839effc16ade242bad3c598a93610601bc650f550ff44228898d88b`다. standalone host의 registry는 이 커널 한 개이고, image가 전체 host에 정확히 한 번 포함됨을 `registry-summary.json`으로 검증했다. `expected-o.bin`/`compiled-o.bin`도 별도 artifact에 보존했다.

동결 E47 src에서 `src/device/sliding/projection.rs`만 `../output-bf16-activation-e47/projection.rs` SHA256 `eb77ad0076e601606bf614ac990be3cba211378f66fda4e0fad83c2834724e3f`로 교체했다. 기존 공식 host/ops 및 다른 device 파일은 그대로이며 새 `probe_output.rs`는 E47 진단과 byte-identical이다. `source-origin.json`과 `manifest.json`에 source·package hash를 기록했다.

E53 `project_output`은 activation을 BF16 TRF에 유지한다. FP8 weight를 BF16 lookup으로 읽어 BF16 contraction하고, 16개 column group의 FP32 결과를 합친다. channel scale까지 FP32로 유지한 뒤 BF16으로 cast한다. 따라서 E47의 input amax/hi-lo quantization을 제거했으며 scale 이전 추가 BF16 경계도 제거한 후보다. 이 진단은 parent가 정한 소스를 그대로 빌드하며 추가 연산/shape 수정은 하지 않았다.

CPU oracle, 허용오차(.05/.01), 각 case 및 관측 한계는 [공통 8-case 설계](../output-boundary-probe-e47/README.md)에 있다. 작은 BF16 입력/expected는 같은 safetensors 파일이고 weight는 같은 Rust 좌표식으로 생성한다. Host-only 실행에서 40개 입력 checksum과 8-case를 완료했고 Device를 생성하지 않았다. 전체 수치 FAIL을 기록한 뒤 exit1을 반환하므로, 향후 NPU 결과에서 exit1만으로 case 누락/timeout으로 분류하면 안 된다.

Docker scratch `/tmp/moa-bf16-output-boundary-probe-e53`에서 설치된 compiler/std0.8.1과 고정 Cargo.lock을 사용했다. 독립 target/kernel output으로 host build **1회, 실패0회**였다. source/log snapshot은 `../../artifacts/bf16-output-boundary-probe-e53/`다. Binary/payload는 ignored artifacts에만 둔다.

```bash
cd /tmp/moa-bf16-output-boundary-probe-e53
export CARGO_TARGET_DIR=$PWD/target
export FURIOSA_OPT_OUT_DIR=$PWD/target/furiosa-opt/kernel
cargo furiosa-opt build --release --locked --bin probe_output
target/release/probe_output --fixture /tmp/moa-output-boundary-probe-e47/suite/boundary.safetensors --host-check-only
```

이 agent는 live source/Git을 변경하거나 NPU를 제출하지 않았다. Root가 제출한job70810을read-only collector로수집했다. [원본 대 E47 실측 비교](../../../../docs/experiments/output-boundary-comparison.md)에서는 원본 case7 PASS/E47 FAIL을 확인했으며, E53은 case0–5와case7 전부maxerror0/exact3840, case6만maxerror1.015627119e-35였다. nonfinite=0, 입력40checksum과업로드3파일SHA/image1회검증도통과했다. [전체JSON](../../../../docs/experiments/e53-output-boundary-probe.json), `evidence.json`에 기록했다. 확인일 2026-09-21.
