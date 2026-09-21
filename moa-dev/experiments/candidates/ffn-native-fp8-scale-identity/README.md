# FFN pair Mul0의 group0에 명시적 ×1 적용

**후속 E28 job70410도 첫 FFN launch에서 timeout으로 종료했다(exit101, rank0).** Q/O는 PASS였으나 FFN output/cycle은 얻지 못했다. Group0에 ×1을 명시하는 변경만으로 FFN timeout은 해결되지 않았다. 원인은 아직 확정되지 않았으며 rank 번호를 특정 cluster/instruction으로 해석하지 않는다. Source는 그대로 동결한다.

2026-09-21, compiler/std 0.8.1. E25 frozen `ffn-native-fp8-scale-contraction/mlp.rs`에서 **한 줄만 바꾼** 진단 후보다.

```rust
// Before: group0 skips Mul0, group1 multiplies by 1/16.
.vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), (), 0.0625f32)
// Candidate: both groups explicitly use Mul0.
.vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1.0f32, 0.0625f32)
```

Exact FFN compile PASS, static **166,220 cycles**로 E25 기준과 같다. instruction count도196개로 같다. 생성 kernel `.bin` hash는 달라서 산출물이 byte-identical하지는 않다. **후속 NPU 실행은 위 E28 timeout 기록을 따른다.**

Group0의 유한 hi-dot에 ×1을 명시하므로 의도한 산술은 같다. “이전 vector operation의 상태가 skip된 group에 남는다”는 가설을 확인하기 위한 변경이며 그런 버그가 있다는 정적 증거는 아직 없다. 32-row tile, packet당 네 block, runtime quantization/broadcast, BF16 block partial, 두 번째 scale contraction, GeGLU/down과 ops는 그대로다. 기반 E25의 추가 BF16 block 반올림과 수치 한계도 유지된다.

E25는 첫 FFN launch에서 timeout/exit101로 종료했고 output/cycle을 얻지 못했다. [공통 경로 감사](../ffn-native-fp8-scale-contraction/timeout-audit.md)에 확인한 사실과 미확인 가설을 구분해 기록했다. 별도 [pair-free 두-dot 진단](../ffn-native-fp8-two-dot-diagnostic/README.md)도 compile되었으나 이 후보에는 그 변경을 합치지 않았다.

scratch `/tmp/moa-ffn-native-fp8-scale-identity`는 E25 scratch의 source/manifests를 복사한 뒤 한 줄만 바꾼 MLP를 넣었다. source와 ops의 단일 변경 불변식을 검사했다.

```sh
./moa-dev/moa bash -c 'set -o pipefail; cd /tmp/moa-ffn-native-fp8-scale-identity; export CARGO_TARGET_DIR=/tmp/moa-ffn-native-fp8-scale-identity/target; export FURIOSA_OPT_OUT_DIR=/tmp/moa-ffn-native-fp8-scale-identity/target/furiosa-opt/kernel; cargo furiosa-opt compile ops::decoder_feedforward --exact --dump-schedule schedules/identity.json 2>&1 | tee logs/compile.log'
```

`mlp.rs`는 완성 소스, `candidate.patch`는 한 줄 diff, `evidence.json`은 source/schedule/binary hashes와 compile 결과다. `compile.log`도 보존했다. 소스를 동결하며 live source, Git, NPU 제출은 하지 않았다.
