# QKV pair hi group에 명시적 ×1

`qkv-fp8-direct-packed-quant`의 동결된 row-interleaved/all-pair 소스에서 두 줄만 바꾼 원인 분리 후보다. Q와 공통 K/V helper의 pair Mul0를 다음처럼 바꾼다.

```rust
// before: group0 preprocessing 생략, group1만 /16
.vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), (), 0.0625f32)
// candidate: group0도 명시적으로 ×1
.vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1.0f32, 0.0625f32)
```

유한 FP8 입력의 dot accumulator에 ×1은 수학적 값을 바꾸지 않는다. hi+lo/16의 합산 순서, 입력 scale, BF16 cast, row channel scale, head RMS/RoPE는 그대로다. row-interleaved mapping과 direct packed quantizer도 바꾸지 않았다. `project_output` 이후 suffix와 `ops.rs`는 baseline과 byte-identical이다. 이 후보가 통과하더라도 일반 FP8 분해의 입력 범위·cancellation 한계는 사라지지 않는다.

실험 동기는 job70363에서 확인한 각 cluster의 첫8 slices/local row0 손상과 첫 pair pass가 V라는 consumer trace다. 기존 `()` group0 skip의 backend state 처리가 원인일 가능성을 분리하지만, 현재 원인으로 확정한 것은 아니다. 기존 진단의16개 직접 손상 위치는 원래 canonical E20 row layout 좌표이며, 이 후보는 row-interleaved layout이므로 실패 시 logical index와 physical slice를 다시 대응시켜야 한다. 특정16개 값에 대한 특수 처리는 없다.

독립 scratch는 Docker `/tmp/moa-qkv-fp8-explicit-hi-identity`이며 target/output 모두 그 안에 둔다. live source, Git, NPU 제출은 하지 않는다. 정적 compile 결과 및 source/kernel hashes는 `evidence.json`, raw log와 schedule은 `../../artifacts/qkv-fp8-explicit-hi-identity/`에 보존한다. 하드웨어 검증 전에는 채택할 수 없다.

Exact QKV compile은 성공했다. 정적 makespan51,574 cycles, instruction165개로 base와 동일하며 세 pair node의 시간도 각각746/1,226/746 cycles로 같다. FOPT binary 크기183,296bytes도 같지만 내용24bytes가 바뀌었다. 변경 구간에는 기존0.0625 immediate 외에1.0f32 bit pattern이 나타나므로 최소한 같은 binary로 identity가 완전히 제거된 결과는 아니다. backend instruction decoder로 전체 의미를 증명한 것은 아니다.

후보 `projection.rs` SHA256은 `1bfa8876a4dc21cb0fd00fccee4cad0231bf588826e0b6b54210ac39e63e09f0`, QKV kernel SHA256은 `0acfd2e3511b29ac4760a5c33facef17dc06acc958f6f8dc6e8b168d405d0538`이다. 이 폴더의 O suffix는 실험 base를 그대로 보존하므로 더 최신 O 구현을 덮어쓸 이유가 없다.

후속 [E28 job70410](../../../logs/e28-explicit-pair-identity-job-70410.log)에서 QKV 공식3개 입력의9개 출력 비교가 모두 PASS했다. QKV cycles106,310 /97,030 /98,428, median98,428이다. 전체 job은 후속 FFN 첫 launch timeout으로 완료되지 않았으므로 전체 PASS로 기록하지 않는다.

**이 결과만으로 `()` skip이 원인이며 ×1이 수정이라고 결론내릴 수 없다.** 실패했던 E20/E21과 달리 E28은 direct packed quantizer도 포함하고, all-pair direct-packed `()` baseline의 NPU 대조는 아직 없었다. Root가 같은 소스에서24-byte 차이인 skip baseline을 분리 측정할 예정이다. 일반 입력 정확성 및 원인 주장은 그 결과와 별개 검증이 필요하다.

E30 대조군의 소스·24-byte binary·전체 ELF 포함 관계를 독립 확인했다. 자세한 범위와 한계는 [control-audit.md](control-audit.md), 기계 검증 결과는 [control-audit.json](control-audit.json)에 보존했다.

E30 skip 대조군 job70440에서 공식15 comparisons 모두 PASS(QKV median98,599)했다. 따라서 명시적 hi×1이 E28의 통과에 필수였다는 가설은 이 대조로 지지되지 않는다. direct-packed 배치/스케줄 변화와 이전 V 손상의 근본 원인은 여전히 미확정이다.
