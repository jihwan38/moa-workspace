# FP8 quantization을 packed DM에 직접 기록

2026-09-21, compiler/std 0.8.1. `qkv-fp8-two-lane-row-interleaved` 대비 quantizer의 저장 위치만 바꿨다. Exact QKV compile은 성공했으며 static makespan은 52,428 → **51,574 cycles**다. 이 후보 자체의 NPU 검증은 수행하지 않았다. **기반 canonical/row two-lane 후보가 E20/E21에서 V-only FAIL했으므로, 이 분리를 해결하기 전에는 채택하지 않는다.**

기존 quantizer는 hi와 lo를 각 DmTensor에 commit한 후 각각 packed tensor의 `Dummy2` component로 DMA했다. 이 후보는 두 값의 산술을 그대로 유지하고, 처음부터 `DmTensor<f8e4m3,...,m![Dummy2,H%480]>`를 만든 뒤 `commit_view`로 component 0/1에 쓴다. lo 계산용 hi VRF도 packed tensor의 component 0 view에서 읽는다. 두 in-slice tile은 서로 다른 480-byte 범위다. input quantization, contraction, row mapping, head normalization, O projection, ops는 변경하지 않았다.

`projection.rs`와 `ops.rs`는 완성 소스이고 `candidate.patch`는 기반 후보 대비 변경이다. source hashes와 exact compile 요약은 `evidence.json`, 실제 compile 출력은 `compile.log`에 있다. scratch는 `/tmp/moa-qkv-fp8-direct-packed-quant`이며 `CARGO_TARGET_DIR`와 `FURIOSA_OPT_OUT_DIR`도 해당 scratch 내부였다. live source, Git, NPU job은 건드리지 않았다.

static 감소는 correctness 증거가 아니다. mutable Element tile의 의존성 lowering과 실제 NPU 읽기/쓰기 순서는 후속 oracle가 확인해야 한다. 원본 소스는 현재 동결한다.
