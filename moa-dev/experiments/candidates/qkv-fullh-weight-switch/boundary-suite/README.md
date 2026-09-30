> 현재 결과: raw9 job **136561**의 73,728 BF16 outputs exact PASS와 full2 job **136578**의 16,384 outputs / 8,380,416 untouched cache values PASS가 완료되었다. [raw9 결과](raw-validation-summary.md), [독립 최종 검토](../adoption/independent-final-review.md)를 참조한다. 아래는 준비 시점 기록이며 `preparation.json`은 변경하지 않았다.

# E144 fullh-v2 boundary validation — prepared, not built or tested

This suite reuses E138 `boundary-suite-v2` tools and exactly the original raw9 wrapper, Rust host and F64 generator. It binds the actual successful `e144-fullh-v2` production build/source ledger in `preparation.json`. No E138 hardware result is inherited. The E144 v1 Switch lowering failure is not the selected source. Production source remains unchanged by this suite.

`verify_preparation.py --check` verifies the three unchanged raw source files, the copied whole2 fixture against the original E53 CPU reference, legacy whole2 host/runner/collector hashes, prepared tooling, and actual selected production source/build hashes. It makes no build or remote request. Preparation status remains `PREPARED_NOT_BUILT_NOT_TESTED` until separate evidence is generated; do not edit that historical status into a result.

The raw9 diagnostic invokes unchanged production `project_qkv_native` on stored BF16 post-preRMS inputs and observes BF16 Q/K/V before head RMS. Nine cases cover zero, K959/960/3839, row/group/cluster mappings, tiny cancellation, scale-before-BF16, and the fourth active term. Expected output is the original F64 stored-BF16/E4 dot and channel scale, checked exactly representable in F32 and then BF16 RNE (signed zero equivalent): 73,728 output values. Host regeneration must match all63 original input hashes. At build, generated truth must equal the historical raw-truth bytes. Exact diagnostic image is captured before host build and compared with the one host-embedded diagnostic image; it is not claimed identical to the whole production image.

The full2 path uses the unchanged E53 host, CPU reference, fixture and collector. It builds the actual complete production kernel; its QKV image must match the selected production exact-image digest and occur exactly once in the host. It checks16,384 outputs,38 actual input hashes and8,380,416 untouched cache entries. The two cancellation inputs lie outside the official generator support; raw9 inputs are selected mathematical cases, not a generic BF16 or reachable-PRNG guarantee. Standard whole performance and extra6 are separate.

No compile, host, CPU full-matrix oracle regeneration, submission, or NPU run was performed during this preparation. Root owns full2 execution; raw9 requires separate root authorization and a passed production graph gate. Compiler target ownership must be agreed before either command.

After authorization, from the repository root (choose the explicitly assigned target):

```bash
python3 moa-dev/experiments/candidates/qkv-fullh-weight-switch/boundary-suite/verify_preparation.py --check
python3 moa-dev/experiments/candidates/qkv-fullh-weight-switch/boundary-suite/raw_oracle.py build --build-label e144-fullh-v2 --target ASSIGNED_TARGET
# Then raw_oracle.py submit / collect / --check, only when authorized.
python3 moa-dev/experiments/candidates/qkv-fullh-weight-switch/boundary-suite/build.py --build moa-dev/experiments/candidates/qkv-fullh-weight-switch/build-e144-fullh-v2-repro.json --label e144-boundary-v1 --target ASSIGNED_TARGET
```

The full2 fixture defaults to this suite's byte-identical138,912B `fixtures/boundary.safetensors`; no ignored legacy path is required. After root submits that full2 package with name `beef-e144-boundary-v1`, `configure_full.py --job JOB` writes the exact collector config from the newly built package. Use the original `qkv-boundary-probe-e53/collect_qkv.py --config .../boundary-suite/collector-config.json`, then this suite's `check_full.py`. That checker verifies actual uploaded bytes and the production image. No full2 submission helper is added here.

Existing raw build/submission/results filenames are exclusive-write. On any real source or build change preserve the existing records and create a new suite/attempt; do not overwrite a frozen failed or successful observation.
