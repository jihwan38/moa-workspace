# E128 fused-scale projection boundary validation

Arena **108341 SUCCEEDED**: eight cases × up/gate ×15,360 rows = **245,760 outputs
passed**, bitwise BF16 except signed-zero equivalence; nonfinite outputs0. This
is a separate diagnostic image and makes no performance claim.

- Result: [probe-results-v2b.json](probe-results-v2b.json).
- Build/source/image: [probe-build-v2b.json](probe-build-v2b.json).
- Original NPU log: [arena-108341.log](probe-evidence-v2b/arena-108341.log).
- CPU-only expected values/checksums: [oracle-only.log](probe-evidence-v2b/oracle-only.log).
- Independent E5M2 quantizer model: [audit](e128-fused-probe-v2b-quantizer-audit.json),
  [actual-host input binding](probe-model-binding.json).

The frozen candidate's entire28,283-byte `six-row-v1/mlp.rs` (SHA256
`3b94e94376d6a933c613102daa64501b7d802563758034255ef9fddb4864de8d`)
is a byte-identical prefix in the probe compile input. Its appended child
wrapper calls the actual `prepare_ffn_fp8_split` and `project_up_and_gate`, then
stores the original local gather/BF16 outputs in canonical L order. No copied
replacement dot, manufactured quantized input, or final RMS hides the changed
boundary. The root device entry reuses M03 v2's supported image-registration path.

The first six cases reuse M03's FP4/F64 oracle and add no changed arithmetic:
all16 FP4 codes with row identity/distinct scales, sparse block identity,
block cancellation, zero, and mixed2^-22 /2^-23 input values. Case6 exercises
K15/16,63/64 and3839 with sparse values; case7 uses amax128 (runtime scale2^-7)
and distinct row6/30/cluster scale patterns. Up/gate scales differ in cases
0/1/3/6/7. All rows, including5/6,29/30,7679/7680 and15359, are compared.
Output buffers start as NaN; all finite outputs are checked before GeGLU.

The independent Python finite-E5M2 RNE model restores all30,720 selected input
values exactly. Its BF16 input fingerprints match the actual Rust host buffers
in every case. The Rust oracle computes FP4 block dots and their scale/reduction
in F64; selected block/final F32 exactness assertions pass. These assertions alone
do not prove every possible native intermediate accumulation order, so the NPU
comparison is the final evidence for these cases. This does not establish general
BF16/full-model equivalence or remove the inherited three-term approximation.
Whole FFN nonlinear/epilogue and standard-input validation remains separate.

The downloaded Arena tar matches the actual host/run.sh package SHA256. Every
activation/weight/scale-value/expected-output fingerprint matches the CPU-only
run. The128,256-byte diagnostic image (SHA256
`0cb6c89c6507cdb0a700aee1df522ed4956d4102849e19b2495d1e3a2792c807`)
appears exactly once in the host binary. Archive/log paths and hashes are saved
in the result; artifacts remain outside tracked source.

V1 exact compile passed but its host attempted unsupported
`f8e4m3::to_bits()`. [V1 build/log](probe-build.json) and sources are preserved.
V2 changes only the checksum to decoded-F32 value bytes. Its first setup refused
to overwrite V1's fixed CPU-audit report path before invoking a compiler;
[that setup failure](probe-build-v2.json) is preserved. V2b uses a versioned output
path with identical V2 sources. [Repair verification](probe-repair-verification.json)
confirms V1/V2b device source and image are byte-identical. These are host/tooling
repairs, not additional kernel designs.

Offline verification (no compile, Arena submission or runtime edits):

```sh
python3 moa-dev/experiments/candidates/ffn-native-fused-scale/probe.py --check
python3 moa-dev/experiments/candidates/ffn-native-fused-scale/validation/probe_model_binding.py --check
```

Adoption and further research decisions belong to the complete-kernel evidence.
This probe contains no sub5% rejection policy and does not infer runtime savings
from its diagnostic execution.
