# E142 full30 production up/gate oracle

This diagnostic binds the entire frozen E142 `full30-v1/mlp.rs` from its actual whole build manifest, then appends the original E141 wrapper without changing it. It calls the production activation quantizer and `project_up_and_gate` and stores their canonical first-BF16 outputs before GeGLU. The new full30 direct native, F32 block-partial materialization and separate scaling/reduction all execute inside that production helper; none is replaced by oracle arithmetic on device.

Five fixture files are byte-identical to E141, including the original eight finite inputs, independent F64/BF16 references, signed-zero equivalence, NaN output initialization and nonfinite rejection. Historical E135_ log markers remain intentionally. The builder requires all8 input-checksum rows and16 projection-checksum rows to match E141's frozen CPU oracle. The new partial layout and consumer do not inherit prior correctness; NPU must check245760 outputs again.

The independent emitted gate checks the three480B up-term copies plus full30 partial row/block coordinates, scale application and canonical gather. This diagnostic does not time the whole FFN, execute the down/RMS suffix, or prove arbitrary-input/model equivalence. Whole standard performance and root's full-FFN stress remain separate. Target `/tmp/moa-c09-validation-probe/target` is exclusive and separate from the m03 whole build.

```sh
python3 moa-dev/experiments/candidates/ffn-direct-full30-partials/probe.py build
python3 moa-dev/experiments/candidates/ffn-direct-full30-partials/probe.py verify
python3 moa-dev/experiments/candidates/ffn-direct-full30-partials/probe.py submit
python3 moa-dev/experiments/candidates/ffn-direct-full30-partials/probe.py collect
python3 moa-dev/experiments/candidates/ffn-direct-full30-partials/probe.py --check
```

Build and NPU result records are created only when these actions execute. Source/shape errors require a concrete supported correction with failed evidence preserved; no test input or precision criterion is relaxed.

Job111704 completed and passed all8 cases /245760 exact BF16 outputs (signed-zero equivalent), with zero nonfinite outputs. The actual uploaded archive and production-prefix source hashes pass `probe.py --check`. Independent emitted review verifies three480B copies, two7200-F32-per-slice full30 partials, matching six-row partial/scale consumers and canonical output pairs, with register lifetimes. Recheck the frozen summary with `validation/verify_oracle.py --check`. Whole-path adoption and performance decisions are deliberately separate.
