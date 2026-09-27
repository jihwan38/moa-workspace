# E135 independent direct six-row fused FFN

**Final outcome: runtime liveness unresolved; not adopted.** Complete candidate. Actual branch start `1643c6e`; source and contemporary performance control are independently declared E134 (`4f8fe7f`). Only `mlp.rs` is overridden. QKV/O, public fixtures and acceptance rules remain unchanged.

The candidate keeps independent compact FP4 up/gate transfers and directly consumes each in five six-row fused lookup/native/scale/reduce streams. This removes 112.5 MiB full decoded-weight DM write and 112.5 MiB read, without E131 joint weight/scale assembly. It adds eight lookup setups; table/register scheduling costs remain unknown. E134 weighted native down is unchanged.

E103 independent direct30-row with F32 block partial failed its oracle; missing final descriptors leave its cause unresolved. This candidate changes that consumer composition and must prove correctness anew. The separate v7c oversized joint-view ingress bug is not evidence that E103 had the same bug.

First gate: exact whole compile and actual compact source bounds, six-row offsets, native/scale/register bindings, then E134 before/candidate/after whole screen and independent original-BF16 projection oracle. A positive result enters frozen bounded validation irrespective of 5%; further parameter tuning and broader architecture decisions remain separate. No adoption or official submission occurs here.

Source/compile: `direct-six-row-v1/mlp.rs` SHA256 `be9bb5a3cd370f11220300f41239370de25e2a8aec52081568f29c345c7fb9f0`; exact compile label `e135-direct-six-row-v1`, static158934 cycles. The complete suffix from `cast_gathered_rows` through weighted down/final epilogue is byte-identical to E134. Static timing is not the hardware decision.

Reproduce the immutable first attempt (use a fresh label for builds):

```sh
python3 moa-dev/experiments/structural_build.py ffn-independent-direct-fused direct-six-row-v1 --label e135-direct-six-row-v1 --plan
python3 moa-dev/experiments/structural_build.py ffn-independent-direct-fused direct-six-row-v1 --label e135-direct-six-row-v1 --dump-graph
python3 moa-dev/experiments/candidates/ffn-independent-direct-fused/verify_graph.py e135-direct-six-row-v1 --check
python3 moa-dev/experiments/candidates/ffn-independent-direct-fused/package.py e135-direct-six-row-v1 --check
```

`probe.py` binds the entire frozen production file as the diagnostic prefix; its wrapper invokes actual `prepare_ffn_fp8_split` and `project_up_and_gate`. The eight selected dyadic cases reuse the independent original-BF16/F64 projection truth, not an E5-transformed reference. CPU E5 rounding analysis establishes exact representation only for those selected inputs; it is neither an NPU result nor a general BF16 assertion. The host checks all245760 first-BF16 row outputs, including six-row, block16/native64 and cluster boundaries, cancellation and tiny terms. Whole-FFN acceptance remains the original public host.

## Hardware result and decisions

- Whole allocation110314: E134 control before/after30PASS; candidate unchanged QKV/O12PASS; first candidate FFN launch times out with code4. No FFN output or timing is available.
- Production-prefix projection allocation110316: case0 launch also times out; zero of the planned245760 output comparisons execute. CPU oracle generation and exact source/image checks passed; the NPU oracle did not pass.
- Candidate source/image, original benchmark inputs, both uploaded archives and raw logs are preserved. `verification.json` PASS certifies that failed-attempt evidence, not kernel correctness.

Adoption is blocked by liveness. Parameter tuning stops because no specific supported repair was identified; no gain threshold is involved. The independent-direct architecture remains unresolved, not disproved or timed as a regression. No repeat/extra6 or new source variant follows this failed screen. See [three decisions](decision.json) and [runtime diagnosis and odd225 hypothesis](runtime-analysis.md).

```sh
python3 moa-dev/experiments/candidates/ffn-independent-direct-fused/verify.py --check
```

`results.py` is the retained successful-screen reporting path and intentionally refuses this invalid screen. The failed [whole result](e135-direct-six-row-screen-job-110314-results.json) and [projection result](validation/probe-results-v1.json) are the source of truth here; no valid candidate performance report was fabricated.
