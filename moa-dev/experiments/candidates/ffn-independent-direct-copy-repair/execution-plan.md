# E141: exact three-copy repair of the frozen E135 independent direct FFN

Prepared only; no candidate repository mutation, build or NPU call. `/tmp/moa-e141-prepare-copy-repair.py` generated `/tmp/moa-e141-preparation/mlp.rs` and exact source diff. Source hash is `17414e1f320c82aea5cbb556a69bae3d2546488d2f7369dfdb4c2045e48ca83a` (37888B). The script checks both source origins and replaces exactly three copy expressions. All other E135 MLP bytes remain unchanged.

Source origin is `ffn-independent-direct-fused/direct-six-row-v1/mlp.rs`, SHA `be9bb5a3cd370f11220300f41239370de25e2a8aec52081568f29c345c7fb9f0`. Replacement streams are copied from E140 `control-copy-fix/mlp.rs`, SHA `9ca1e995272b14983d1a03db7381ff27dde05811606693bdb2c5b09f04128b7d`: `device.sub.begin(term.view()).fetch<H/32%15,H%32>().collect<...>().commit_trim<H%32>().commit_view(...)`. Each term copies480B at0/480/960 into1440B. No weight geometry, six-row native body, term arithmetic, down, padding, input or output changes.

The future recipe's base commit must be the actual adopted E140 source commit, and its other45 compile inputs must match E140. MLP is the explicit E135-origin override above. Separate comparison is `baseline/e140.json`; source reference build is `down-native-consumer-local/measured-build-e140-copy-adoption.json`. Root supplies adopted commit/tag and actual new branch-start commit. Do not retain the E135 E134 control: its O image is old `fe90...`, whereas E140 inherits E137 O `4dde46...`. New host must embed exactly E140 QKV/O and the newly compiled FFN.

The rationale is a demonstrated source-contract hazard, not a guessed table or bank fault. The original E135 whole and up/gate oracle each copy1440B from480B terms. Lo/third writes intersect16 live ring32 `custom_snoop_bitmap` control entries. The bitmap's actual graph consumer and embedded exact-image contents were verified. Old jobs110314/110316 timed out with code4 before producing candidate outputs. The audit does not locate the runtime timeout instruction or prove that correcting copies will restore progress. `/tmp/moa-e141-preparation/e135-bitmap-overrun-audit.*` freezes this basis.

Reused files and strict gates:

1. Use E140 `build.py/common_build.py` or the equivalent existing structural builder with explicit recipe, exact kernel `ops::decoder_feedforward`, and frozen graph/schedule. No live-source replacement. New candidate owns its own source/build records.
2. Adapt E135 `verify_graph.py`: retain independent packed HBM coverage, separate scales, ten six-row lookup/native streams, row indices0/6/12/18/24, F32 row-pair8B commits, two512×120B gathers, VRF/TRF consumer lifetime and unchanged suffix checks. Compare the suffix to E140 copy-fixed source rather than old E134. Add all three emitted Sub480B read/write ranges plus offsets0/480/960, and ensure no copy reaches the live Switch bitmap. Graph IDs must be derived or reverified from the new graph, not inherited blindly.
3. Adapt current E140 `package.py/screen.py/results.py` strict source46/current-control3images/fixtures/runner/upload verification to E141/E140. Keep three stages E140-before → candidate → E140-after, original3inputs,45comparisons,70s. Host build may overlap read-only gate checking; Arena requires a passing actual gate. First whole execution establishes liveness and output correctness before cycles become a performance result.
4. Reuse E135's actual `probe.py` plus all five `validation/probe-v1` source files byte-for-byte for wrapper/host/truth. Only builder recipe/source/label/namespace binding changes; historical `E135_*` host tokens can remain to preserve host code identity and are explicitly documented. Wrapper calls actual `prepare_ffn_fp8_split` then actual `project_up_and_gate`, and stores the first BF16 projection outputs before GeGLU. New production MLP must be the byte-identical prefix in the diagnostic crate. Exact oracle image is newly built; old c7a8... is not inherited.
5. Oracle's eight unchanged original-BF16 cases check245760 outputs (8×2×15360), all FP4 codes, row5/6 and other six-row boundaries, cluster7679/7680, K15/16/63/64/final3839, distinct up/gate scales, cancellation, zero, tiny2^-22/2^-23 and amax128. CPU quantizer audit proves exact restoration only for these selected dyadic inputs. F64 originalBF16 dot truth checks block and final F32 representability; no transformed-E5 truth or general BF16 claim. Keep original bitwise comparison, signed-zero equivalence and nonfinite rejection.

After root creates the named worktree and baseline, commands follow existing forms (labels provisional until root selection):

```
python3 <candidate>/build.py <candidate-name> direct-six-row-copy-v1 --label e141-direct-six-row-copy-v1 --plan
python3 <candidate>/build.py <candidate-name> direct-six-row-copy-v1 --label e141-direct-six-row-copy-v1 --dump-graph
python3 <candidate>/verify_graph.py e141-direct-six-row-copy-v1
python3 <candidate>/verify_graph.py e141-direct-six-row-copy-v1 --check
python3 <candidate>/package.py e141-direct-six-row-copy-v1
python3 <candidate>/screen.py prepare e141-direct-copy-screen e141-direct-six-row-copy-v1
python3 <candidate>/screen.py submit e141-direct-copy-screen
python3 <candidate>/screen.py collect e141-direct-copy-screen
python3 <candidate>/probe.py build
python3 <candidate>/probe.py submit
python3 <candidate>/probe.py collect
python3 <candidate>/probe.py --check
```

Whole source/host use one assigned warm target; projection oracle uses a separately coordinated warm target. At most two Arena jobs may be active, each70s maximum. No new profile, seed sweep or generic oracle framework is needed for the first decision. If the repaired whole passes and improves, freeze for two independent repeats plus extra6 and required boundary correctness; gain below5% is eligible. A correct whole regression is a performance result for this repaired combination; a continued timeout remains liveness evidence and needs a newly demonstrated mechanism for another repair. Copy-only correctness evidence cannot be inherited as whole performance. Root owns adoption, tuning and next architecture as three separate decisions.
