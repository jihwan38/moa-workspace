# Fixed-shape FFN direct-decode boundary recomputation

Closed after E123's first full-FFN screen; E122/M03 remains adopted. Job105045 passed60standard comparisons across four processes. FFN medians were E104336282 → E123329395 → M03331848 → E104335390: E123 reduced cycles by1.9179% versus the bracketing E104 mean and0.7392% versus M03, in one allocation. This is below the5% exploration screen. No repeats, expanded correctness, adoption or further tuning were performed. [Decision and limitations](decision.json), [index-compatible E123 report](../../../../docs/experiments/e123-ffn-direct-joint.json).

The retained finding is that direct lookup/native composition can pass the complete standard FFN gate after its source-transfer contract is repaired. The added packed assembly and combined lookup/native stream did not yield a large improvement. The isolated materialized diagnostic remains a recorded timeout. Its failure is separate from both valid production controls in the whole-FFN screen.

Branch `opt/ffn-direct-boundary-recompute`, actual starting commit `0a4417c076eff78ba885b0f142c66c81e9bc9b4f`. Source arithmetic and performance control are E104 `7f891f9c48e670fe014b2ac5bd2c92bae3bac433`; the direct producer is frozen D-FP4-NATIVE v7c. This is a conditional repair of that execution family, not a new independent architecture. This candidate freezes E104 inputs; baseline adoption is managed separately by the root agent.

[Full execution graph, ownership, arithmetic contract, costs, prior failures and first stop conditions](design.md) are recorded before implementation/compile. The first variant keeps every original joint Projection2/Row30/block240 operation. It recomputes projection0 row0 on all512 slices using compact FP8 materialization and overwrites the960B F32 block partial row before the original block-scale/reduction. The observed16 bad rows are not singled out. No input/fixture-dependent branch exists.

The first gate compares every final F32 row of both projections against an independent exact oracle over the four frozen v7c inputs. A materialized control reloads activation/scale separately for each projection, matching E104's proven operand-lifetime boundary; it is itself newly compiled diagnostic code and must pass. The unchanged direct v7c source producer is a negative comparison, not a performance candidate.

No speedup is claimed from this probe. Added HBM outputs change its boundary, and a passing primitive would still require complete FFN assembly, contemporary controls and expanded correctness. A shifted corruption region or unexplained nonfinite output closes the fixed-boundary hypothesis; source/API errors with a demonstrated repair are recorded and repaired. No fallback-coverage sweep.

## First compile and concrete source repair

v1 exact hybrid compile rejected the merge typed DMA: it attempted225x256B=57600B from a960B-per-slice compact source because destination padding described the whole joint allocation. No NPU run. The frozen error is `source_accessing_range_local.end <= input_tensor_range`. v2 replaces only this merge with an explicit Main fetch/commit of240F32/960B per slice; full joint shape, first-row scope and arithmetic stay unchanged. This is a source transfer-contract repair, not evidence the architecture is impossible.

v2 exact hybrid compile passed. Its host build rejected unused formatting arguments in the diagnostic decode error print. v3 removes those host-only arguments; probe.rs is byte-identical to v2. No numerical or kernel-layout repair was made.

## Inherited v7c joint-ingress descriptor defect

Before Arena, v2/v3 emitted graph inspection found both unchanged v7c joint HBM loads described58,982,400B each, twice the intended29,491,200B matrix. With57600B slice stride, the last source end is29,548,800B:57600B beyond the public matrix. [Exact descriptor audit](descriptor-audit-v2.json). This proves a generated transfer-contract defect in the inherited implementation, not its contribution to the previously observed16 bad rows. v3 compiled and packaged but was not submitted.

Root authorized v4 to first repair ingress: each public matrix loads into compact Row30 FP4 DM, then a raw Main stream copies exactly those values into its joint Projection tile. The complete joint lookup/native graph is retained. `direct_fixed` tests this repair alone; `hybrid` adds the fixed all-slice first-row fallback. The historical `fused` image remains available for source comparison but is excluded from the new runner because its emitted address range is invalid. Materialized control uses compact loads and per-projection operand reloads. All executed variants require descriptor-range verification before submission. Added packed DM assembly copies are real candidate costs, not free preprocessing.

## v4 compiled transfer contract and Arena gate

All three executed kernels compiled with exact graph/schedule/IR/VISA dumps. [Machine-readable descriptor audit](descriptor-audit-ffn-boundary-v4.json) rechecks each HBM FP4 ingress against its actual input allocation: materialized and corrected direct each read two29,491,200B matrices; hybrid additionally reads983,040B for row0 on512 slices. Both corrected producers add two29,491,200B raw FP4 Main copies. Their fetch/commit sequencers contain1800packets of64FP4 elements per slice, exactly57,600B. Hybrid's raw F32 merge contains30packets of8F32 elements, exactly960B per slice. The graph evidence is checked by `audit_descriptors.py`; these are logical descriptor volumes, not measured hardware transactions.

`verify.py ffn-boundary-v4` checks recipe/source/evidence hashes, all compile artifacts, packaged host and each embedded image, then descriptors and any collected oracle results. Its initial JSON comparison used tuple dimensions against reloaded list dimensions; this verifier-only mismatch was repaired and verification passed before the job ran. Job105009 was submitted with the immutable v4 package; `arena.py` now also invokes verification directly before any future submission. The runner executes materialized control, corrected direct, and hybrid in that order and preserves all stage exits.

## Arena105009 result and E123 complete-FFN gate

[Verification](verification-ffn-boundary-v4.json) confirms corrected `direct_fixed` and `hybrid` each passed122,880F32 row comparisons exactly across4cases and both projections, with zero nonfinite values. The materialized diagnostic timed out at its first kernel launch (exit101, zero comparisons); it is unresolved and **not a passing control**. Overall job status isFAILED because this stage failed. Full E104 remains a separate proven production source/package, to be rerun in the next comparison. Corrected direct does not need the first-row fallback. The result supports revisiting the historical DMA contract; it does not isolate DMA bounds from all added-copy scheduling effects or prove a hardware root cause.

Root authorized E123 `whole-v1`: replace only E104 `project_up_and_gate`, preserve production quantizer and all consumers. Public FFN graph remains preRMS → actual3termE5 quantization/replication → up/gate → BF16 GeGLU → E104 down block16F32 → BF16/global scale → postRMS/residual. The new up/gate graph is compact FP4 HBM load per projection → raw FP4 Main copy into joint Projection2/Row30/H3840 DM → one complete inline lookup/native stream with input TRF and restore-scale VRF → joint F32[2,30,240] DM partials → unchanged per-projection5×6row blockscale multiply and F32 reduction → two F32 row outputs → existing cross-cluster F32 gather/BF16 cast. No intermediate HBM projection output, fallback, free preprocessing, or public interface change remains.

Row ownership stays `l=7680*c+30*s+r`. Compact input and joint packed matrices survive their Main assembly/read; the quantized activation DM is consumed by Sub-to-TRF, restore scale by Sub-to-VRF. Joint partials survive both projection reductions. Every Main commit has exact element sequencers, and actual descriptors must be re-audited after whole-graph compilation. Existing F32 row gather and BF16cast define the consumer boundary; global scales, erf, down reduction order and final epilogue are inherited unchanged.

Logical up/gate costs: remove112.5MiB FP8 DM writes and112.5MiB native FP8 DM reads. Add56.25MiB joint packed DM writes and56.25MiB packed native reads. The existing56.25MiB packed decode-read role becomes raw assembly-read; compact56.25MiB HBM ingress/DMwrite stays. Thus both DM write and DM read decrease by56.25MiB logically, with real raw-copy work, changed lookup/native dependency, and potentially longer joint buffer/register lifetimes. Activation TRF/scale VRF reload occurs once for the joint stream instead of once per projection. F32partial28.125MiB and per-row scale/reduction remain. Bytes and static schedule are not measured bus traffic or additive runtime savings.

First whole-FFN screen uses unchanged standard hosts/fixtures in order E104before → E123 → original M03 → E104after. Source manifests, all embedded image hashes and the actual uploaded archive must match. All controls must be valid; isolated oracle results do not replace full public correctness. Target gain is reported before root decides further validation.

The frozen whole-v1 recipe repeats the assembly source read under both added and unchanged work. [Cost accounting correction](cost-accounting-correction.json) preserves that recipe and counts it once. The corrected weight-only table is:

| Logical up/gate weight bytes, MiB | E104 | E123 |
| --- | ---: | ---: |
| HBM read / compact packed DM write | 56.25 / 56.25 | 56.25 / 56.25 |
| Compact packed DM read (decode / raw assembly) | 56.25 | 56.25 |
| Intermediate DM write (FP8 / joint packed FP4) | 112.5 | 56.25 |
| Native stream DM read (FP8 / packed FP4) | 112.5 | 56.25 |
| Listed DM writes total | 168.75 | 112.5 |
| Listed DM reads total | 168.75 | 112.5 |

Exact compile/host build and whole source/image/weight-descriptor audit passed. E123's static schedule is157,208cycles versus E104157,045; this is not an NPU speed estimate. Job105045 executes the four-stage full-FFN gate. Reproduction commands, each with a fresh label when building/submitting, are `whole_build.py ffn-direct-boundary-recompute whole-v1 --label LABEL --package`, `whole_verify.py LABEL --write`, and `whole_screen.py prepare SCREEN LABEL` → `whole_screen.py submit SCREEN` → `whole_screen.py collect SCREEN`. Run from the workspace root with the scripts under this candidate directory. The builder retains the standard structural source plan and uses the isolated `/tmp/moa-c01-probe/target` compiler directory.


The primitive oracle uses the same blockscale tensor for up and gate, so it cannot detect a swapped scale-input binding. The complete graph has distinct public scale bindings and passed standard correctness; the root's independent review also checked projection strides, output bounds, and IO5/IO6 ownership. This evidence is still limited to the recorded inputs and does not substitute for expanded correctness.
