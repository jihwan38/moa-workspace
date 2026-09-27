> 공식 제출 [`892178e8`](../../../../docs/experiments/e134-official-submit-20260927.json)도 완료:
> FFN314619cycles, 6.1405524×, 2026-09-27 18:04 KST 51/69위. 아래 원본의 미제출 표시는 당시 상태다.

> 채택 완료: **E134 / baseline-e134 / 4f8fe7f**. [현재 구성](../../../../baseline/e134.json),
> [후속 채택 결정](adoption-decision.json), [채택 소스 검증](validation/adoption-verification.json).
> 아래 frozen validation의 pending 표시는 root 채택 전 상태이며 원본을 보존한다.

# E134 weighted-residual native down

Frozen `weighted-v3` improves whole FFN against contemporary E128 in all three independent allocations. Paired median change is **−4.451846%**, with candidate314266 versus paired control328908.5 aggregate cycles. All135 standard-stage comparisons pass; they reuse three distinct standard inputs. Extra6 job109976 passes30 comparisons with identical device images, giving nine distinct standard inputs.

| Job | Candidate FFN | E128 before / after | Paired change |
|---|---:|---:|---:|
|109952|314224|329532 /328492|−4.494669%|
|109962|314266|328568 /329249|−4.451846%|
|109973|315179|329022 /327629|−4.004106%|

The actual-down oracle109958 passes all nine selected cases /34,560 BF16 outputs exactly (signed zero equivalent), including E133's failing tiny case4 and the cluster/row-half construction case8. Separate whole-FFN stress109965 uses identical constructed inputs and actual production images against the official PyTorch reference with unchanged tolerance: E128 and E134 match all3,840 outputs exactly; E133 fails1,920 outputs by0.029296875. The constructed input is within the generator's component ranges; it is not an observed PRNG seed or official grading run. These diagnostic timings are excluded from performance. `validation_verify.py --check` passes all required bounded evidence checks (`adoption_ready=true`). Root owns final live-source/tag/baseline adoption; this candidate has not changed live code or submitted an official entry.

This is one coupled precision repair of E133, **not a new execution architecture**. Source starts from frozen E133v2 `c3ae74b5…`, assembled over source baseline E128 `cc82b9c…`; Git work starts at `543a947…`. E128 remains the explicit performance control. Public input layouts and all other source files are unchanged. The same-scale E133 positive prototype and its numerical failure remain preserved.

Per cluster, compact post-GeGLU BF16 values select power-of-two scale s. For u=F32(x/s):

```
hi = E5(u)
lo16 = E5(F32((u-hi)*16))
third256 = E5(F32((u-F32(hi+lo16/16))*256))
```

The native down stream keeps `FfnTerm` in its F32 output, multiplies each term by the matching coefficient `[s,s/16,s/256]`, and only then sums terms in VE. The existing16-byte F32 partial commits, H60×block120 partial DM and five12-row blockscale/column/cross-cluster reductions remain unchanged. Native DPE time reduction must not sum weighted terms before their restoration. E128 up/gate/GeGLU and `finish_down`/public entries retain exact source bytes.

Three explicit32-byte scalar commits assemble a96-byte per-cluster coefficient tensor before broadcasting it to the existing owners. Direct FP4 lookup feeds native E5; there is no full decoded-down-weight DM or new weight assembly. This retains E133's logical removal of56.25MiB decoded-weight writes and reads each, while keeping14.0625MiB F32 partial writes and reads each. Added work relative to E133 includes three producer multiplications, coefficient setup/replication and native-to-VE individual term traffic/reduction. These storage counts do not measure bus traffic or predict reclaimable cycles. Static estimate153359 and hardware cycles above are separate evidence.

`numerical-bound.md/json` independently bound the normalized BF16/F32 representation: scaled residual inputs stay at most32768, maximum normalized reconstruction error is2^-25, and enumerated |u|≥2^-17 values reconstruct exactly. Still smaller values may round or vanish. This is not general BF16 equivalence, a bound on arbitrary projected/post-RMS error, or a full-model accuracy result. Selected raw-down, full-FFN stress and nine standard input checks remain distinct evidence.

V1 and V2 generated graphs but failed final SFR lowering at `sfr_impl.rs:868:33`. V2 moved only decoded lo16/16+hi from Sub to Main and retained the failure. V3 additionally moved the three new coefficient producers to Main; exact fullFFN compilation passed. All numeric/layout parameters stayed fixed. `compile-attempts.json` preserves source/build/log/graph hashes; this supports a composition-sensitive context repair without identifying the private compiler bug. A separate V1 down-only compile also failed; no V1/V2 NPU image was run. No tile or multiplier sweep occurred.

`verify_graph.py` checks emitted matrix coverage/coordinates, independent cluster scales,480-byte term copies,32-byte coefficient copies at0/32/64, coefficient ownership, native Term preservation→restoration→VE sum, original partial/reduction coverage and live register capacity. The coefficient reader's scalar addresses derive from the emitted symbolic mapping; binary SFR reader instructions are not independently decoded. Numerical NPU oracles provide separate evidence. `independent-compiled-review.md/json` records a second source/graph review.

Recheck without compile or NPU:

```sh
python3 moa-dev/experiments/candidates/ffn-native-down-weighted-residual/verify_graph.py e134-weighted-v3 --check
python3 moa-dev/experiments/candidates/ffn-native-down-weighted-residual/validation_performance.py --check
python3 moa-dev/experiments/candidates/ffn-native-down-weighted-residual/validation_extra.py verify
python3 moa-dev/experiments/candidates/ffn-native-down-weighted-residual/down-oracle/probe.py --check
python3 moa-dev/experiments/candidates/ffn-native-down-weighted-residual/full-ffn-stress/verify_results.py --check
python3 moa-dev/experiments/candidates/ffn-native-down-weighted-residual/validation_verify.py --check
```

Aggregate performance reports supersede the original first-screen decision for validation status; first-screen records remain immutable. All three uploaded standard payloads and canonical stage packages are byte-identical, and the aggregate reports satisfy `experiment_index.measurement/check_comparable`. Two repeat-b submission attempts returned no job; one confirmed rejection was the account's two-active-job quota. Attempts and no-duplicate checks remain preserved; only the three successful allocations above enter performance. Tuning stops on this frozen variant; a gain below5% did not prevent bounded adoption validation.
