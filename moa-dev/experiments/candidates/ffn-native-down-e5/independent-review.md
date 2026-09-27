# E133 independent source review

Source reviewed: [v1](down-e5-v1/mlp.rs) SHA256 `9614d57eec7f3465929480f2db6c3a4955d04edb35957d7805f486371d8838ee` and
[v2](down-e5-v2/mlp.rs) SHA256 `c3ae74b55b306742b69b438bf2ea603409a0644c55f9004d7fe01665c656a5d3`.
This review checks source arithmetic and ownership. It does not independently
certify lowered descriptors, NPU correctness or performance. The source-only
review did not expose v1's expanded DMA extent; the producer/consumer graph gate
did, so v1 is not safe to package despite its plausible typed source.

- `UpGateCluster = m![Dummy2]` is a real two-cluster L-half owner. It remains on
  the compact input, amax/scale, spatial quantizer, broadcast, TRF and F32 partial.
  The two 7680-value halves do not accidentally share one scale. `DownQuantSlices`
  assigns sixteen 480-value pieces before replication. The ring32 transfer keeps
  the existing column ownership and adds the three-term axis.
- `project_down_tile` supplies E5 TRF `[3,1920]`, or5760 bytes/slice. The native
  contract reduces the term axis while retaining four separate16-element block
  lanes. F32 input scale is restored before four-F32/16B partial commits; the
  existing blockscale F32 multiplication and local-K reduction consume those
  partials. There is no new BF16 blockpartial cast. The unchanged downstream
  12-row stores commit F32 row pairs/8B, then four columns and two clusters are
  summed before the original BF16 boundary.
- Source prefix before the new helper and suffix from `let mut down` through
  `finish_down`, global scale, residual/RMS and original up quantizer are byte
  identical to E128. This equality was checked directly, not inferred from names.
- For cluster `c`, slice `s`, local row `r` and local K `k`, public input mapping
  is `h=60*(s//4)+r`, `l=7680*c+1920*(s%4)+k`. For `c<2,s<256,r<60,k<1920`,
  the final packed-weight byte end is29,491,200 and blockscale byte end is
  3,686,400, exactly the public tensor extents. These are mathematical bounds;
  the graph gate must separately verify actual emitted descriptors and pitches.
  F32 partial `[60,120]` is28,800 bytes/slice,14,745,600 bytes overall.

[v1 descriptor failure](graph-verification-e133-down-e5-v1.json) records1440B
reads from each480B term source and term1/2 writes past the1440B destination.
This is a concrete newly introduced transfer-contract defect, not a measured
arithmetic or performance loss. v2 replaces only those three down-assembly
`to_dm_view` calls with Sub fetch/collect/commit of15×32B at offsets0/480/960.
The source repair is consistent with the intended480B copy. A fresh exact
compile and descriptor gate must establish the emitted behavior; v1 evidence
must remain unchanged. The inherited up quantizer is outside this repair.

The actual quantizer performs `hi=E5(x/s)`, `lo=E5(x/s-hi)`,
`third=E5(x/s-F32(hi+lo))` with one power-of-two scale per cluster. It is an
approximation for arbitrary BF16 inputs. At amax128, scale is2^-7 and x=2^-24
becomes exactly half the smallest E5 subnormal, so RNE can erase all three terms.
The recipe exposes this limitation correctly. Neither E128 up/gate validation
nor C07 down evidence establishes accuracy on post-GeGLU values.

C07 E95's exact frozen source
[e4781ff…](../c07-native-scale-dpe/evidence/e4781ff20e948eacbf1d38ab330361db9821c00b644a9e9b21cbd446bbd9b584.rs)
uses two E4 terms, broadcast-before-quantization at width3840, H-half cluster
ownership, and F32 blockpartials. Its comment claiming BF16 partial rounding is
stale: the actual tensor is F32 and has no BF16 cast. E133 instead quantizes the
M03 L-half before broadcast with three E5 terms and restores cluster-specific
scales. E95's327093-cycle basic screen is positive but incompletely validated,
not a rejected down-arithmetic family and not transferable correctness evidence.

If the complete standard screen is positive, the next oracle must call the
actual candidate down helper from BF16 inputs and compare against independent
original-BF16 truth, not its own E5 reconstruction. Include different cluster
maxima, all-zero input, term underflow, cancellation across blocks/columns and
clusters, and L16/1920/7680 boundaries with varying FP4 codes and blockscales.
Pair a reconstruction audit with original truth to distinguish quantization
loss from mapping/reduction defects. Standard-positive repeats and extra6 remain
separate gates; a negative whole screen stops parameter tuning and does not by
itself establish a mathematical failure or ban a specific supported repair.
