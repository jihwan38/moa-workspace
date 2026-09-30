# E147 scalar supply: selected complete path, not a measured improvement

E147 starts at `2bdbe9137b999176dabf510cfe517267cbea7298` in branch `opt/qkv-scalar-sub-vrf`. Source and performance control are adopted E144 (`baseline-e144`, source commit `3586c417e43f4ea3afc7df7c428d381b7dcbcc5e`). Unadopted E146 Q/K staging is excluded: its first positive did not reproduce, with three-pair median +0.406876%. E146's passing raw9/full2 observations do not establish E147 correctness.

The selected path combines one Main scale producer with Sub group4 scalar/coefficient broadcast directly into VRF. It preserves four E4 terms, factors `[1,16,2^22,2^27]`, signed residual RNE arithmetic, full-H E144 weight supply, shared activation TRF, Term4 then K4 reduction, common scale then channel scale, first BF16 boundary and head/RoPE/cache operations. This note records inspected source/API support and the evidence motivating an experiment. No E147 compile, NPU result or gain is claimed here.

## Current producer, transfer and consumer

Frozen source: `moa-dev/experiments/candidates/qkv-fullh-weight-switch/fullh-v2/projection.rs`, SHA256 `3d2500295785592bebdc723fb62617398e2e001ba41c63b2a5c078cda5e00fed`. Source `prepare_qkv_terms` uses local H960 absmax/floor, K4 max, and exponent-mask/power2 scale as three Main streams. Scale is written only to q0 of each Group64; a typed DMA copies its padded scalar packet to all four K owners before a Sub load supplies the quantizer VRF. Four Main streams construct normalized coefficients; Sub packs four scalar time slots into16B, another typed DMA distributes them, then Sub loads the coefficient VRF. The full allocation of the leader scale is separately relabelled to anonymous64 replicas and copied to canonical head slices for post-native scaling.

Current exact schedule, `.../evidence/e144-fullh-v2/schedule.json`, uses `DmaEngine` for V weight122, quant-scale172, coefficient181 and Q weight186. Quant-scale172 reads only tensor97 and writes111; Sub180 reads111 and produces scale VRF112; term0 Main185 reads112. Q weight186 reads only HBM tensor12 and produces weight115: there is no scalar-value SSA edge into Q weight. The static schedule places V weight→scalar DMA→coefficient DMA→Q weight on this resource; its intervals are not a physical arbitration trace.

Actual D145 `moa-dev/experiments/candidates/e144-qkv-readiness/profile-analysis.json`, SHA256 `016402b8d0f5cebd7fd3b5dc2304a69fb860794b6676dfd1e9d09148723e75ef`, narrows the hypothesis. Across six cluster observations:

- Q weight starts exactly7 cycles after quant-scale VRF becomes ready; it starts350/352 cycles after scalar DMA ends.
- Q weight starts760–766 cycles **before** coefficient DMA ends. Scalar DMA itself starts4243–6524 cycles before V weight ends. Marker envelopes therefore overlap; a claim that these complete DMA operations strictly serialize on hardware would be false.
- Scale-ready→quant-scale-VRF-ready paired median is9914 cycles. Term0 begins one cycle after that VRF; V native begins one cycle after the shared activation TRF.
- Q/K native begin two cycles after their own weights become ready. Advancing only V quantization cannot by itself prove earlier final completion.

This supports testing removal of scalar distribution from the shared DMA scheduling path. It does not identify a pure bus-stall fraction, prove scalar DMA alone causes Q's issue time, or justify subtracting9914 cycles from whole runtime. Sub routing/bitmap work, different register schedules and the unchanged long weight streams may dominate the replacement.

## One Main scale producer: exact API order and numerical boundary

Proposed chain over the already-present BF16 H960 consumer shard:

`fetch BF16→cast F32→Tag→reinterpret i32→BitAnd(0x7f800000)→reinterpret F32→Narrow4→Mul0(1/128)→local Max(H960)→Widen1#8→ClipMax(2^-126)→K4 Max→leader scale DM`.

For finite BF16 inputs this computes the same scalar as `2^floor(log2(max(abs(x),2^-119)))/128`. The mask removes sign and mantissa together; max commutes with this monotone exponent operation. Applying the minimum after local reduction retains the floor for zero and tiny inputs. It does not change the four residual factors or normalization division. `moa-dev/experiments/candidates/qkv-scalar-sub-vrf/numerical/scale-proof.py` and its result separately bind the numerical audit; NPU checks remain required.

Installed SDK0.8.1 primary source is in container `moa2026-dev`, registry directory `/home/moa/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/`. `furiosa-opt-std-0.8.1/src/engine/vector/stage/markers.rs` defines the order `Logic→Narrow→Fp→IntraSliceReduce→Widen→Clip`, and explicitly permits Clip→InterSliceReduce. `engine/vector/tensor/vector_tensor.rs:502–524` exposes the F32 IntraFirst/Way8 inter-slice reduction. **Clip before local Max is not a legal stage order**; the selected placement is after local Max/Widen and before K4 Max. InterFirst K4Max→Tag→mask/Mul also has a public route (`vector_tensor.rs:649–677`), but only merging those last two streams would retain the local-max DM boundary.

The chosen complete fusion adds a multiply on all960 values in each of512 consumer slices:491520 value multiplications rather than128 leader-scalar multiplications. It removes two Main task boundaries and the local-max/global-max DM intermediates. Existing scan/reduce work remains. Pipeline overlap may absorb some work, but no exact lowering or cycle saving follows from the typestate alone. Verify actual Fp/Clip/reducer order, initialized q0 result and group-preserving K4 ownership in the emitted graph.

## Sub broadcast→VRF: exact layout, packets and prior support

Keep the four existing coefficient-value producers and compact pack initially. Each of128 `(cluster,group)` leaders has the same scale and correctly ordered coefficient vector. Scale and coefficient Sub streams use the same physical source owner `4*g`; all destination slices `4*g+q`, q=0..3, receive that leader's value in the same cluster.

Use `CustomBroadcast { ring_size:4 }` with input Slice `[QkvGroup,1#4]` and **anonymous** output Slice `[QkvGroup,4]`. Collect into VRF, then publicly relabel the already-copied Slice axis as `[QkvGroup,H/960]`. Cluster `m![2]`/`Dummy2` relabelling is permitted only with identical physical ownership and actual duplicate values. Keep the existing independent canonical-head scale DMA; direct quantizer VRF does not replace its consumer contract.

- Scale: physical input packet32B (`F32[1#8]`), only the first4B is meaningful. Sub fetch Time1/Packet1#8, Switch, Collect Time1/Packet1#8, to_vrf.
- Coefficients: compact input packet16B (`F32[Term4]`), all four cells meaningful. Sub fetch Time1/Term4, Switch, Collect Time1/Term4#8 pads to32B VRF. It must not fetch32B from the compact16B allocation. Padded upper VRF cells are not coefficients and are not consumed by narrowed Term4 arithmetic.

Actual E144 graph `.../evidence/e144-fullh-v2/graph.json` has scalar `DmaCommand#O15` (graph key94) and coefficient `DmaCommand#O74` (key96). Each has8 PE descriptors × `(4×8×2)` iterations =512 packets. Scalar descriptor ends `None=1:32`; coefficient ends `None=1:16`. Source replica stride0 rereads the same leader; destination stride4194304 advances the physical slice.

| Quantity across two clusters | Scalar | Four coefficients |
|---|---:|---:|
| Meaningful values at128 leaders | 512B | 2048B |
| Unique physical leader packet regions | 4096B | 2048B |
| Useful values after512-slice replication | 2048B | 8192B |
| Encoded DMA read/write payload, each side | 16384B | 8192B |
| Resulting VRF packet storage | 16384B | 16384B |

The graph's leader DM allocation sizes additionally include invalid padded q slots; they must not be described as initialized values. Encoded payload counts do not establish physical bus transactions or bandwidth. The replacement removes two typed-DMA operations and replicated DM intermediates but retains VRF packet writes; it adds Sub Switch routing and bitmap/control dependencies. The retained coefficient producers, packet pack and head-scale transfer are not free or removed.

The public `engine/switch.rs:57–86` is Main/Sub-generic and preserves Packet; `engine/collect.rs:89–99` supports Sub Collect→VRF. `tensor/memory.rs:1488–1527` explicitly permits unsafe `VrfTensor::reshape` to name a broadcast Chip/Cluster/Slice axis whose physical units already hold copies. It is a relabel, not a transfer, and must preserve per-level size/wire ownership. Naming an Element broadcast would invent data and is not this use.

C04 `c04-down-transfer/four-ring16-sub/mlp.rs:383–401` is a measured Sub CustomBroadcast precedent: it commits anonymous copies to **DM**, then relabels DM and loads TRF. E80 passed accuracy but regressed in repeated timing. C03 `c08-fused-qkv/c03-columns8/projection.rs` demonstrates Sub **InterTranspose**→VRF, without a CustomBroadcast bitmap. Neither is direct evidence that the new combined route already lowered. E55 `qkv-bf16-broadcast-trf-e55` failed a named-output bitmap at SubSwitch→TRF and lacked a public TRF relabel escape; public VRF reshape is the concrete difference here. Do not generalize E55 into a SubSwitch prohibition or call E147 exact support established before compilation.

## Gates and decision scope

The exact build must preserve E144 full-H Q/K/V ingress and unchanged whole O/FFN images; validate all256 group4 bitmap entries, fetch32/16B bounds, VRF element order and complete lifetime coverage, canonical head-scale DMA/source equality and the one-Main numerical order. Existing raw9/full2 inputs and references can be reused with the new whole-source/image binding; their old results cannot. Three contemporary performance allocations and standard extra6 are separate from boundary accuracy. A modest reproducible gain still qualifies for adoption; a failed lowering needs a specific source/API diagnosis, not a parameter sweep.

## Same-turn exact-build follow-up

The selected `e147-scalar-v1` subsequently passed its first exact compile (static48436). `scalar-v1/projection.rs` is39440B, SHA256 `e21e207081935b441530e78c56ffbc1731f484c68844fc692ee283815a82f3cd`; whole QKV is219904B, SHA256 `fd2e252603b4de38389d8720edd78d6ae1b5c82514f6869934a5ebbb46c5f5fb`. The build is `moa-dev/experiments/candidates/qkv-scalar-sub-vrf/build-e147-scalar-v1-repro.json`. Actual graph inspection confirms the one-Main arithmetic/reduction order, two Sub→VRF CustomBroadcast operations, and one shared bitmap load. In the actual graph, `RenegadeCommand#O14` and `#O70` both use tensor34 and their256 bitmap entries each equal `4*(slice//4)`. Thus the historical API/prior-support limitations above describe selection-time evidence; this specific route now has exact lowering evidence. The final whole graph gate, changed-boundary NPU checks and performance remain separate pending gates at this follow-up. Root checked unchanged native/head/O source; no E144 timing or correctness result is inherited.

The new shared bitmap is not a zero-cost constant: actual `DmaCommand#O13` has8 PE descriptors ×32×2 packets ×32B =16384B encoded DRAM→SRAM payload, with input33/output34 covering2×256×32B. Both Sub operations reuse this one load. This newly added16KiB control transfer must be recorded alongside the removed24KiB scalar/coefficient StoS payload; their cycle costs are not interchangeable.

The independent compiled audit and thin `graph-verification-e147-scalar-v1.json` subsequently passed and were frozen. Boundary production selection now binds the actual46 source inputs and whole image; a new raw9 diagnostic/full2 host build is authorized. No boundary NPU observation is implied by that gate.

Changed-boundary follow-up completed: new raw9 job137109 passed73728 exact BF16 outputs and full-production-image full2 job137116 passed16384 outputs plus8380416 untouched cache values. Original references and actual uploaded packages were verified; see `moa-dev/experiments/candidates/qkv-scalar-sub-vrf/boundary-suite/result-summary.json`. These finite observations do not establish generic BF16 correctness or a performance gain; repeat/standard validation and adoption are separate.
