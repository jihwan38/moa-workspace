# E150 down-scale supply: prepared source replacement, not a measured improvement

2026-10-01. Root selected this bounded replacement after the E150 readiness profile (job141326): last native→first reduce median10,244 cycles, scale DMA end→reduce1,281, and scale VRF end→reduce exactly1 cycle in all6 observations. Those spans locate readiness; they do not identify exclusive DMA bus stalls or a reclaimable10,244-cycle saving. No new compilation or NPU job was performed by this preparation.

`prepare.py` reads frozen `ffn-down-weight-stages/staged-v1/mlp.rs` SHA813aa7068376a41833723748ba31bf43008972bb5ff9b4e1e7cbc3d185a23530, writes `mlp.rs`, `source.diff`, `source-contract.json`, and verifies every intended scale coordinate. `--check` does not write. Prepared source SHAe412ef9422a17cba47b94bbda8955cf45be98825e7dd3577efa09cab6d50f893 (40179B). Only scale DM ownership and its existing five Sub VRF producers change. Native halves, weighted quantizer, F32 partials, multiplication/reduction/cast order, up/gate and public functions are unchanged.

## Coordinates and exact target question

HBM remains original E4 `[H3840,L/16=960]`. Cluster c owns scale columns480c..480c+479. Source slice4g+b (g<64,b<4) owns15 rows h=60g+4r+b and a contiguous480-byte scale segment per row. Type:

```rust
DmTensor<f8e4m3, Chip, UpGateCluster,
         m![H / 60, H % 4], m![H / 4 % 15, TileL / 16]>
```

For existing consumer i<5, tile r=3i..3i+2; Fetch Time `[TileL/1920=4,H/4%15=3]`, Packet `[TileL/16%120]`. Cast exactly to F32 and route `InterTranspose{slice1:4,slice0:1,time0:3}`. Desired destination slice4g+q gets Time `[a3,b4]`, meaning rows12i+4a+b and scale columns120q..120q+119. The old VRF type `[H%60=12,TileL/16%120]` and old collect packet8 then feed the unchanged downstream consumer.

This uses the actual E144-v2 low-bit source ownership order (quotient first, modulo second). It does not reuse the incorrect E144-v1 time-order hypothesis. CPU checks cover every source byte exactly once and all30,720 destination row packets. They do not verify target lowering. Exact compile must first resolve the partial named quotient `[H/4%15=3,H%4]` as canonical `[H%60=12]`; accepting some numerically equal shape with changed wire order is not a repair. The480-byte F32 packet must pass the actual Sub Switch lowering and collect constraints; no arbitrary padding is proposed.

## Public SDK support and observed precedent

Installed SDK0.8.1 source, extracted at `/tmp/moa-opt-std-release/furiosa-opt-std-0.8.1`:
- `src/engine/switch.rs:57–86`: generic TU `switch` supports `CanApplySwitch`; Packet is passed unchanged and `verify_switch` invokes target `config_switch` validation.
- `src/engine/mod.rs:290,301,305`: `PositionFetchCast → Switch → Collect → to_vrf` stages are public.
- `src/engine/collect.rs:91–99`: Sub collect-to-VRF directly returns the typed VRF; no extra DM commit is required by this API.
- `moa-dev/experiments/candidates/c08-fused-qkv/c03-columns8/projection.rs:26–36` and84–94 actually used Sub cast→InterTranspose→collect→VRF. E106 passed15 standard comparisons but regressed QKV30.33%. That is target execution precedent for the chain, not evidence for this packet geometry or a performance benefit. E105's separate padded DMA defects must not be inherited as evidence about E106.
- `moa-dev/experiments/candidates/qkv-fullh-weight-switch/fullh-v2/projection.rs:232–233,280–281` uses source row-low ownership with the actual quotient/modulo output order. Its Main packet differs from this Sub F32 packet.

## Costs and prior failures

HBM payload remains3,686,400B, compact DM payload remains3,686,400B, and each of five scale VRFs remains5,760B/slice. Source logical segments become7,680×480B versus30,720×120B. A480B logical segment is not an assertion of one hardware DMA packet: actual descriptor packet size, source pitch/extent and destination row ownership must be counted after compilation.

Added Switch payload is14,745,600B (14.0625MiB) of F32 over the five Sub consumers, plus routing/config setup and any bitmap loads. This is logical stream payload, not measured NoC hops or bus bytes. There is no additional materialized DM scatter, scale matrix packing, or new numerical conversion. Existing F32 cast occurs before routing and is exact for finite E4 values.

Closest rejected paths:
- E45 `ffn-down-scale-coalesced`: representative row-group HBM staging followed by another typed SRAM→SRAM scatter;15PASS but+3.77% FFN versus its historical repeat control. Its exact scale transfer sum increased9,774→11,672. Current proposal replaces that extra DM scatter with the existing Sub→VRF producer and uses current two-cluster K4 native-down ownership. This distinction is concrete but does not prove lower total cost.
- E59 `memory-layout-e55/down-scale-split`:120→128B padded row pitch,−0.24% within repeat variation. This proposal adds no padding and is not that layout.
- E33/C01 scale-as-second-contraction changes arithmetic/storage and has separate failures; it is not this supply-only transformation.
- Bounded repository search found no prior FFN `InterTranspose{4,1,3}` down-scale producer. This is not a claim that every historical artifact has been searched.

## Required first gate and whole-path decision

1. Bind exact E150 source and public interface; verify only the two source regions differ.
2. Enumerate emitted HBM descriptor bounds/coverage (end3,686,400B), source rowlow b, sourcecluster c, and compact row pitch480B; no padded view DMA expansion.
3. For each of five streams inspect input tile base3i×480B and actual Fetch times/strides, Switch output `[a,b]`, and VRF reader row order12i+4a+b. Consumer24..35 must still cross native partial halves correctly.
4. Verify five5760B VRF allocations, immutable96B weighted coefficients and1920B post-RMS gamma against the *whole* schedule and actual overlap. Nominal7776B simultaneous subtotal does not include all potentially live allocations. Check routing bitmap/config bounds too.
5. Prove source/image QKV and O remain E150-identical. Run the normal contemporary before/candidate/after45-comparison screen. A positive frozen candidate needs existing raw-down9 and full-FFN boundary plus repeats/extra6; static schedule alone cannot adopt or reject a whole-path benefit.

The profile identifies a sensible readiness target. Added Switch traffic/setup and altered Sub/VRF scheduling can erase or reverse any shorter HBM ingress; no speedup or ranking prediction is made.
