The explicit adjacent-replica LUT prototype passes source review, plain hostcheck and both exact resource compiles. There is no standalone ELF, device execution, measured gain or promotion. Source is `78054937`, preregistered plan `219e0152`, base `8afe741d`; the original retained-axis NO-GO and actual323173 evidence remain unchanged.

The table is I32 `[Mantissa128,Replica16,Record8]` (65,536 bytes), with both leading axes explicitly indexed and full32-byte records as payload. The relative byte offset is `512*m+32*(p%16)`, from0 through65,504; the final record ends exactly at65,536. This is a new offset-content ABI with unchanged grouped scratch extents. No retained-axis zip or fabricated SPM/device address is used.

A separate, fully initialized512-byte HBM owner `[P512%16,Record8]` contains `32*r` in lane0 and nonzero canaries in the other lanes. Public `to_dm` broadcasts real named coordinates onto existing grouped slices, then a typed lane0 view feeds VRF and AddFxp in the index pass. H maps `p=256*c+4*r+j`, so `p%16=4*(r%4)+j`; L maps `p=256*c+16*r+j`, so `p%16=j`. The independent coordinate proof enumerates every512×4/16 slot and all128 mantissas. Actual DMA values still require the standalone oracle.

The original amax/floor/p20 multiplier/rounding/scales/packing/H16 staging/L store bodies are byte-identical after reversing only the recorded addressing delta. Original quotient/scale/packing fixtures remain untouched. The new source probe uses independent poisoned old/new outputs, all five fixtures per width (including two true permutations), candidate-first verification and completed old/new ABBA. Codes and scale bits must match the original control exactly; offsets are checked against the new physical oracle. A permuted input moves its mantissa with the source data while its replica remains the destination row's `p%16`. All output checks repeat after timing, and all input/table/base/shift bytes are reread. This probe has only typechecked.

| Static complete graph | Original grouped-store | Replicated LUT |
|---|---:|---:|
| H3840 timeline cycles |29,069|29,115|
| L15360 timeline cycles |67,271|67,239|
| H placed live SRAM address-union bytes |11,796,480|11,812,864|
| L placed live SRAM address-union bytes |47,185,920|47,202,304|

Both graphs have no DramStack. Added replica-base HBM→DM costs872/1,739 estimated cycles for H/L; added DM→VRF costs279/327. Each graph gains one input DMA and one VRF transfer. Existing8 TU instances and their summed spans are unchanged; extra shift/Add are inside the original index TU. The explicit base path materializes65,536/262,144 whole-chip DM and VRF bytes for H/L. Its lifetimes and all aliases are in the replay JSON. SRAM union is a lower-bound accounting over placed addresses, not a proof that unplaced aliases require no storage; the full records and conservative live-record sum are retained.

The LUT gather estimates remain7,886/28,521 cycles. The compiler therefore provides no evidence that this address distribution speeds up the lookup. Tiny endpoint differences (+46/−32) include overlap and are not measured savings. Original offset stores and reads remain charged:8,192/32,768-byte owners, twice those bytes written+reread; full-record gathers deliver65,536/262,144 logical bytes. New retained HBM is65,536+512 versus the original4,096-byte table, adding61,952 bytes. Host table/base construction/upload is explicit startup work outside a future matched kernel timer.

Actual323173 case2 used `A=14*2^shift`, making every original offset3,072; the candidate will distribute those requests across offsets49,152 through49,632. The observed slow fixture correlated with this address pattern but emitted no operation spans. Contention is still only a hypothesis, and neither this resource result nor a future micro gain may be added to the separate seven-to-three complete-FFN result.

Read-only replay:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 experiments/int4-grouped-lut-replicas/audit_source.py > /tmp/replica-source.json
cmp /tmp/replica-source.json experiments/int4-grouped-lut-replicas/source-proof.json
PYTHONDONTWRITEBYTECODE=1 python3 experiments/int4-grouped-lut-replicas/analyze_resources.py > /tmp/replica-resources.json
cmp /tmp/replica-resources.json experiments/int4-grouped-lut-replicas/resource-result.json
```

[runs/](runs/) preserves commands, raw logs, frozen source manifests and fresh-package records. [sdk-contract.json](sdk-contract.json) contains exact installed source hashes/line excerpts, including correction of the initial plan's `mapped.rs` shorthand to actual `tensor/memory.rs:332–348`. [source-review.json](source-review.json) records the independent read-only PASS. Original Cargo.toml, Cargo.lock and toolchain remain unchanged; both resource builds used separately created empty kernel output directories and the exclusively owned tail256 target. Root alone may authorize an ELF or Arena gate.
