# M08 pair-preserving four-way O source review

Source deliverable: `/tmp/moa-o-pair4-projection-v1.rs`.

- Origin: E122 `c3553ab87046fb4d59c10149a82517f8440bba27`, `furiosa-opt-gemma4-12B/src/device/sliding/projection.rs`.
- Origin SHA256: `380e57d6502e26086bd6d933ce762f4ad5de18587d52a8ef4aac7fa87e3b9c9f`.
- Candidate SHA256: `ae4f6aba8e50dcb6e3118a5a9b25ba105e4dd63a9087148bd716f568232d78bd`.
- Only `project_output` and its O-only `OutputColumns` alias differ. All QKV helpers, channel-scale helper and residual helper are byte-identical. Caller RMS code is not modified.
- Recreate source and logical assertions with `python3 /tmp/moa-o-pair4-source-build.py`; results are `/tmp/moa-o-pair4-source-review.json`.

## Mapping and arithmetic

`OutputColumns=[H/480%4,H/2%4,Qs/256]` gives physical slice `64*g+16*r+q`. Local weight `[H/8%60,H%2,Qs%256]` gives `h=1920*c+480*g+8*t+2*r+j`, `K=256*q+u`. There remain 120×256 FP8 bytes per slice, two clusters, 256 slices per cluster. Python enumeration verifies 61,440 unique 256B HBM packets cover exactly 15,728,640B and each slice receives 30,720B.

The Sub BF16 activation→TRF sequence is unchanged. Its destination slice type changes only the replicated H ownership. Main FP8 lookup→BF16 contraction still reduces packet K32 and time K/32%8, leaving `[H/8%60,H%2]`. The 16 contiguous q slices then reduce in F32 while retaining `[H/480%4,H/2%4,1#16]`. No new cast, scale, arithmetic approximation, weight copy, partial-output materialization or Switch is introduced at the source level.

Transpose moves only the inner j pair into the packet: Time `[H/8%60]`, Packet `[H%2#8]`, then `commit_trim<[H%2]>` stores two F32 values per commit (8B). This is the same pair packing mechanism as the original O, with a different outer row coordinate. `OutputLocalRows` has 120 entries and no padding. The inherited `Dummy2 → H/1920` reshape changes the coordinate name, preserving the physical order and per-level sizes.

The final `to_dm` still targets canonical `OutputEpilogueSlices=[1#32,H/480]`, local `[H%480]`. Logical source pair byte is `8*t`; canonical destination pair byte is `32*t+8*r`. Both and all strides are multiples of8. Python verifies exactly 3,840 unique F32 output rows and15,360B coverage. Channel scaling remains F32→BF16 at the original boundary; downstream RMS and residual are unchanged.

## SDK/API review and limits

Manual installed SDK0.8.1 review found no obvious public API/type contract mismatch:

- `engine/contraction/outer/mod.rs:124`: Main stream BF16 and BF16 TRF satisfy the same supported operand pairing as baseline.
- `engine/contraction/time.rs:21` plus lower `verify/engine/contraction/time.rs:43`: retained H factors preserve their input order and padding; only the same inner K factor is removed.
- `engine/contraction/lane.rs:50`: lane mapping remains one valid lane padded to8 and Interleaved.
- `engine/vector/tensor/vector_tensor.rs`: same F32 inter-slice reduce operation, same ratio16 intended; only retained H factorization changes.
- `engine/transpose.rs:68`, `engine/commit_adapter.rs:145`, `engine/commit.rs:27`: inner H%2 pair packing and8B commit preserved.
- `tensor/memory/dma_layout.rs:8–90`: intended gather packet and stride are8B aligned.

No Rust typecheck, compile, host execution or NPU execution was run by this agent. The lowerer delegates transpose/fetch/commit parameter synthesis to a private implementation (`lower/src/lowering/mod.rs:111` onward), so manual shape reasoning does not guarantee successful lowering. Actual producer/consumer descriptors and allocation must be audited after the root's compile.

Required first gate:

1. Weight DMA covers the exact15MiB once, without padding expansion or OOB; DMN order and actual source strides match the intended ownership.
2. Main retains ratio16 F32 reduce and 8B paired commit; no unexpected restore Switch or weight materialization appears.
3. Gather writes canonical H480 once and stays bounded/aligned. Baseline actual descriptor is32×480B; candidate is expected to require1,920×8B. Equal logical bytes and one source-level stage do not imply equal transfer cost.
4. Complete O correctness and contemporary whole-O cycles decide the candidate. An ingress-only result cannot be called a kernel gain.

One clear API/shape error can justify a bounded correction. A compiler's distinct scheduling/packetization choice is evidence to inspect, not permission for an unbounded layout sweep.
