# Preregistered retained-HBM three-graph diagnostic

Base d21224649eef82f5c3bb293aac44b2d1c28b58e8 is the actual323213 complete
seven-call checkpoint. Its intended-code correctness passed; both >=10% gates
failed (1089.835/1086.606 us versus current q0201067.0648125/1064.23625 us).
No performance or model-quality conclusion follows from a reduced launch count.

A: prefix_sliding_bf16 -> quantize_pack_input. Original residual, canonical
prefix BF16, packed H, group scales and relative LUT offset HBM owners remain.
B: typed I4 project_upgate -> geglu_bf16 -> quantize_pack_hidden. The packed H
I32 owner binds to this typed I4 input only at the public runtime boundary.
Original joined BF16, canonical hidden BF16, packed L, scales and offsets remain.
C: typed I4 project_down -> original post_feedforward. Packed L I32 binds only
at this launch boundary. Original projection/global BF16 cuts and residual tail
remain. Every helper is a plain typed function with byte-identical donor body;
existing standalone device wrappers delegate to those helpers.

B/C raw binding validates every slot's exact checked byte extent, 256-byte start
alignment and checked end before function lookup. Every mutable owner is checked
against all immutable inputs and all other mutable owners. Rust borrows and
public Buffer ownership clones live until the awaited run completes. No address
construction, copies or I32/I4 alias formals inside a graph are permitted.

The new diagnostic retains layer0 supplied weights, both original row orders,
two feedback updates, unchanged independent per960 quotient/packing/F64 reference,
all11 snapshot fields and immutable/reuse checks. It compares the three-call
candidate with a separate seven-call scratch on each step, including prefix and
mid-residual after A, then all11 final fields. A normal typed launch arm embeds
B/C images and must equal the raw arm. Negative binding tests fail before lookup.
Current q020 single-call graph remains the production timing control. Both-order
>=10% completed-chain gain versus q020 remains mandatory. Seven-vs-three timing
is separate diagnostic evidence. All restores/downloads/oracles remain untimed;
all three awaited runtime calls and all HBM boundaries/metadata are timed.

Resource gates report whole timeline endpoints, SPM/SRAM/TRF extents and peak
liveness, noDramStack, operator counts and immutable standalone-image parity.
Compare A/B/C with the corresponding standalone endpoint sums only as compiler
estimates. Inspect prefetch/lifetime overlap separately; do not infer that four
waits equal the required128.79-129.48 us. Preserve concrete failures and stop if
resource/lowering cannot support this exact contract without a new mechanism.

No standalone ELF, Arena, model integration or production activation is authorized.
