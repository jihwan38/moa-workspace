# Complete approximate per960 F4 weight micro

Base8df08ec3. Candidate and original current-E4 complete post-attention/norm/FFN/
postnorm calls retain all eight scratch outputs. Intended exact reference is a
separate dense E4 expansion of the SAME unscaled F4 codes with the SAME F32
per960 partial-scale placement. No scale is folded into E4. Only candidate's
fixed table lookup differs from this reference; all eight outputs must match.
Original E4 stage differences are measured, not treated as correctness failures.

Provided layers0/5/6/47, two actual row permutations each, two feedback steps,
separately poisoned stage buffers, candidate first. At a feedback step all three
arms start from the same previous candidate state, so reported original error
is a same-input layer comparison. Per-row permutation must match every stage.
All current and quantized weight/scale bytes are checked immutable. The CPU
static helper independently encodes all weights and defines zero/ties/signs.

Independent F64 checks sample local scaled UP/GATE/DOWN partials and final DOWN
projection, preserving explicit BF16 row/global boundaries. Partial tolerance
1e-5*sum(abs(products))*abs(group_scale)+1e-5 accounts for F32 contraction;
final tolerance includes one BF16 ULP plus the same accumulated absolute-error
bound. These are numerical reference checks, not CPU-bit parity claims.

Inclusive ABBA compares original current one-launch chain vs candidate one-launch
chain on identical reuploaded projected/residual inputs, completed before stop.
All scale/DMA/lookup/materialized-F32-partial cost is included. Upload, CPU
encoding, immutable checks and readback stay outside timing. Predeclared >=10%
actual complete-chain gain required before recommending any whole-model work.
No server/Worker/Model changes or floating activation quantizer change.

CLI case ranges0..2 and2..4 allow bounded70s jobs if full weight quantization and
readback are expensive. Root alone dispatches Arena. Old exact provided-per16
reconstruction and coarse240 holds remain separate.
