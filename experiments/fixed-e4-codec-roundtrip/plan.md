# Finite E4 storage codec device gate

Base q022 `98662213`. The authorized compressed storage keeps q022 numerical
values and must preserve every finite E4 byte through read, BF16 arithmetic
representation, and commit recoding. Existing producer gates establish rounded
BF16 values; this gate additionally checks raw E4 output bytes.

Upload all 254 finite codes in row-dependent permutations across B16, R64 and
P128 head layouts. Decode with the public Main E4-to-BF16 lookup, recode with the
q022 BF16-to-F32-to-E4 cast, and compare both complete outputs with the independent
integer CPU codec. Preserve signed zero and both finite saturation endpoints.
Verify the uploaded input remains byte-identical. A separate B16 reader-only
case includes both NaN codes: require NaNs at their own coordinates and exact
finite neighbors; record actual NaN payloads without requiring an unproven
payload convention. No model, quality, cache-offset or throughput claim.

Root owns the r64-server Cargo target. Use a new empty FURIOSA_OPT_OUT_DIR,
fresh-package source verification and immediate captured executable. No serving
code or existing numerical helper is modified.
