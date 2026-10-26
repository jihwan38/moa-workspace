Actual323102 completed successfully in25.831942403s. All required current-E4
anchors passed, and the original323067 matched step0 drift was reproduced.
On these synthetic layer0 inputs, activation quantization causes substantially
more projection/GeGLU drift than weight re-encoding alone. This is a local
attribution result, not a model-quality pass or evidence for a specific repair.

Frozen executable:6,878,176 bytes, SHA256
`9a50b33d754f4eeeb85af7c3ea917e35f1ff2400f9aaf93152fd7cb9df8f1551`.
Build and actual source commit466d2800; all413 input hashes match, including all
411 original complete-probe inputs. The19 original device images are identical;
standalone input E4 quantization also matches q020. The newly bound existing
hidden E4 quantizer passed every production-hidden code/scale comparison.
Raw log SHA256:
`b67a50815b4773ade0d40c157c21c40d861f626e6fbbee290db406e181af0106`.

Exact checks, reconciled from the source loops:

| Check | Count |
|---|---:|
| BF16 comparison events |204,472,320|
| Independent integer/F64 samples |4,096|
| Permutation values |167,117,824|
| Direct immutable-byte comparisons |878,180,488|
| Replay/post-reuse bytes |792,346,624|
| Native weight E4 codes independently checked |176,947,200|
| Native weight row-scale bits independently checked |34,560|
| E4 activation anchor codes, both row orders |19,660,800|
| E4 activation anchor scale bits |2,048|

EE UP/GATE, GeGLU, DOWN and tail each have zero differences in both row orders.
Thus the covered scale/layout/BF16 boundaries reproduce the current production
path, including the reference's added identity scaling/F32 DM boundary. The II
arm also matches the previously independently checked intended I4 path. These
facts rule out those tested mismatches as an explanation for this fixture;
they do not prove equivalence for all inputs.
The4,096 integer/F64 samples cover intended integer-code arms; they are not a
general F64 oracle for mixed E4 factor projections.

All-row reconstruction against the same BF16 values, original row order:

| Fixed input | E4 relativeL2 | A4 relativeL2 | E4 zero fraction | A4 zero fraction |
|---|---:|---:|---:|---:|
| Pre-FFN prefix,512×3840 |.026252|.655033|.022116|.921493|
| Production GeGLU,512×15360 |.026616|.504361|.182193|.933448|

The original BF16 inputs have only2,177/1,966,080 and609,315/7,864,320 zeros.
Every row's error/scale and all channel maxima are retained in the raw JSON
records; the second order follows `(73*p+127)%512` exactly. Tiny differences in
aggregate F64 SSE arise from accumulation order, not different row results.

Provided-BF16 weight reconstruction error:

| Projection | Current E4 relativeL2 | Direct per960 I4 relativeL2 |
|---|---:|---:|
| UP |.021322|.203050|
| GATE |.021649|.224123|
| DOWN |.022997|.216505|

The native code/row-scale oracle reconstructs the provided F4/per16-E4 weights
before globals and independently checks the production E4 preprocessing. The
I4 weights are encoded directly from those provided BF16 values, with no extra
native row-dequant factor. Full row errors and every provided-column min/max
are retained. These reconstruction metrics use RN32(code×scale), followed by
F64 error accumulation; they are not exact dot-product error decompositions.

Matched-input factors, relativeL2 against required exact EE:

| Output | Activation only IE | Weight only EI | Both II |
|---|---:|---:|---:|
| UP |.465667|.130060|.483041|
| GATE |.767540|.246710|.800923|
| GeGLU |.816305|.275624|.846778|
| DOWN on fixed production hidden |.346620|.115473|.365646|
| Tail on fixed production hidden/midpoint |.330883|.103249|.348104|

The DOWN table uses production GeGLU BF16 for every arm; it intentionally
excludes the upstream hidden drift. The full candidate remains joinedL2
.574482206, DOWN.792839655 and final residual.846608646, exactly reproducing
323067 step0. Nonzero II−IE−EI+EE interactions are retained in JSON; L2 effects
must not be added. No divergent-feedback step is used for this attribution.

Completed-await wall time, all eight candidate samples included:

| Candidate stage | Mean µs |
|---|---:|
| Prefix norm/residual/BF16 output |83.960|
| Input rational quantize+pack |49.104|
| Checked raw UP/GATE |384.267|
| Global/GeGLU/canonical BF16 bridge |188.274|
| Hidden rational quantize+pack |83.483|
| Checked raw DOWN |273.998|
| Postnorm/residual/layer gate |52.933|
| Clock gaps between measured spans |.476|
| Enclosing candidate total |1116.495|
| Original one-call production control |1076.908|

Each order has four restored executions per arm in two-sample ABBA blocks.
Candidate versus current production means are1123.307 vs1077.068µs for order0
(4.2930% slower), and1109.683 vs1076.748µs for order1 (3.0587% slower).
There are no downloads or intermediate validation inside the timed path; raw
binding validation is included. These are clock-instrumented completed calls,
not a device-op trace or a decomposition into host waits versus DMA/compute.
`arena-records.json` is empty. The prior complete performance HOLD remains.

Activation-range conditioning and whole-path cost analysis are motivated as
possible next investigations, not validated solutions. No balancing, clipping,
calibration, new model, official submission or production promotion occurred.

Replay: `python3 experiments/int4-drift-attribution/analyze_actual.py
--verify-local-artifacts`. It verifies raw hashes, source/artifact/image lineage,
all counters and negative controls, row permutations and all16 timing records.
The replayed JSON is `actual-result.json`. round2_path independently confirmed
the same hashes, counts, factor anchors and timing means.
