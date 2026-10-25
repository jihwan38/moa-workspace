This is a full-P512 integer packing and public raw-buffer ABI experiment. It
changes neither quantization nor a serving caller. The input is ordinary
canonical I32 codes in [-8,7]; the output owns dense packed bytes in the exact
P512H960/P512L960 order needed by the planned per960 I4 consumers.

The preregistered resource gate is **not fully passed**:

| Shape | Complete estimated cycles | Frozen cap | Result |
|---|---:|---:|---|
| P512×H3840 | 52,085 | 50,000 | HOLD |
| P512×L15360 | 180,627 | 200,000 | PASS |

These are static compiler timelines, not measured latency. Neither schedule has
DramStack. Both perform the entire shape in one kernel; no P128 tiling or extra
launch is hidden. Source/probe typecheck passed in a004 and, after the coverage repair, a005. No standalone ELF,
device execution, quantizer/GEMM integration, or model validation was performed
at that initial resource checkpoint.

The hidden schedule allocates the complete 31,457,280-byte I32 tensor in SRAM
pages 0 and 1. Its 3,932,160-byte packed result occupies page 2 and coexists with
the input during packing: the peak live SRAM payload union is 35,389,440 bytes.
The input shape peaks at 8,847,360 bytes. The compiler also allocates 16,384 bytes
of VRF for the eight shifts broadcast to all 512 slices. Reshape aliases are
counted once, as address unions; they are not extra copies.

The actual canonical-DM → blocked-HBM stores are charged: 36,504 estimated cycles
for H and 142,715 for L. Canonical input loads cost 7,030 and 26,475 respectively;
packing TU costs 1,240 and 4,120. The common 32-byte shift-table HBM load costs
5,708. These are individual operation spans, not additive wall-time claims.

The initial source 7a43bf6b failed before scheduling: reducing the named K axis
also removed retained K/8 time, so the subsequent transpose had a mapping
mismatch. The original failure and source binding remain in a001. Source
9f9a2762 repairs only that issue with distinct Word and Nibble axes. The SDK's
consuming DM reshape is explicitly unsafe because equal sizes alone cannot
prove physical order. Here the preconditions are:

* Chip, Cluster=P/256, Slice=P%256, dtype I32, and padding are unchanged.
* Local k becomes (word=k/8, nibble=k%8), whose linear offset is 8×word+nibble=k.
* The eight shift records retain their offsets 0..7 when renamed Nibble.
* After reducing Nibble, pairwise transpose/trim emits word=2×pair+lane,
  lane∈{0,1}; Word→K/8 leaves each local word offset unchanged.
* These consumed same-wire regroupings do not implement the blocked layout.
  The subsequent real DMA writes word offset
  `(((p/256)*(K/960)+k/960)*256+p%256)*120+(k%960)/8`.
  Nibble lane is k%8, even channels in the low nibble.

For the legacy UP ABI `[P/256,H/960,P%256,I4K960]`, I4K960 is exactly k%960 in
this last formula. DOWN uses the identical L960 rule. The separate owning I32
buffer therefore has the same bytes as those typed I4 inputs; it is not a tensor
dtype cast. The checked adapter binds that owner through public DeviceSend /
Buffers / backend Function::run, verifies exact whole-owner sizes, 256-byte
alignment and non-overlap, and retains the borrowed owners through await.

`src/bin/test_int4_pack_p512.rs` contains a future byte gate: independent
pair-nibble and inverse-index eight-field encoders, all 16 signed top nibbles,
endpoints/zero, explicit nine-bit identities for all 512 rows in every K960 block,
a true row/channel permutation, candidate-first raw binding versus
typed I4 readers, and complete input/table/output immutability. It verifies the
blocked-byte ABI only; it does not establish a GEMM or quantization result.

Reproduce the static report with
`python3 experiments/int4-rational-pack-p512/analyze_resource.py`.
`resource-result.json` binds both frozen schedules and compiler allocations;
`provenance.json` binds source/config, copied run records/logs, and compiled
resource artifacts. Original Cargo.toml, Cargo.lock and toolchain are retained.
The first failed variant and input-cap miss are not relaxed or erased.

Peer review found that the original fixture repeated token rows modulo16. Probe-only
commit c2746fda replaces that fixture and asserts every full row identity; no
device result ever used the inadequate fixture. Kernel/resource source 9f9a2762
and both frozen schedules are unchanged.

After the parent LUT device gates passed, the root authorized one bounded actual
pack/layout diagnostic while retaining the original input-resource HOLD.
`actual-probe-plan.json` preregisters ten cases, 49,152,000 integer codes,
24,576,000 packed bytes, six read-only invalid-binding checks, separate completed
H/L call timings, and one root-only run capped at 70 seconds. This is not a
quantizer/GEMM/model integration or a runtime gain claim.

Source fbab0a3d adds compile-time public-trait assertions before the consumed DM
reshapes and timing/owner checks to the existing probe. The root independently
reviewed it before a007 built the standalone executable using a new empty public
`FURIOSA_OPT_OUT_DIR`. A006 source check and a007 build passed; all 401 recorded
source/config hashes match. The captured ELF is
`runs/a007-rational-pack-p512-actual-build/built/target/release/test_int4_pack_p512`,
SHA256 `cd870cffa309a89bee202ef24e9821a96ef79fdeb97f7f386e8eb3a8369bf1b2`.

The read-only ELF registry audit binds all four embedded kernels to the fresh
output files. Both pack kernel blobs are byte-identical to their earlier
resource-compiled blobs, so the const assertions did not add device work.
`actual-build.json` and `actual-build-registry.json` preserve those bindings.
No execution result is claimed by this build evidence.

Root's single actual run **322961** subsequently passed in 1.433910345 seconds.
All ten fixtures checked 49,152,000 codes and 24,576,000 packed bytes, including
all signed top nibbles, full row/shard identity, a true permutation, typed/raw
reader equality, post-reuse immutability and six invalid-owner rejections.

| Complete awaited call | H3840, µs | L15360, µs |
|---|---:|---:|
| Pack only, range | 48.760–50.206 | 146.996–148.059 |
| Pack only, mean | 49.407 | 147.752 |
| Pack + raw I4 layout copy, range | 96.992–99.432 | 186.125–186.543 |

Each range contains four samples of 16 calls, from the original and permuted
fixtures in ABBA order. Setup, host oracles, uploads and downloads are excluded.
The raw consumer here is a layout copy, **not a GEMM**. Activation quantization,
GeGLU, the complete FFN and model behavior were not measured. The original input
resource screen remains **HOLD52,085>50,000**; the actual numerical pass and
cost measurements do not relabel that screen or establish a serving gain.

The collected log contains **zero** `NPU_SPAN` or profile-marker records and the
Arena records list is empty. Consequently, this run provides no per-operation
actual DMA/TU attribution. The earlier store spans remain static estimates only;
no profile rerun was performed. Reproduce all exact-count, timing and artifact
checks with `python3 experiments/int4-rational-pack-p512/analyze_actual.py`.
The copied raw files are under `actual-322961/`, and the deterministic report is
`actual-result-322961.json`.
