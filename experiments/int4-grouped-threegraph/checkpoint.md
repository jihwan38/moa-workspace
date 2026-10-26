Historical resource checkpoint below: subsequently root authorized fresh ELF
build a012 after source reviews passed. See [build-result.md](build-result.md)
and [a012-image-audit.json](a012-image-audit.json). Actual execution remains pending.

# Source and resource checkpoint; actual execution pending

Functional source **d5e70b98**, preregistration **3a051632**, base **d2122464**.
The bounded current work is stopped after its already-running resource checks.
No standalone executable, Arena run, serving change or model promotion occurred.
Original q020 official result and all failed seven-call evidence remain unchanged.

## Completed gates

- Host check a001 passed11.16s with original Cargo.toml and frozen SDK0.9 lock.
- Six device bodies were extracted byte-for-byte to plain typed helpers; existing
  standalone wrappers delegate with identical signatures/arguments. The original
  seven-call probe, independent oracle and q020 control sources are unchanged.
- Batch independent source extraction/order review passed for d5e70b98.
- A/B/C exact resource compilation passed. No DramStack allocations were found.
- All seven same-source standalone images are byte-identical to the images in
  actual323213 ELF3444ef561e2d39d6a15ec6340548d07e65b873d2884b62ac04f9ea2ccd66cdd4.

| Graph | Composition | Sum separate endpoints | Composed endpoint |
|---|---|---:|---:|
| A | BF16 prefix + H quant/pack |71249|69985|
| B | I4 UP/GATE + BF16 GeGLU + L quant/pack |534126|520660|
| C | I4 DOWN + original postnorm/residual/layer gate |255988|255623|
| Total | Seven calls versus three |861363|846268|

These are compiler timeline endpoints, including each schedule's initial offsets;
the summed reduction is15095 cycles (1.7525%). They are not elapsed microseconds
or a measured launch-cost bound. B preloads the original two GeGLU global scalars
at61593..73009, before the last joined-output store ends326618. This demonstrates
compiler scheduling freedom separately from removing four host waits; it does not
isolate how much actual latency that overlap saves. All canonical BF16, joined,
packed, scale and offset HBM stores/reads remain. No data-layout/quantization change
or cached computation was introduced.

Compiler configured capacities are SRAM268435456, TRF33554432, VRF4194304 and
SPM32768 bytes. Peak unions of explicit allocated address ranges are:

| Graph | SRAM | TRF | VRF | SPM |
|---|---:|---:|---:|---:|
| A |15745024|0|3932160|448|
| B |153894912|33554432|3948544|1024|
| C |157286400|33554432|3932160|192|

The schedules also contain17/33/8 address-null SRAM lifetime records. These are
retained in raw schedules and excluded from the explicit-address union; the table
is **not** a complete physical/runtime peak-memory proof. All successful compiler
allocations and noDramStack findings are preserved; no inference from summing
incompatible tensor lifetimes is used.

## Prepared but unexecuted correctness and performance probe

New test_int4_grouped_threegraph compares all11 candidate scratch fields against
separate seven-call and normal-typed three-call owners, two feedback steps and
both true row orders. Existing independent BF16/per960/F64 oracles are retained.
Expected composition checks are255918080 bytes each for3-vs7 and raw-vs-typed,
15728640 mid-residual bytes,190 prelookup invalid-owner checks, and196608000 original
intended-code BF16 comparison events. These are planned counts, not device results.
B/C raw binding checks exact13/9 slots, checked extents/ends and256-byte alignment;
all5/2 mutable owners must be disjoint from every immutable/mutable argument.
I32/I4 reinterpretation remains only at the launch boundary. Both Rust owner
borrows and public Buffer clones live through the awaited run.

The unchanged q020 graph remains the mandatory complete performance control.
All three waits, runtime validation and HBM boundaries are inside its ABBA timer;
7-versus3 diagnostic timing is separate. Prior323213 remains2.13%/2.10% slower:
another129.48/128.79us is needed to meet the preregistered10% q020 gate. Neither
resource success nor four fewer launches establishes that saving. Model/task
quality remains entirely untested for this arithmetic.

## Resume point

Independent binder/probe review must be recorded (if unfinished), then root must
separately authorize any fresh standalone ELF build with new OUT_DIR/capture.
Rebind all19 inherited device entries and new graph images before root-only actual
execution. Require strict11-field parity/reuse/negative gates first, then both
>=10% complete q020 ABBA gates. No current evidence authorizes model promotion.

Replay source identity with audit_source.py and resources/image identity with
analyze_resources.py. The latter reads existing artifacts only. Raw11 local run
records/logs/source hashes, ten schedules and fresh resource binaries are retained
alongside resource-result.json. Targetffn-k256 is idle after a011.
