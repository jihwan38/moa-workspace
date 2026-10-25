# Clean c070 integration

Base eb5004a. Functional order: e95d07a (exact stage extraction, composed kernels1..16 and retained allwidth micro) then178ef5d (per-call host adapter, proven Automatic activation, explicit original diagnostic, matched model probe). No Group/P512 FFN source or flags change.

Automatic composes only pure independent B16 singleton calls, canonical BF16 pools and Hsplit O. Counts1/>16, mixed prompt calls, legacy caches and disabled Hsplit retain original routes. Explicit original and composed diagnostics preserve AutomaticR32/O32 and native output choices; the model probe now explicitly requests the original control, so later default activation cannot contaminate it.

All NPU evidence comes from the frozen experimental tree/source recorded in build.json,build-all.json,model-build.json. The exported kernel/stage/adapter code is unchanged from those gates. The only post-gate serving change is Automatic dispatch to the already-gated graph and an explicit inverse control; root will build/freeze its combined server and validate public behavior. No new NPU dispatch was made by this agent.

Source extraction preserves every original HBM boundary and arithmetic. SDK rejects making the mutable O view inside the kernel; the exported ABI intentionally takes the existing host-created output view. Do not replace it with an internal reshape. The earlier distributed commit is rejected and absent.

Original implementation and numerical controls remain callable. Historical source, immutable ELFs, failed build diagnostics, full logs and allwidth/model probes are preserved.

Post-activation host checks: `runs/a008-b16-attn-output-export-tests` PASS111/111, 0failed/ignored/filtered (1m37s build,0.22s tests); `runs/a009-b16-attn-output-export-check` plain server+explicit-original model probe PASS9.79s. Both used CARGO_INCREMENTAL=0 and the owned reusable target; tests used package debug0. Generated Cargo.lock SHA256a99652661c588fe5c3a8b23278f2390d8cd2e99f2b33bcab6f52e97dbba04e2e exactly matches the gated source tree. Independent activation audit passed: original selector preserves R32/O32/use_wide, and default activation cannot contaminate the updated explicit control.
