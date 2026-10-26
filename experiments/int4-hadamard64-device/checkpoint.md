# Fixed H64 device-layout checkpoint

Direct half-view route is **HOLD**. Functional source `18e50b57` preserves the fixed sign-before-transform, six RN32 stage order, and one final BF16 rounding. Plain hostcheck passed (11.44 s), and the CPU coordinate audit covers all 1,966,080 H3840 inputs and the 16 frozen root vectors.

The one authorized resource compile rejected stride1: the packet `H/2%8` has actual DM stride2, while Fetch requires its innermost packet axis to be contiguous. Public `begin_interleaved`/`unzip` does not make this partner fetch free or legal. The SDK suggests a real post-fetch transpose; implementing that or a separate partner staging owner would require a new, fully charged design. No retry was performed.

No schedule, allocation report, kernel image, standalone ELF or device result exists for this prototype. Therefore no cycles, SRAM-peak, spill-free, H+L, or complete-FFN performance claim follows. The source cost model remains an uncompiled logical count. The earlier CPU sampled numerical gate and official q020 are unchanged.

`result.json` binds the source, copied raw runs and failure. The tail target is idle. Reproduce only the read-only coordinate audit with `PYTHONDONTWRITEBYTECODE=1 python3 experiments/int4-hadamard64-device/audit_mapping.py`; do not start an additional compile without the parent's next bounded scope.
