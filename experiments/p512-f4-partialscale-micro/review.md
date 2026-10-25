# Bounded independent review

`batch_kernels` reviewed frozen `d7823768` host/oracle without source edits.
Residual, output, joined, and quantized-input CPU index formulas match the
existing aliases; UP/GATE concatenation and scale order match projection0/1.
The73r+127 mapping is bijective within512/256/384 live ranges, keeping padding
fixed. Every arm receives the same previous candidate residual. All seven
candidate snapshots are copied before reuse, and original error is described
as matched-input chain drift, not model quality.

The F64 projection samples use independently decoded fixed codes/group scales
and the actual immutable quantized inputs; final norm checks use the separate
untimed original norm stage. These are toleranced CPU samples. Exact all-buffer
reference is the dense fixed-code graph with identical group-scale placement.
No separate GPU local-partial export or CPU all-output bit claim is made.

ABBA contains the complete current norm-store+FFN+postnorm call in both arms;
reset uploads, snapshots, and static weight preparation are outside. No
concrete ownership, reference, mapping, or timing issue was found. Actual
compiled correctness, >=10% speed, and later model quality remain separate
gates. Typecheck and release build pass; root alone dispatches hardware.
