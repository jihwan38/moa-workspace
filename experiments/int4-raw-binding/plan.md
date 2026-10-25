# Standalone I32 packed bytes to normal I4 consumer

Base394d60a2. Public API audit01281099 in int4-padded-trf established
DeviceSend::bind -> Buffers -> public backend::npu::function/Function::run.
No private fields, unsafe casts, SDK edits, new dependencies or model route.

Input B16,H3840 signed4-bit codes; producer output I32[B,H/8] contains two
little-endian words per16nibbles. Consumer is an ordinary typed compiled
I4[B,H] x I4[RawM32=32,H] -> I32[RawM32,B] exact integer dot. Standard typed
control actually runs, ensuring registry image inclusion. Raw adapter keeps
its owner read-only through awaited run and uses a distinct preplaced output.
Exact byte extent/dimensions/alignment checked before normal runtime validation;
foreign-device/provenance checks remain enabled. No returned typed I4 alias.

Firstresource compile; then supplied integer fixture strict gate covers all16
nibble codes, all-negative/extreme/zero, nonidenticalrows, dense/sparseweights,
row/columnpermutations, independent CPUi64 accumulated oracle, exact typed
control, poisoned outputs, raw input immutability. Independent host word and
nibble pack implementations must produce identicalbytes. Invalid-size adapter
checks must fail before dispatch. Runtime reject tests must not invoke invalid
memory. Cancellation lifetime argument follows pending Buffer clones and runtime
drain-on-drop; any actual cancel exercise must complete/drain before reuse.

No quantization quality or model speed claim. Batch agent separately owns a
public device mask/shift/add integer producer; integrate only after its gate.
