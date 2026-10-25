# Checked packed-I32 / I4 binding micro

The normal typed I4 consumer compiled in a001. Its resource schedule uses only
three HBM input/output allocations (30,720 / 61,440 / 2,048 bytes), no DramStack,
and a 32 KiB physical TRF allocation per PE (the I4 source has 30,720 logical bytes).
The cluster/slice are each one live element with physical padding; this is an ABI
and integer arithmetic proof, not a throughput forecast for a complete model.

The producer is byte-identical to ops_int4_pack_words at 4aae1ab7, independently
passed r1185/321124. The new joint gate checks its exact 30,720 output bytes before
binding that same immutable I32 owner as the verified consumer's input Buffer.
The generic adapter accepts the producer's named dimensions without reshaping.

Known-byte raw and typed controls exclude packing; the separately labeled joint
pack-plus-raw timing includes both completed launches. It is not a model speedup
comparison. Eight fixtures cover every signed nibble, extremes, zero, sparse and
random values, plus a true paired column/row permutation. All 512 dot results per
fixture are checked against an independently accumulated I64 oracle. Output
owners are distinct and poisoned before first use. Input bytes, producer words,
shift constants and repeated-use outputs are checked. The deliberately shortened
buffer must be rejected by both the adapter and normal runtime, without output
mutation.

The adapter borrows source owners immutably and the output owner exclusively for
the entire awaited call. DeviceSend only clones allocation-owning Buffer handles
into a local list. The public runtime preserves provenance, extent and topology
validation. Its submitted pending command retains Buffer clones; Drop waits for
an outstanding ticket before releasing them. Cancellation therefore does not free
storage still used by the device. This is a source-audited lifetime argument, not
an actual cancellation stress result. No Rust I4 alias, raw pointer, unsafe cast,
private SDK field, runtime verifier bypass or new dependency is introduced.

Nothing here establishes quantizer quality or changes model arithmetic/storage.
