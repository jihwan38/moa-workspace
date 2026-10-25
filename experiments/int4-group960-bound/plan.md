# Checked packed-I32 to existing I4 projection binding

Base4dc3c818 retains actual layer0 intended-code proof job322975. This gate
changes only the host call boundary, not the compiled projection arithmetic.
No source/model-quality or complete-FFN claim is added.

UP/GATE slot order is packed-I32 input, activation scale, typedI4 joined
weight, group scale, mutableBF16 joinedoutput (fiveowners). DOWN adds global
scale before mutableoutput (sixowners). Exact expected bytes are respectively
[983040,2048,58982400,491520,31457280] and
[3932160,2048,29491200,245760,4,3932160]. Every bound buffer must have that
exact size,256-byte-aligned start and nonoverflowing checked end. The mutable
output interval must not overlap any immutable input interval. Input-input
aliasing is not prohibited because all are borrowed read-only; the probe uses
separate inputs. Static typed mappings preserve all weight/scale/output shapes.
PackedInput/Hidden retain [P/256,K/960,P%256,K%960/8] I32 words, exactly the
same bytes as the validated [P/256,K/960,P%256,K%960] I4 input.

Only public DeviceSend clones owning Buffer references; original HbmTensor
borrows remain across lookup and Function::run.await. The adapter never creates
an HbmTensor alias, modifies an address, copies input bytes, or reinterprets
inside a device graph. The ordinary typed call embeds the same kernel path
and empty generic key in the probe. Later actual validation must compare that
embedded image to the existing validated artifact.

A new standalone probe keeps the two original typed arms and adds a separately
poisoned raw-bound arm. Its input words are independently assembled from the
canonical CPU integer codes by nested blocked-coordinate traversal and eight
little-endian nibble fields; equality to the existing pair-of-nibbles encoder
is checked for every byte before upload. All BF16 outputs, both true row
permutations, all original independent i64/F64 samples and all post-reuse
input/weight/scale/source immutability remain. New packed owners and raw outputs
are checked after every reuse. There is no device packer or quantizer in this gate.

Negative validations operate on public Buffer clones/slices, never fabricated
addresses: wrongcount, each shortslot, unaligned exact-sized input slice,
wrong slot ordering, and overlap of each immutable slot with output. No invalid
case reaches function lookup or Function::run. Output poison is read back before
any real launch. Actual-address overflow is guarded, not claimed dynamically
exercised through a fake buffer.

Timings retain typed-reference ABBA and add a separately labeled typed-I4 versus
raw-I4 ABBA (eight completed calls per block) for each projection. CPU encoding,
upload, download, profile, allocation, GeGLU and activationquant/pack remain
outside projection timers. Additional rawI32 inputs4,915,200B and outputs
35,389,440B raise estimated live payload351,824,900→392,129,540B, excluding
allocator/runtime overhead. This is bookkeeping, not measured peak memory.

Authorization ends at source and recorded host check. No device build/Arena.
DOWN245684>240000 HOLD is unchanged.
