# Exact physical sliding pool32

Source/control is c079 `394d60a290e6137c88938300468050128de39be6`.
Root authorized this isolated implementation after the read-only memory audit.
No q018 LM, F4 arithmetic, full-page, context, query-batch or request-output changes.
Root alone dispatches Arena and decides activation.

The change reduces authoritative BF16 sliding slots from64 to32, while preserving
R64=64 query/scratch/logit axes, R32 hot stages, B16, Ts1024, Fc256, Tf512 and the
8192-position serving context. Thirty-three and later requests wait in the existing
FIFO queue; no accepted request is dropped or truncated. Full pages still reserve
the complete prompt+generation budget before admission. At8192 positions the256
full pages already admit at most16 owners, unchanged.

Predeclared affine invariant, before implementation: for every s in0..31,
t in0..1023, n in0..7 and d in0..255, both old/new K and V use byte offset
`2 * (((s * 1024 + t) * 8 + n) * 256 + d)`. Per-ring stride4194304B,
per-token stride4096B, per-head stride512B and BF16 order are unchanged. Only the
allocation's outer extent changes268435456→134217728B. Forty sliding layers,
two pools/layer save10737418240B (10GiB). Full-attention storage stays2147483648B.

The numerical kernels already accept the separate physical axis Rc64; most use
whole-pool gather/scatter with absolute offsets, not padded host slices. The current
Worker wrongly couples its requested storage capacity to R64::SIZE for this change;
it must request the physical-axis size instead. Merely passing requests32 to the
old constructor leaves the64-slot allocation and saves no memory.

`physical-literals.json` inventories every non-bin physical64/65536 literal.
Repair the physical pool axis, zero-pool output tile, and Ts#65536 / Rc=1#64 view
contracts to the new32768/32 extent. Keep R64=8#64, P128%64=32#64, all full-page
131072/256 extents and greedy65536 sentinel arithmetic untouched. The physical
axis may retain an explicit compatibility alias for existing imports; its declared
dimension and public comments must identify32 slots unambiguously.

Ownership guards remain: Workspace rejects requests>physical slots; Model validates
each ring against workspace.requests before execution; independent owners retain
distinct rings/pages; active prompt allocations remain excluded from decode. R64
metadata uses valid last-live offsets for padding and zero masks, and counts<64
commit only live B16 segments. The compact serving path therefore never writes
padded R64 rows. Automatic17..32 hot stages and forced-R64 diagnostic remain valid.

Gates, with no new correctness assertion inferred from compilation:

1. Resource/type gate for returned zero pool plus representative whole-pool
   gather/scatter and fused attention signatures. No hidden64 allocation/view.
2. Physical micro: every slot0..31, exact address/strides, first/last/tail ring,
   stale finite contents, distinct K/V, partial live commits, untouched neighbors,
   all returned zero bytes, and unchanged full-page255 addressing.
3. Bounded actual model gates covering all11 existing Group/Single/Pair/Tail256/
   Tail384/Three schedules, wrap/reset/NaN padding, sentinel and full per-owner KV.
   Existing44-owner fixtures cannot run unchanged. Use at most32 distinct owners,
   with B16 interleaves in paired arms and a separate17..32 R32/R64 history gate.
4. When comparing physical64 and32 programs, freeze both source/ELFs with an
   identical deterministic bounded owner/probe schedule. Cross-run digest equality
   must be labeled as digest comparison, not an in-process all-byte comparison.
   Until both execute, do not claim old64-model parity.
5. CPU admission32, invalid slot32, deferred33rd request, reservation reuse/epoch,
   cancellation with active checkpoints and session compaction; all package tests
   and plain server check. No source-only test weakens existing64-query tests.
6. Actual complete model memory after all lazy/media scratch, six task suites,
   HTTP throughput/TTFT/TPOT and sustained queue behavior before promotion.

The tradeoff is general maximum32 concurrent resident short requests. It can reduce
throughput or increase external TTFT above32 arrivals. Worker receipt diagnostics
start at dequeue; that never removes actual HTTP queue delay. This is a memory
strategy, not a demonstrated speedup, and changes no cohort/credit deadlines.

# Joined-weight sharing audit

SDK0.9.0 `tensor/memory.rs:179–235` makes HbmTensor's owner/from_parts/own private.
Public owning reshape at363 consumes its input and preserves total physical size;
it cannot split one owner into two. Public view/tile at293/486 is borrowed and
retains outer padded extent; a host late-half view is not automatically a dense
UpWeight owner. Existing `&HbmTensor<UpWeight>` consumers cannot accept it unchanged.

Public runtime Buffer::slice (buffer/mod.rs:36) does share Arc ownership. A checked
DeviceSend→Buffers→Function::run adapter could bind exact immutable UP/GATE byte
halves to compiled images, without constructing private tensors. It requires its
own layout/argument/lifetime/cancellation gates and is not a drop-in typed API.

The safest typed deduplication would move a single JoinedWeight owner into
NativeFp8Weights and pass the whole owner into native consumers, selecting0/1 on
device. GroupPrefill would own only shared activation scratch. This changes the
B16 ingress/epilogue, native R32/R64, P128 and fallback/core interfaces; no arithmetic
change is intended, but all those compiled paths need gates. Current joined shape
is [Projection,L/240,H/960,L%240,H%960], exactly two dense old UpWeight halves.
Startup should join/drop separate weights layer-by-layer, not hold a full second
model during migration. Removing duplicate payload saves5662310400B plus5898240B
duplicated row scales if those too are unified. Original provided FP4/global/block
scales and DOWN remain. This retains64 slots but is a broader launch/API rewrite.

F4 per960 shadows from8df08ec3 need4246732800B weights +35389440B scales for all3,
or2831155200B+23592960B forUP/GATE. Current measured warm c079-equivalent headroom
3515675648B therefore leaves660927488B afterUP/GATE, before new scratch/allocator
effects; q018 warm2507943168B cannot fit evenUP/GATE shadows. Rc32 adds exactly10GiB
payload headroom without depending on either approximate method. Actual allocator
and peak-lazy memory still require a full-model gate.

The inactive disk growth is independently inventoried in
inactive-debug-growth.json: e2e's completed debug069d081abdb74c61 and6aca037d7b1a7242
families total7299553880B. No files were archived or deleted by this audit.
