# Fixed unscaled E4 B16 KV micro

Parent functional kernel 4bf039b2; resource/evidence 27471cbf. The three parent
modules are byte-identical in this tree. Only a host diagnostic, executable and
independent CPU codec are added. No production call or cache dtype changes.

Four cases select supplied O layers 0/6/46/0, live widths16/14/16/14. Only the
O matrix and channel scale are loaded from model.safetensors. All Q/K/V and
past cache contents are synthetic, not real model histories. Each case has two
steps: append/read, then reverse the live owner rows and advance each owner by
one position. All arms receive the same prior candidate attention as the second
query (matched synthetic feedback). Slots include0/31/32/63 and positions
256/1023/1024/8191; the second step exercises physical ring wrap. Inactive K/V
rows are NaN but are outside the valid-prefix append; inactive Q is zero and
masks are zero. Future and unused cache entries contain finite poison, since
NaN V times masked-zero weight would itself alter the unchanged attention math.

The exhaustive codec fixture uses all65,280 finite BF16 encodings across the
K/V append tensors. Nonfinite code positions are replaced with signed largest
finite BF16. CPU conversion uses an independently decoded E4 integer lattice
in units2^-9, binary nearest selection with even-code ties, signed zero, and
explicit finite overflow saturation to signed448. Separate CPU tests cover all
126 adjacent E4 midpoint boundaries and their BF16 neighbors, both signs, plus
all finite BF16 codes against a linear nearest-code search. Hardware overflow
semantics must match this frozen expectation; any mismatch fails and is kept.
The existing complete sliding_16 graph is used; codec attention/O is not an
accuracy assertion because this fixture has all masks zero.

Three independent pool/output sets are used. Candidate stores E4 and receives
original BF16 append values. Intended control uses the original current
one-launch BF16 commit/attention/O graph, with CPU E4-roundtripped BF16 pools
and append inputs. Their attention and O bytes must match exactly. A third
current BF16 graph receives original BF16 history/append values; its drift is
reported separately and is not judged exact or as task quality. All pool bytes
are checked against independent CPU scatter updates, including untouched slots
and inactive-prefix exclusions. Separate NaN destinations are restored before
every feedback step. Q/K/V, metadata and supplied weight bytes are immutable.

ABBA12 repeats compare the complete current one-launch BF16 graph against the
complete candidate: append codec/scatter, old-history gather/decode, unchanged
BF16 QK/PV/softmax, supplied O. Uploads, allocations, metadata preparation,
readbacks and CPU checks are excluded equally. Repeated append is idempotent;
pool bytes and candidate results are rechecked after timing. Profile markers
wrap one complete current/candidate call after untimed correctness and ABBA.
A fixed aggregate >=10% completed-call gain is required after all diagnostics
print. Exact intended bytes are mandatory; no threshold relaxation.

Arguments FIRST END select a half-open subset of cases0..4. Default runs one
case0. Root alone chooses Arena submissions (recommend0 2 and2 4, or individual
cases if full-pool reads approach70s). No host integration is authorized here.

Build must use record_run.py --fresh-package --capture-artifact with the actual
binary path, in the exclusively owned tail256-model target. Captured source
hashes and immutable binary are the handoff; later mutable target files are not.
