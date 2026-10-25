# Three real quarters after Tail256

Read-only proposal on source2c69fbb. No Tail384 device or host implementation,
performance result, or Worker dispatch is present in this commit.

`GroupPrefill::advance_rows` in src/host/prefill_group.rs already separates four
physical tensor quarters from the actual PreparedWide state slice. QKV, norms,
residual, O and FFN use the same four tensor quarters throughout. Only
attention/commit and next_layer increments iterate actual states. There is no
states[3] dependency in this executor. The explicit debug length guard currently
accepts2 or4 and would need3 added. `GroupPrefill::supports` must remain exactly
four independent owners; no weakening or invented padding request is needed.

A dedicated PreparedTail384 with states:[PreparedWide;3] can retain owner first
and final row maps. One owner384 maps final row2; independent256+128 maps1,2;
independent128+256 maps0,2; three independent128 map0,1,2. Each public owner range
must be aligned128, checked for overflow, final-only logits, and independently
reserved before private chronological quarters are derived. Only tensor row3
x/attn_s/attn_f is initialized as padding; it receives no metadata or KV calls.
Existing GroupPrefill scratch is sufficient: no model weight or KV duplication,
no new dense kernels, no extra persistent allocation. Active allocations expose
only real owners and existing overlap guards remain in force.

Model-only gate first: exactly three original P128 controls against one padded
P512 dense pass; all vocabulary outputs and all raw cache bytes, independent
sentinel reservation, NaN padding, starts0/256/896/1920, ring/page wrap, reset,
owner-order permutations and48 independent R64 layer interleaves. Include every
preparation allocation/upload, all48 completed layers and finishes in timing.
One/two/three-owner output maps need separate checks. Exact math is expected from
row-isolated dense axes but still requires the actual gate.

Initial Worker scope should prefer existing valid real512 quad/full pair and
single512 before any padded tail. A new oldest-three complete128 selection must
not skip a short/unaligned owner. Three-quarter combinations can follow only if
none of those full512 routes applies. Tail384 costs class4, exactly like Tail256;
its checkpoint retains scheduled ranges and yields after a full layer. No extra
arrival wait, cohort deadline change or speculative credit is needed. Consider
three-owner512 ([128,128,256]) separately: existing public full512 constructors
do not support it, so do not describe it as an available route or silently
invent an owner map while implementing Tail384.

The measurable opportunity is replacing current Tail256+single128 when three
complete quarters are already ready. P512 dense work currently executes twice
across that pair of calls; a three-real-quarter pass would execute it once and
add only the third actual attention/cache segment and owner finish. Tail256
observed135ms and two-P128 control189ms do not establish the unmeasured Tail384
latency. HTTP benefit depends on the frequency of eligible triples and remains
separate from the model gate. No claim about C15/C16 or leaderboard rank follows.
