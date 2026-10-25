# Matched model gates and clean c073 integration

Sourcecc1915c / ELF66ee9513, default-off candidate on c072624431f:

- r1076 job320297, actual512-token histories:23,855,104 BF16 logits exact;
  3,321,888,768 paired cache BF16 values exact; diagnostic elapsed18.470s.
- r1077 job320298, actual1024-token histories: same counts exact;
  diagnostic elapsed24.114s.
- Aggregate47,710,208 logits and6,643,777,536 cache values. Both jobs also
  passed retained B16/R64 outputs, devicegreedy/None/all-banned policies,
  seeded stochastic samples, scalar/B16/R32 width transitions, position0
  reuse,48 active P128 interleaves,31-token prompt reuse and explicit
  original epilogue/O/QKV fallback followed by re-entry.

Matched warm eligible steps2/3/4/7/9 put the48-layer FFN epilogue at
14.11–14.17ms control versus13.63–13.69ms candidate, saving about0.44–0.52ms.
Complete-step deltas range0.27–1.53ms and include host/cache/order effects;
unchanged scalar/R32/fallback deltas must not be attributed to ingress.
This supports the micro's bounded gain, not an HTTP/SLO throughput claim.
Every measured step and per-stage value is in model-results.json.

Clean export basefd3c81d preserves currentc073 exact sliding/full stationaryO
and projection-owned groupedFFN. Integration order: a880173 scoped kernels/
host,677fb80 probe,ed8580b separate authorized Automatic activation. Only
lib.rs additive module exports conflicted; all existing exports retained.
Explicit original ingress diagnostic remains false. Automatic enables new
UP ingress only for all-singleton canonical B16 calls with existing staged
and composed FFN enabled. Width1/scalar,17+R32/R64, ordinary prompts/groups,
mixed calls and disabled epilogue retain their previous routes.

Storage interplay review: c073 GroupPrefill owns ProjectionStorage.layers
and output plus independent P512 scratch. B16 uses only existing Prefill
native_q_input/input_scale/hidden/q_hidden/hidden_scale/projected/x, with
immutable original native weights retained. New kernel adds no host-allocated persistent
HBM workspace or shared ownership, cache write, owner metadata or yield
change. Its compiler-emitted16KiB Switch constant is recorded in the micro
resource evidence; this is not a model/cache workspace allocation.
All five relevant B16 control/candidate/norm files are byte-identical to
passed micro r1056. No L480 quantizer change is included.

The first export CPU attempt a007 reached an unresolved additive lib marker
and failed before test execution; it is retained. After explicit union and
separate activation commit, a008 passed111/111, zero filtered or ignored; a009 plain server/probe check
passed10.20s. Independent batch_kernels activation/interplay audit passed.
Historical B16AttentionOutputDiagnostic and its explicit original both keep
old ingress; compare them together, not current Automatic against one old
O control when attributing O-only timings. The new ingress probe uses its
own explicit original/candidate pair. Root owns
fresh combined public6/performance and final release build; no new Arena
submission or root-source edit was made by this agent.
