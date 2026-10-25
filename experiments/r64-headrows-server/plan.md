Base root4f0be45 (c056 functional source), passed HeadRows kernel3a3ebaf/318853.
Only the canonical R64 full-window fallback switches to head-local SlidingRows.
All512/768/mixed512 dispatch conditions, KV commits/masks and full attention stay
unchanged. OriginalDecodeHeadRowsDiagnostic passes a per-call boolean through
Decode64::run to decode_sliding; there is no persistent selector or global toggle.

Probe recomputes both control and candidate histories sequentially in one pool,
retains every control BF16 logit byte, and forces identical R64 arithmetic.
Uniform mode has64 independent1024-token prefixes; mixed mode changes one request
per R8 group to0/511/512/767/768/1023/1024/1535. All other rows remain1024, so each
active group requires the full-window fallback. Four disjoint physical full pages
per request cover1535+postdecode; both ring/page permutations are bijective.
64→64reverse→7→64permuted tests new full-window and unchanged partial mixed routes.
Estimated uniform job50–55s from observed512history fills; each job stays70s.
No source mutation in root or already-frozen prefix768 server.
