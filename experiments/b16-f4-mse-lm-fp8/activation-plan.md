# Isolated q021 quality candidate

Root authorized activation after r1303/r1304/r1305 passed. The only activation
change is B16_LM_FP8_ENABLED=false→true. Model::new already prepares the returned
immutable q018 FP8 embedding shadow after complete q019 MSE FFN preparation;
Automatic selects it only when execution reaches B16 terminal LM. This is the
existing q018 LM scope, including noncanonical/mixed-call singleton B16 routes;
the serving Worker uses canonical pools. The F4 eligibility guard remains
separately stricter and is unchanged by LM selection.
Explicit OriginalB16LmDiagnostic remains false, and LM-only selectors still
normalize before F4 eligibility. Scalar, prompt, R32/R64, attention, FFN and
pool geometry are unchanged. Original BF16 embedding remains retained.

The two q020 CPU startup files were imported byte-for-byte from d52f0c08 in a
separate commit; source hashes are in parallel-import.json. Cargo.toml and the
ignored Cargo.lock remain byte-identical. No worker/admission policy changes.
Startup still publishes each complete immutable shadow only after successful
preparation. The parallel MSE helper changes scheduling of independent complete
240-row blocks, retaining the exact serial packed bytes and scales.

The frozen default-OFF model diagnostic at553c44c6 intentionally checks absent
startup shadow and Automatic remaining BF16. Its captured ELF remains the
oracle; rebuilding that same default-OFF-specific probe against an ON source
is not an activation check. This activation builds only the serving server,
with all library tests and a plain server typecheck. The next required evidence
is fresh public/task evaluation of the captured q021 server, not a claim that
the earlier teacher-forced model checks establish accuracy.
