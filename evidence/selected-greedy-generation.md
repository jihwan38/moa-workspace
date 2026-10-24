Generation now caches the exact request banned set once and exposes it through
banned_tokens(), with sampling_config() and SamplingConfig::is_greedy(). The
latter is exactly temperature<=0.0, preserving NaN/stochastic branch selection.

advance_inputs_greedy(count,token,on_delta) accepts a caller-selected exact
CPU-equivalent greedy result. It checks complete input count, greedy policy,
final-prompt/decode applicability, vocabulary range and bans before mutation.
It then enters the same advance_selected transition as the existing logits
sampler: EOS handling, streamed decoding, reasoning marker handling, stop-prefix
buffering, callback cancellation, completion counts, max-token limit and next
position remain shared. Existing advance/advance_inputs signatures are intact;
stochastic sampling still calls the unchanged sample implementation.

Worker/Model wiring is deliberately separate and remains with the root agent.
The external selector must match CPU total_cmp ordering and lowest-ID ties.
No request text, media, benchmark flag or answer changes this dispatch contract.

Validation: four CPU selected-token state tests pass, using a small local
WordLevel tokenizer fixture. They exercise logits-vs-selected parity plus
concrete expected stop-buffered output, reasoning/content events, EOS, length,
callback cancellation, rejected selections without phase changes, intermediate
prompt chunks, zero completion, and stochastic top_k1 fallback. All20 host tests
also pass. Initial test-only callback lifetime inference error was fixed with
an explicit Delta<'_> parameter; no production transition change was needed.
