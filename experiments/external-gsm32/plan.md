# Preregistered external GSM8K training diagnostic

Before any model response, pin the current public OpenAI grade-school-math
repository commit and raw `grade_school_math/data/train.jsonl` SHA256. Selection
is exactly `random.Random(20261026).sample(range(numrows), 32)`, in that order,
split into first16/last16. No response-based filtering, resampling, or test split.

Reuse the organizer's exact GSM user prefix before the first double newline,
request model/messages structure, max_completion_tokens768, temperature0, and
unaltered scoring.py. Ground truth exists only in harness scoring data, never in
request bodies, server sources, or server environment. Both shard lists and
request bytes are identical for c081/q020 and any later q021 frozen candidate.

Only copied harness select_test changes to one call for each of16 distinct rows
at concurrency16. Preserve the existing complete result loop; record full answer,
usage, score and separate completion status. A completed wrong answer contributes
accuracy failure but does not make the research runner fail/skip later jobs.
HTTP/protocol/incomplete failures remain separate. Root must execute every shard
for every designated candidate, even if another result is wrong or times out.

No warmup or answer optimization. Each independent Arena job has a hard70-second
limit; timeout/partial records do not become a complete accuracy denominator.
This is additional research evidence, not the official score, a guaranteed
private-accuracy estimate, or a request for use of private/test data.
