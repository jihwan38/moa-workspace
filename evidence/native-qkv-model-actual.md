Actual318125/r331-native-qkv-model513-submit, root-integrated diagnostic.
This is an approximate model comparison, not an accuracy pass.

513 tokens, same other kernels and disjoint canonical caches:
BF16 QKV control449.727ms; native QKV413.745ms (~8.0% shorter).
Prompt262144 logits: exact11377, maxabs2.78125, relativeL2.0803775;
both argmax100, top10 overlap7.

Identical teacher-forced subsequent tokens expose persistent KV differences:
- Step0 relativeL2.502541, maxabs19.9375, argmax107→237076, top10 overlap1.
- Step1 relativeL2.182243, maxabs8.6953125, argmax524→504, top10 overlap6.
- Step2 relativeL2.189571, maxabs7.78125, argmax107→107, top10 overlap6.
- Step3 relativeL2.080008, maxabs6.038086, argmax560→560, top10 overlap4.

All logits were finite. No claim of acceptable task quality follows from this:
synthetic prompt argmax preservation does not offset changed following logits.
Root owns real public quality suites before any production adoption. The
explicit BF16 diagnostic776fc296 remains available even if a separate candidate
changes its ordinary default. Default in this worktree/export remains BF16.
