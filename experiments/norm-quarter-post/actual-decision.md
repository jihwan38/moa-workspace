Actual318918/r692 passed all512 rows in four cases, with all7,864,320 BF16
outputs equal to each of two device controls (15,728,640 pairwise comparisons).
The independent CPU H240 producer layout remained immutable. F64 sampled
residual/gate references met the original0.003 NRMSE bound; CPU is not bitexact.

ABBA first candidate after control124.5–128.2us vs control after candidate
155.1–160.3us (~18–22% improvement). Consecutive candidate109.1–112.6us vs
consecutive control135.4–137.2us is reported separately. Reset uploads are outside
timing; each entry includes one completed kernel. This is not an HTTP score.

Actual tid0 Task253853→207225cycles (-18.4%); DMA count20→11, union248894→170802;
TuExec count40→12, union34936→23019. Intervals overlap, so category sums are not
additive. This supports reduced ingress and repeated quarter scheduling as the
mechanism, with identical H480/BF16 arithmetic. Recommend matched model gates.
