r1184/job321121 completed successfully in11.345s on frozen source3a877b1e,
ELF fbac12cf562b1c65c3586315f42f6985fefa828a4134916161473f1ad707abc5.
The parser and actual-result.json retain exact stdout/provenance.

Completed input-quantizer + native-LM ABBA measured1,141.37–1,160.59us against
current exact BF16 storegroup2,378.47–2,402.29us. Six paired mean reductions are
51.62–52.23%, clearing the requested25% complete-LM threshold. Startup quantization
was4.402ms, measured separately. Persistent allocator increment1,007,732,480bytes
is below1.1GB; original BF16 embedding remains present. This is a standalone memory
measurement, not full-model capacity proof.

Intended algorithm gates passed: all1,006,632,960 provided E4 embedding bytes and
262,144 F32 scales match independent CPU quantization, all six input quantizations
match, and all3,072 sampled intended-quantized F64 outputs match exactly after the
specified BF16/softcap boundaries (zero tolerance failures). Original BF16 control
F64 samples also match. Candidate/control row permutations are exact, both tables
and normalized inputs remain immutable, and every output is finite.

Approximation remains material:13,103,568/25,165,824 BF16 logits differ from the
old LM. Relative L2 is4.96%,4.78%,2.28% for the three fixture families; max absolute
error15.5391. Six of384 greedy policy selections change. All six are the explicitly
winner-banned arbitrary-mask policy, representing three distinct fixture rows and
their permutations; empty/default/all-banned policies have0/288 changes. These
synthetic fixtures are not task-accuracy evidence and do not establish actual-model
argmax stability. The older BF16-H960 strict failure remains separate and unchanged.

Trace supports the movement/epoch mechanism. Cluster0 control Task4,768,936 raw
versus candidate inputquant13,643 + LM2,239,521. Summed DMA3,960,484→2,070,741;
TU count97→54 including quantization, summed TU564,557→188,748; old32 separate
TRF stores disappear into the one retained native-input epoch. Engine spans overlap;
these are raw counters, not additive wall time or a clock-frequency claim.

Root authorized a new isolated default-off B16-only model hook and real-history
approximation/capacity gate, then fresh public quality only if worthwhile. No serving
activation is justified by this micro alone. Model histories must be teacher-forced
identically; only the terminal B16 LM may differ, so KV equality remains mandatory.
