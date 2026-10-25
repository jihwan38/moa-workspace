q018-b16-lm-fp8 is an isolated approximate task-quality candidate. Root authorized
the source-only Automatic B16 flag after both OFF-source actual model/cache/memory
gates. It is not root production integration or an official submission.

Activation45c3b46c changes only the constant/comment. Independent e2e review passed:
Model::new prepares immutable shadow once; Automatic uses it only for B16 terminal
LM; explicit original-LM diagnostic still overrides it. Other model widths,
prefill heads, Worker policy and selection API are unchanged. Original BF16 table
remains allocated and supplies token embeddings and every original LM path.

All127 library tests passed, zero filtered, with CARGO_INCREMENTAL=0 and package-only
testdebug0; plain server check passed. Root will run fresh six public suites and
multi-token B16 GSM/IFEval width traces before any quality/performance decision.
The frozen OFF model probe is preserved and not relabeled as an ON-source run.
