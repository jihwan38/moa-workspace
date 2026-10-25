# Two-owner384+128 actual-model gate

Arena318938 / r697-prompt384128-model-submit passed with args1664 1920 1.
Source07fcb9e, immutable probe manifest8ad5404. Every vocabulary value matched
four original native-P128 passes for both owners,48 independent R64 interleaves,
four subsequent two-owner B16 decodes beyond2048 and position-zero slot reuse.
Actual checkpoint timing (excluding downloads and independent decoder calls)
was375.943 ->172.256ms; reset365.084 ->162.565ms. This establishes about204ms
saved model-call work for this tail shape, not an HTTP throughput/SLO result.

Independent e2e_evidence review confirmed separate actual-owner validation,
chronological A quarter commits, unique allocation reporting, row2/3 output
mapping and unchanged discard-on-partial-failure contract. No ordinary
four-owner independence guard was weakened.
