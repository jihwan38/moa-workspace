# E150 readiness independent review

`independent-review.py --check` verifies the measured E150 FFN image (`52dc83a9…`, 300032 bytes) against the diagnostic host. Only profile-name bytes differ; tasks, addresses, tensor I/O, constants, and the other QKV/O images are unchanged. The diagnostic crate preserves all 46 source inputs except the host-only test entry point. The 142 emitted marker kinds and order match the frozen schedule.

The two down native consumers (657/665) read weight DMA outputs 430/647, lookup tables 516/532, and the same activation TRF 653 and coefficient VRF 630. The two weight views select rows 0–29 and 30–59. Up canonical gather 498, GeGLU 513, and down scale ingress 666 are source-bound landmarks.

This is one E150 internal trace allocation bracketed by E147 whole controls. It does not contain a contemporary E147 internal trace. Marker intervals include waiting and instrumentation; overlapping intervals are not additive savings or physical-resource attribution.
