Independent offline review passed; reproduce with `python3 independent-review.py --check`.

The E147 and E148 O diagnostic executables differ from their respective unlabelled hosts only in the original O image's marker-name byte fields. Task bytes, addresses, constants, I/O, and the other two kernel images are unchanged. All 45 control and 40 candidate operation markers correspond in order and kind to the image-bound schedules. Candidate instructions 93/106 are Main+Sub+VE direct F32 accumulation-to-VRF tasks; 55/146 are Sub CustomBroadcast-to-VRF tasks, all labelled TuExec.

The separate FFN diagnostic uses the current E147/E142 FFN image `c8185c19…` in the same unlabelled control host, with only FFN marker names patched. Its 138 operations match the bound schedule. Both up/gate native consumers read the single emitted activation TRF produced by instruction 274. Two source reload calls therefore cannot be counted as two emitted transfers.

This gate validates provenance and observation mapping. It does not establish a physical contention cause, convert static intervals into NPU timings, or remove instrumentation overhead. Runtime comparisons must retain same-cluster observations and plain whole-kernel controls; overlapping intervals are not additive savings.
