# Three-owner full512 model-only gate

Base1e2e7ea (Tail384 model on root650642a). Dedicated PreparedThreeOwner512
wraps the original PreparedGroup plus exactly three first/final row maps.
The public constructor validates3 independent real reservations with aligned,
checked, final-only ranges of256/128/128 in any order before privately deriving
four chronological quarters. Original four-independent-owner and Tail guards
are unchanged. Every quarter is real input; no padding or invented owner exists.

The same existing prepare_rows/advance_layer/finish kernels execute all48layers.
Model active allocations expose actual owners once each and preserve shared
scratch/physical page guards. Finish returns exactly3 owner-ordered outputs.
The prepared checkpoint may yield only between whole layers to independent
request decode. No Worker, admission, cohort deadline or device module changes.

Oracle requires four current P128 exact reference plus current Tail256 and
stronger Tail384 decompositions, with identical true-prefix recomputation before
each control. Inclusive completed preparation/layers/finish timing, no reads or
independent interleaves in timing. All vocabulary values and whole cache bytes,
physical aliases, reset/permutation, page/ring boundaries and48 R64 interleaves
are checked. Root alone dispatches NPU; compiler success establishes no speed.

The final probe records two meaningful packed controls: two Tail256 checkpoints
(each two real quarters), and Tail384 with128 from each owner followed by the
doubled owner's remaining128. The latter is the stronger upcoming Worker
alternative. Both controls independently match all4-P128 reference outputs and
recompute the same true prefix before execution; no request computation is reused
between different owners. The Tail384 control poisons its inactive quarter with
NaN, so subsequent full512 preparation must overwrite all four real quarters.
Control/candidate KV slots remain disjoint and whole-cache comparison follows.

Host validation:104 library tests passed; probe type check passed. No device or
Worker change in this source. The optional case6 includes48 R64 interleaves;
cases7/8/9 cross the2048 ring/page boundary for each doubled-owner permutation.
