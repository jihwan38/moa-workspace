# Same-model media preparation gate

Source uses root0463ae3's explicit per-call media-batch booleans. Defaults are unchanged. One model/weight allocation and one canonical64 pool are created; no duplicate12B weight load. Candidate and control own disjoint sliding rings and three reversed full pages per ring. Capacity1536 covers the longest1027-position test.

Model comparisons hold all transformer dispatch/arithmetic constant and change only media preparation. Candidate runs first. Live media rows select nonmonotonic/repeated indices from33 image patches and129 audio frames; this is a synthetic functional fixture, not a benchmark-specific route. Zero/constant/impulse/random source rows and clamped positions are included. Text tokens remain intermixed and padded embedding rows are checked by allP128 snapshots.

Cases:17 mixed at0;127 images at385;128 audio at384;128 permuted mixed at0. Actual text histories are computed before nonzero-start cases. Each pair has all262144 logits compared and three teacher-forced next tokens. A long-history pair then resets to a different17-row mixed prompt at0. A same-owner512 chunk tests chronological quarters. Four owners at0/384/512/896 use text-only/image/audio/mixed inputs, with48 paired independent R64 decode calls between candidate layers. Finally256+256 image/audio owners test the pair constructor. Each grouped owner receives three post-prompt teacher-forced decodes.

Expected totals:96 all-vocabulary comparisons (25165824 values) and9 completeP128 embedding snapshots (4423680 BF16 values). Every comparison requires zero bit differences and zero nonfinite values. The embedding snapshot API runs candidate first and returns(control,candidate); no KV writes occur. Supported-shape snapshot calls are rejected while preparation is active. Each completed layer is the only interleave point; the same Model/Workspace remains owned throughout.

Preparation and layer+finish elapsed time are reported separately. Logit downloads and interleaved decode work are outside those timers. These are single model-pass diagnostic timings, not warm HTTP throughput or official scores. Device build and hardware execution remain root-coordinated.
