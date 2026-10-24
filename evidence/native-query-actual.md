Actual NPU318089, run r312-native-query-padded-scale-fix-submit, source2868a38.
Binary661c205d5f95bdf2f5cd037bb009e8d761119c03dc0f47ee4d878a1c53c36a03.

All491,520 input FP8 values and128 row scales match the independent CPU row
quantizer for seeds0,31,79. All524,288 fused-versus-staged outputs match each
seed. The dyadic CPU quantized-input oracle matches every524,288 output;
the two nonrepeating random cases match2,048 independently computed sampled
outputs each (all128 rows,16 boundary/distributed channels).

BF16 activation comparison: seed0 is bit-exact, while random seeds31/79 differ
in409,050/408,949 of524,288 outputs. RelativeL2=.0243571104/.0244095880;
maxabs57/60 in this synthetic domain with per-row power2 magnitude variation.
Those are measured precision changes, not a correctness-equivalence claim.

MatchedABBA20 warm launches: BF16 TRF32 control204.358/204.813us; native fused
279.242/278.742us. REJECT production adoption: numerical change and36% slower.

Next bounded diagnostic separately times already-proven quantize_input and
project_query_quantized, including a two-launch sequence. These include HBM
traffic and do not exactly decompose fused internal DM redistribution. A
measured difference can identify whether the known per-row-to-shard transfer
is a worthwhile specific repair; no quantizer algorithm/layout changes yet.
