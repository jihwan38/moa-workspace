# Power2 A4 actual foundation gate — model HOLD

Arena321149/r1194 passed0.347s with source de41950e and frozenELFd5a1c47b0d599380b99a64921ae471ca6d083123aabfbf2fa962a218acdd0ccb. All491,520 ordinaryI32 codes and128 F32scales matched the independently defined mathematical power2/ties-away reference exactly. Thresholdneighbors, extremefiniteBF16 exponents, zeros/subnormals, truepermutation and inputimmutability passed. Independent arithmetic/probe review byround2_path passed.

Completed quantizer calls averaged15.766us, range14.653–16.503us. This includes code/scalestores but excludes packing and theINT4 consumer. Do not add separate micro timings into a serving claim.

Providedweights were reconstructed from the originalF4 andE4 blockscales beforeglobal scaling, sampling64 rows from eachUP/GATE/DOWN inlayers0/6/47 (4,423,680 values). Power2I4 row reconstruction relativeL2 was0.233–0.411; per960blocks reduced it to0.198–0.315. These large errors and missing complete-chain/modelquality evidence justify HOLD onA4 model integration. The exact arithmetic and packed-word/public-binding foundations remain available. The old amax/7 DivF guard failure is preserved, not relaxed.
