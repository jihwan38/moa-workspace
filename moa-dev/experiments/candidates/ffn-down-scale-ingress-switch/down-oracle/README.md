# E151 down-oracle preparation

Original E134 reference/input sources are byte-identical, carried through the passing E150 suite. The complete E151 production MLP is bound at build time. Raw down uses a separately compiled diagnostic image; full FFN must embed the exact whole production image. This preparation claims no new hardware validation. Build records, actual uploaded archive and result checks determine final scope.

Raw: nine cases/34,560 BF16 outputs, exact with signed zero equivalent. Full: the original single constructed FFN regression input/3,840 outputs and original tolerance. Timings are excluded from performance. NPU execution requires a positive first whole screen.
