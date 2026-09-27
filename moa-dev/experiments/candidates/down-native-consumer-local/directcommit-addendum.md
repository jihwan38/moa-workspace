# E140 local-v1 assembly selection update

The initial design note conservatively retained E134's separate hi/lo/third DM tensors plus three Sub copies into packed terms. Implementation owner subsequently selected the already lowered E138 direct packed term-view commit pattern, transferred to one-byte E5 elements. This is a supported source-pattern hypothesis for the new full FFN, not an existing E140 exact-compile result.

With this selection, three quantizer streams directly write their1920B term slots at byte offsets0/1920/3840, and later hi/lo readers access those same initialized views. Total final term writes remain2,949,120B across512slices; the additional separate raw-term tensors and2,949,120B Subcopy read/write pass are absent. F32 restored-sum DM3.75MiB, BF16consumer distribution1.875MiB, residualVRF loads and nativeTRF2.8125MiB remain. This removes the initial note's deliberately conservative copy cost, without implying latency savings.

The independent gate must check literal indexed term offsets, Main1920B commit projection, hi/lo/Sub read bounds and no padded typedDMA, not merely allocation sizes. Whole register and arithmetic gates remain unchanged. Original `/tmp/moa-e140-down-local-design.md` is preserved as the preselection design rather than silently rewriting its cost assumption.
