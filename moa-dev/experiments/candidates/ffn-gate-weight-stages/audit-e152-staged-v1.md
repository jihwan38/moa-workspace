# E152 actual compact gate-half gate

`audit.py e152-staged-v1 --check` and the thin `verify_graph.py` adapter pass. The E150 source changes only the gate weight/helper path. Original up, both quantizers, gate scale/reduction order, GeGLU and adopted E150 down remain unchanged;45 other compile inputs are byte-identical.

Each gate HBM half selects15 rows using the original30-row pitch. Actual descriptors emit15360×960B per half, totaling the original29491200B once. Each slice receives28800B FP4. Native F32 writes are900×16B to the original28800B partial allocation at0/14400, with both writes joined before the five original6-row consumers. The consumer rows12–17 correctly cross the two halves. Gate arithmetic differs only in row extent; scale multiplication and F32 reduction order are unchanged.

Up and both gate halves read one emitted shared activation TRF. All actual VRF/TRF consumers fit allocation lifetimes and all live VRF buffers fit8KiB without overlap. The unchanged E150 two-stage weighted down also matches emitted arithmetic. Native lookup count increases4→5; no extra packed/partial assembly is introduced.

Static schedule only: first gate DMA2603–17590, initial canonical input gather17590–18550, second gate DMA24914–39901. The intended transfer interleaving is present. This is not a hardware latency or whole-gain prediction; extra lookup/setup and changed context scheduling remain real costs. Finite production up/gate oracle and contemporary whole measurement are separate requirements.
