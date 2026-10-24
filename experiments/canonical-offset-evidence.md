# Canonical whole-pool addressing gate

Commit b6e8127, Arena 317936 (root r183): ring IDs 0/31/63 and page IDs 0/127/255 passed exact reads and writes. The CPU oracle also checked all 134,217,728 BF16 ring values and all 67,108,864 page values, including untouched neighbors. This proves the public whole-pool plus byte-offset ABI; it does not yet prove model execution or admission.

Earlier dense and padded host-view variants are rejected: 317890 failed tile metadata validation; 317917 passed ring0 but failed BufferTooSmall at ring31. Padded kernel extent remains the full pool after the host window offset. The supported replacement passes the original unshifted pool and an explicit i32 byte address to device gather/scatter.

The canonical constructor allocates Rc64 sliding and Fc256 full pools directly, with no legacy Vec or mirrored pool. The default constructor remains unchanged. Scalar execution is the first implemented consumer; B16/P128 execution and admission are being wired before production activation.

Initial whole-model device build at 3aa5bb6 rejected scalar full-attention V gather: its destination had live Gf replication, but the one-address gather domain cannot implicitly replicate that axis. Repair stages the selected page across Tf/2 slices, then uses ordinary DM redistribution to the original Gf×Tf/32 layout, matching the already validated B16 whole-pool adapter. The gather/scatter indices and attention arithmetic remain unchanged.

Whole-model gates 932f5a0 passed on actual NPU: 317951/r197 at start509/count3 and317952/r198 at start896/count128. Both compare prompt, 48 interleaved singleton decode steps, C2→C1→C2, position0 reuse and forced scalar against identical arithmetic on separate legacy KV. Every logit was exact, including remote rings63/31 and full pages255 downward. There are no cache migrations in the canonical candidate.

R64 norm/media insertion gate6b5e0f8 passed317956/r200: all64 rows of all three helpers exactly matched four B16 controls across two seeds, zero/tiny rows, signed gates, and nonselected-row preservation for inserts0/31/32/63. R64 LM head adoption imports exact winner298e094 (9.852ms vsinitial17ms, root NPU gate) without changing embedding/extraction.
