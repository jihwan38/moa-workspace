# Canonical whole-pool addressing gate

Commit b6e8127, Arena 317936 (root r183): ring IDs 0/31/63 and page IDs 0/127/255 passed exact reads and writes. The CPU oracle also checked all 134,217,728 BF16 ring values and all 67,108,864 page values, including untouched neighbors. This proves the public whole-pool plus byte-offset ABI; it does not yet prove model execution or admission.

Earlier dense and padded host-view variants are rejected: 317890 failed tile metadata validation; 317917 passed ring0 but failed BufferTooSmall at ring31. Padded kernel extent remains the full pool after the host window offset. The supported replacement passes the original unshifted pool and an explicit i32 byte address to device gather/scatter.

The canonical constructor allocates Rc64 sliding and Fc256 full pools directly, with no legacy Vec or mirrored pool. The default constructor remains unchanged. Scalar execution is the first implemented consumer; B16/P128 execution and admission are being wired before production activation.
