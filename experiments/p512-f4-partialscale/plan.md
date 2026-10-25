# P512 fixed-F4 group960 partial scaling: resource gate

Base/current control: c079 `394d60a2`. The control is the complete
`ops_group_ffn_projection_storegroup::sliding_projection_owned` graph, including
current post-attention normalization/quantization, joined projection-owned FFN,
and post-FFN normalization/residual. No host dispatch changes.

Candidate keeps all token/output/K geometries, E4 activation quantization,
GeGLU and global-scale BF16 boundaries, HBM scratch and output ABIs unchanged.
Each current E4 weight group960 is statically approximated by fixed E2M1 F4
codes plus one power-of-two F32 scale (helper8bf9789c, CPU proofd5758321).
Joined UP/GATE retain one projection per cluster and eight P64 chunks. DOWN
retains token-cluster ownership and four P64 chunks per cluster.

Added per projection: 240 F32 group scales per PE, fixed-F4→E4 fetch lookup,
one local F32 group-scale multiply before the original H4/L16 inter-slice Add,
and a real reduced-F32 DM boundary before the original xscale/rowscale/BF16
epilogue. The extra DM boundary is required by SDK sequencing constraints;
no intra→inter→intra single pipeline is assumed. Weight delivery halves, but
lookup and partial liveness can erase the benefit and are included in the gate.

UP/GATE scales: `[FfnProjection,L/240,H/960,L%240]`.
DOWN scales: `[H/240,L/960,H%240]`. Existing row dequantizers remain separate.
This changes weight quantization and local scaling placement; it is not claimed
bitexact to the current E4 model. No A4 activations or geometry sweep.

Gate: same-tree complete composed control/candidate resource schedules,
at least10% estimated reduction with no DramStack. Arithmetic/source audit and
B16 complete intended-reference micro must pass before any P512 NPU/host work.
Actual complete timing and fresh model/task quality are separate later gates.
