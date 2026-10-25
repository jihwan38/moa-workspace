# Provided-weight vision B16 probe

Base API1818875+a8c1b98. Source ownership is this probe only; root owns device kernels and integration. The executable mmaps the provided model file via load::model_dir(), but uploads only nine vision tensors; it never constructs the12B language model. Position vectors reproduce runtime::gather_pos_embedding, including clamp and BF16(col+row).

The original path is patch LN→nine K768 BF16 partials→balanced BF16 tree→BF16 bias→LN2; BF16 positional residual→LN3; unweighted RMS→five K768 BF16 partials→balanced BF16 tree. Candidate fused output runs first into independent NaN buffers, followed by separately poisoned three-stage B16, then sixteen original scalar paths. Every stage and final BF16 bit is compared, including zero padding. Four cases cover zero/constant/random/sharp rows, affine row permutation, active7 and active1, both clusters, and clamped position endpoints. Separate normalized-patch and text-RMS diagnostics compare scalar/generic norm outputs bitwise.

Independent CPU projections use the observed normalized BF16 inputs and actual dense weights. Each K768 F64 sum is cast to BF16, then the exact original balanced BF16 tree/bias is applied. The nearest-F64 bit difference count is reported. The fixed independent acceptance bound uses gamma768=(768*2^-24)/(1-768*2^-24) times sum(abs(products)), with1e-36 absolute flush allowance, through monotone BF16 rounding at every partial/tree/bias boundary. This accounts for F32 accumulation rather than claiming hardware accumulates F64. Interval failures must be zero; all scalar/candidate gates separately require zero bit differences.

Kernel-only timing uses preallocated/uploaded buffers and completed launches, ABCCBA variants scalar16/stagedB16/fusedB16, five repetitions per entry. Inclusive timing includes CPU position gathering, input uploads, output zero allocations/uploads and completed launches; three repetitions per entry. It excludes weight startup, downloads and prompt scatter for both paths. Trace labels scalar_one_patch separately from batch16; no extrapolated16-patch hardware trace is claimed.

Common indexed row-copy safety gate uses independent B16/P128 sources and P128 destinations. Source rows0/7/last and destination0/63/127 are crossed; every destination element is compared after each of18 writes, including unchanged rows. Source tensors are checked unchanged. No scatter timing claim.

Build and NPU execution are separate gates. Root alone dispatches hardware. No result is asserted here before actual logs.
