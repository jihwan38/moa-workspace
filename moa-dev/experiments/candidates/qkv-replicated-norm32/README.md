# qkv-replicated-norm32

Independent exact QKV compile: 56719 static cycles, previous two-cluster ring16 61994. Hardware result pending. Only the QKV function body changes.

Ring32 uses eight anchor copies per cluster and a 32-slice broadcast group. The replicated-normalization variants read the same hidden input into both clusters and independently execute the unchanged generic RMSNorm in each cluster, so the normalized BF16 vector is already local to both clusters before replication. All original RMSNorm arithmetic, epsilon, BF16 output boundary and projection operations remain. Extra computation is intentional to avoid cross-cluster transfer; exact hardware correctness must still be checked.

The ops.rs file includes its previous surrounding experimental state, including FFN hierarchical replication. Apply the incremental patch or merge the QKV body instead of replacing unrelated current changes.
