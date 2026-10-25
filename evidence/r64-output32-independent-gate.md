# R64 output32 independent-buffer gate

The initial R64 output32 differential reused output storage, which could conceal a missing write. Source2a02376 strengthens the probe only: candidate partial/final tensors are separately allocated and initialized to NaN for each seed, while control tensors use separate storage. All values must be finite and bit-identical. Production kernel f73da1d is unchanged.

Arena318506 (`r488-r64-o32-independent-submit`) passed three seeds, every sliding/full K1024 partial and every final output. Sliding candidate0.1185–0.1192ms versus0.1369ms control; full0.1695–0.1706ms versus0.1863ms. Twenty completed calls per timing block; timings include the complete kernel. This closes the missing-store false-positive gap. Whole-model and HTTP evidence are separately recorded for the integrated candidates.
