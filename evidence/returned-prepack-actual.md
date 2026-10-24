Returned allocation kernel source0c4c6da and probee132abb passed actual Arena318033,
run r267. Both up/down shapes and two distinct seeds matched every row-major FP8
byte, F32 dequant-scale bit and blocked FP8 byte against the unchanged original
kernels. Retaining the previous returned allocation while processing the next
seed preserved every byte. This includes signedzero input rows and varied E4
block scales. No kernel arithmetic or physical blocked layout changed.

The loader now uses returned prepack/reblock buffers and drops each temporary
row-major tensor immediately after reblocking. It retains the same one blocked
FP8 weight set plus per-output scales. Original FP4 tensors remain untouched.
Removed work is16,986,931,200 HostTensorzero/PDMA bytes across48layers; original
checkpoint loading and actual quantization/reblocking still execute normally.
Actual full-server startup improvement remains a separate parent-run gate.

Measured per-projection allocation+prepack+reblock: up old173.492/195.627ms
versus returned0.604/0.367ms; down old172.984/173.158ms versus0.931/0.708ms.
These timings include old hostzero/PDMA but exclude input upload and output
verification reads in both paths. They are startup microbenchmarks.
