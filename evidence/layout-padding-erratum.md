SDK layout terminology correction

`m![1 # N]` denotes one live element padded to physical extentN; it does not
meanN independent copies. `DummyN` is a distinct named axis used for explicit
replication. Earlier experiment-plan phrases such as replicatedCluster1#2 or
128KiB times512 physical copies were inaccurate descriptions. Measured kernel
latencies and exact-output checks remain valid. Cluster splitting activates both
clusters for independent request/query halves; distributedtile copies activate
more slices instead of funneling the fullpayload through one padded live slice.
No multiplied physical-byte estimate should be inferred from padding alone.
