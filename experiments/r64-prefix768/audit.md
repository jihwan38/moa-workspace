Read-only mapping review by batch_kernels: the partial join preserves the named
Ts plane order. Half stores Ts/32%16 in slices and Ts/512 as two local elements.
The512-key result fills high=0. The256-key result fills high=1 only for low0..7.
Narrowing to %8 with physical#32 preserves every Ns stride32; restoring Half
preserves zero-initialized high=1/low8..15. The subsequent actual to_dm moves
high=1 into original physical planes16..31. Thus planes16..23 contain the tail
and24..31 remain zero before the original32-way F32 reduction. The memset covers
both local bands for all16 live low planes. Random values vary across every
request, Ns, group, key and component in the hardware/scalar probe.

Host cargo check succeeded. Device compile5 succeeded. No NPU execution or host
adoption has occurred at this checkpoint. Probe poisons every future key from
position+1 through1023, not the former prefix512 range, so valid512..767 history
is retained. Original scalar references and full-window controls remain intact.
