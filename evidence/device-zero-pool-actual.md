Actual NPU zero/ownership gates passed at source004c966 (kernel) and4c76894
(strengthened probe): Arena318021/r252 and318022/r253.

Every134217728 sliding BF16 value and67108864 full BF16 value had rawbits0.
Retaining the actual80 sliding+16 full pools (22GiB) took46.098/46.101ms total
allocation+initialization. The strengthened run wrote nonzero markers into the
first live allocation's last ring/page, read them back exactly, and verified
that the final live allocations remained allzero. First sliding/full kernel
launches including initial binding were1.167/1.085ms in r252.

Parent startup profiling of the old path (318023) reported tokenizer1.159s,
weights36.402s, then exceeded70s while still constructing Workspace. The new
constructor keeps the exact same canonical shapes/finitezero contents and only
replaces four HostTensorzero/to_hbm callsites with returned-device allocations.
Legacy/reference storage paths remain unchanged.
