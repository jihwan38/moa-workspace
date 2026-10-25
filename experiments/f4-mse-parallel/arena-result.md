# Actual Arena CPU result

Root-dispatched r1292/job321485 passed the standalone full gate. Exact source
`40864041`, ELF SHA256
`89ec00db8f0814b3bd61c6f0b4551a6a586c97f63638c79a59a1a3efe2b13027`.
All117,964,800 packed bytes and245,760 scale values compared bitwise pass;
all8 timed outputs have the same checksum.

Actual environment: available_parallelism=8, selected workers=4,
cpu.max=800000/100000, cpuset/affinity0-63. This gate therefore exercises a real
eight-CPU quota, unlike the unlimited local container.

Completed serial CPU calls:131.536/109.309/109.250/109.663ms.
Completed parallel calls:70.908/61.393/61.334/87.053ms.
Means114.940→70.172ms,1.638x or38.95% lower. Input generation and comparisons
are outside timers; each timed call includes allocation/thread spawn/join and
ordered result assembly. All measurements are retained, including the last
parallel outlier. No per48-layer startup or serving gain is inferred here.

Actual peak VmHWM is204,100KiB, while final VmRSS is118,932KiB. Both include the
parity process; the final RSS is not the peak and neither is an encoder-only
measurement. The local separate-process +28.45MiB allocation result remains
separately labeled.

Root authorized a separate q020 caller adoption after this gate. Keep original
q019 frozen. Actual48-layer startup, all124 quality fixtures and longer HTTP
gates remain required; D2H/upload ordering and device math must remain unchanged.
