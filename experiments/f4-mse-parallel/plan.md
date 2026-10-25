# Bounded parallel CPU MSE encoding

Base: clean q019 `13174d8b`, existing serial arithmetic from `8e7ce99e`.
Only add an optional CPU encoding API. Leave `b16_f4_weights::projection`,
Automatic selection, all device kernels, Cargo and D2H/upload sequencing unchanged.

Each contiguous 960-byte E4 group independently builds a histogram, chooses the
same minimum integer-SSE candidate with original tie priority, then packs codes.
Partition only complete blocked rows so each worker can invoke the unmodified
serial implementation. Scoped threads borrow input and return private output;
join in input order, adjust any NonFinite index to the original byte index, and
concatenate in that order. No shared mutable tensor or numeric reduction.

Worker count is min(available_parallelism, 4, row-block count), with one-worker
serial fallback. No request/model content controls worker count. Preserve the
original serial API and table initialization. Added cost: per-projection thread
creation and a bounded final output join, at most roughly one extra packed
projection in host memory. Measure that cost rather than assuming scalability.

Run pure-std standalone tests/benchmark against the original serial function:
full 15,360x3,840 and 3,840x15,360 deterministic provided-shaped finite E4 bytes,
all 254 finite encodings, signed zero, nearest-even ties, group/row permutations,
unaligned shard counts and deterministic earliest errors. Compare every packed
byte and scale bit. Separate processes report own /proc peak RSS. ABBA completed
CPU encoding includes thread creation/join and final assembly; input generation
and parity checks are outside timers. Record available CPUs/cgroup limits; use
one bounded two-CPU-affinity sensitivity run if local allocation exceeds two.

Local Docker currently reports 24 CPUs, cpu.max=max/100000, cpuset0-23 and no
memory quota. This is not evidence of the Arena server CPU quota. No NPU run or
startup speed claim follows from the local micro alone.
