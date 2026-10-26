# Per960 activation foundation checkpoint

Plan `3093b1c8`; functional source `55a55affabad9ab69ada2fafcef62fc1272edbe0`.
Independent e2e source/fixture/probe review passed. Plain host check passed10.74s.
The two resource compiles used fresh package tracking and distinct empty public
compiler output directories on the exclusively owned tail256 target. All416
source/dependency inputs match each captured run; original dependency files and
whole-row quantizer/packer/projection/oracle files are unchanged.

| Complete static graph | Cycles | Prior whole-row context | Main added costs |
|---|---:|---:|---|
| H4 |60292|23918|offset and scale stores17823each; gather7886|
| L16 |74636|45745|offset and scale stores8457each; gather28521|

Both compile without DramStack. The explicit HBM scratch owners are8192/32768B.
Placed live SRAM address-union peaks are11804672/47218688B; largest tensors are
7864320/31457280B I32 codes. One unplaced index-access alias remains per graph;
full allocation records are retained. These are compiler allocations, not runtime
allocator measurements. H retains real H16 packed movement; L has no packed-DM
join because its natural grouped map already matches the16row store ABI.

The probe source checks every code/packed byte/group scale/relative offset, true
row/group/channel permutations, reuse and input/table/shift immutability. It was
not built as a standalone ELF and never executed. Grouped scales change the
quantization contract; the previous whole-row schedule is a cost context only.
There is no grouped correctness, speed, full-FFN or model-quality result.
Original complete FFN323067 performance HOLD and large matched synthetic drift
are preserved. No serving source or final candidate changed.

Replay preserved evidence without compiling or executing devices:
`python3 experiments/int4-grouped-activation-quant/analyze_resource.py`
