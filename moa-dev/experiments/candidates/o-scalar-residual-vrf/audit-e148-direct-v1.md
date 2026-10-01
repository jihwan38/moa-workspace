# E148 independent actual graph gate

The frozen [audit](audit.py) reproduces [audit-e148-direct-v1.json](audit-e148-direct-v1.json) with `python3 audit.py e148-direct-v1 --check`. NPU correctness and complete-path performance remain separate gates.

Only `projection.rs` changes among the46 actual compile inputs. Its QKV prefix remains byte-identical to E147. In O, all four residual arithmetic expressions and coefficient constants are unchanged. Native60+60, shared TRF, Term/K16 reduction, direct disjoint raw writes, canonical gather, common/channel scale, RMS and residual suffix retain the original source, except removal of the obsolete coefficient DM-to-VRF load. The old E137-v3 O graph is used only after its production O image is verified identical to E147's O image.

Main O2 performs exponent mask → multiply2^-7 → local K256 Max → floor2^-126 → K16 Max. The finite-BF16 CPU proof exhausts65,280 input encodings and verifies the monotone scale identity. Sub O4/O58 use one shared16KiB bitmap with all256 destinations selecting leader `16*(slice//16)` in the same cluster. Source-valid scale indices0..7 and coefficient indices0..3 become32B VRF packets. The coefficient valid sequencer/printed fetch-size/padding equals the old compact reader; hidden physical bus-read granularity is not inferred.

Main O22/O30 preserve the exact original F32 reader and VE Mul/Add sequence, append WriteVrf, and remove their SRAM commits. Actual scheduled instructions93/106 occupy MainContext, SubContext and VectorEngine. Each1024B/slice output overlaps the prior1024B residual while computing, but all live VRF allocations, including later gamma/channel scales, remain disjoint within8KiB/slice. These direct stores remove two512KiB DM writes and subsequent two512KiB Sub reads; they add Main/Sub coupling and retain VRF writes.

Both weight DMA descriptions and native descriptions occur twice because the60-row stages share shapes. The audit therefore compares their separate IndexAccess windows at row0/60, rather than pretending description identity identifies a stage. Full weight bytes remain15MiB,256B packets; native output windows cover row0..119 once with8B pair commits. The complete post-native operation signatures match the original graph.

No hardware latency is inferred from eliminated bytes or static intervals. Exact target lowering and allocation checks do not prove arbitrary BF16/model correctness or hidden register timing; original standard and boundary inputs must run on this exact production image.
