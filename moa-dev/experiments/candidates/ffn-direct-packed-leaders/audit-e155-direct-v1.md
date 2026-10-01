# E155 independent compiled gate

Whole and raw graph review scripts pass with `--check`. All46 compile inputs are bound to adopted E154; only MLP changes and45 inputs remain identical. The reproduced two-part source patch changes the private broadcast helper input to CL0 and removes its prior two-cluster H480 packed intermediate. It does not change RMS, Max/scale, term arithmetic, bounded copies, ring or native/down arithmetic.

The actual direct seed descriptor has384packets of480B. Each reads initialized source `(cluster0, shardq, termt*480 .. (t+1)*480)` within a1440B packed slice and writes `(clusterc, leader32g, termt*3840+q*480)` for all `c<2,g<8,t<3,q<8`. The184320B total covers all16leaders without gaps or overlap. CL1 receives92160B fromCL0; removing23040B intermediate materialization therefore also increases cross-cluster logical payload by80640B. This is not a predicted cycle saving.

Actual Max ratio256 with8valid output replicas, direct scalar VRF,512-consumer restore, three480B Sub copies and ring32 bitmap bounds remain valid. Original four whole native streams/five down-scale streams preserve descriptors and arithmetic. All VRF lifetimes fit8KiB per slice without live overlap; registered inputs outlive their consumers. QKV/O production images remain E154-identical.

The separately compiled original raw-upgate wrapper binds the full new MLP and checks the same changed direct seed route with two native streams. It excludes preRMS. These structural checks do not inherit an NPU result from E154 or explain M06's prior timeouts; hardware correctness and complete-path timing remain separate.
