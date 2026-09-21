# FP4 decode fullshape: replicated input 단계 동결

SDK0.8.1에서 두 host build 모두 성공했다(compile2회, 실패0). NPU 미실행이며 root 요청에 따라 서로 다른 cluster/slice 입력을 갖는 [후속 tagged probe](../fp4-decode-probe-fullshape-tagged/README.md)를 별도 제출했다. 후속 job70658은 두 경로 exact PASS지만 이 replicated-input binary 자체는 NPU 미실행이다. 이 폴더의 source와 package는 보존한다.

이 진단은 통과한 작은 probe(job70620)의 두 변환 pipeline을 그대로 유지하고 DM shape만 **2clusters ×240live/256physical slices ×32rows ×3840columns**로 확대한다. 입력 `[row,k]`를 모든480logical slices에 복제하므로 slice/cluster 간 순서 교환 오류는 이 단계만으로 구별하지 못한다. 출력은 typed DMA가 유효240slice를 모은 `[cluster,slice,row,k]`이다. padding을 유효 데이터로 취급하지 않는다.

case1은 모든256 ordered nibble pairs와16 FP4 codes, 양쪽 nibble, 각mod64위치의code coverage를 검사한다. CPU oracle은 독립 BF16 bit상수표이며 signed zero를 구별한다. input HBM roundtrip, NaN초기화output, legacy/fused각58,982,400개exact검사와cluster/code/nibble/mod64별오류진단을 구현했다.

출력 하나117,964,800B(112.5MiB), 두 출력 총235,929,600B다. legacy검사뒤HBM출력을drop한후fused를할당한다. `--host-memory-only`는 Device생성없이같은크기의sentinel/HostTensor변환/두전체검사를실행해1.11345초,최대RSS234,560KiB를기록했다. 이것은Arena호스트/NPU/PDMA의70초내완료나최대메모리를보장하는측정은아니다.

동결package: `../../artifacts/fp4-decode-probe-fullshape-package/`, fixture불필요. binary SHA256 `f080a6a47e31f29764ddc0c5a55cfe670f0e3c99d7787ab0f478186aebba658e`,2,228,392B. ELF등록두deviceimages는컴파일된kernel파일hash와일치한다. 상세hash/shape/CPU측정은`evidence.json`,`registry-summary.json`,`compiled-input-hashes.json`에있다.

재현은E37원본snapshot을별도scratch에복사한후새`src/decode_probe.rs`,`src/bin/probe_decode.rs`및`lib.patch`만추가한다. 기존공식kernels와test_kernels는그대로다.

```bash
cd /tmp/moa-fp4-decode-probe-fullshape
export CARGO_TARGET_DIR=$PWD/target
export FURIOSA_OPT_OUT_DIR=$PWD/target/furiosa-opt/kernel
cargo furiosa-opt build --release --locked --bin probe_decode
./target/release/probe_decode --oracle-only
./target/release/probe_decode --host-memory-only
```

이probe는DPE/scale/runtimequantizer/전체FFNbufferlifetime을포함하지않는다. 공식score나성능결과가아니며source/Git/NPU제출을변경하지않았다.2026-09-21확인.
