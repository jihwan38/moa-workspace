E55의 K/V weight를 공개 `begin_interleaved`로 하나의 FP8→BF16 lookup·contraction stream에 넣는 후보다. **첫 exact compile이 native contraction lowering의 축 불일치로 실패해 동결했다.** Schedule, graph, device image는 생성되지 않았고 성능·NPU 정확성은 측정하지 않았다. E55의 기존 K/V 분리 구현을 유지한다.

기준은 adopted E55 `d50eb159d9a260c411f2ebdc96adc5e0dc82afcb`와 동일한 [frozen crate](../../artifacts/qkv-bf16-fused-channel-scale-e54/crate)다. 기존 후보에서 같은 K/V single-stream 시도는 찾지 못했다. [Shared TRF](../qkv-bf16-shared-trf-e55/README.md)는 activation TRF만 공유하는 다른 변경이며, [table 공유 API 감사](../../../../docs/research/fp8-lookup-table-sharing-audit.md)는 prepared-table handle 유무를 조사했다.

## 구현과 보존 범위

[projection.rs](projection.rs)만 변경했다. `project_key_value`는 기존 두 HBM→DM weight 전송과 channel-scale 전송을 유지하며, matrix selector `Dummy2`의 0=K, 1=V를 추가한다. 두 scale DM을 Sub `begin_interleaved` → FP32 fetch → 기존 inverse row Switch로 읽어 `VrfTensor<…, m![Dummy2, Ps/8%4#8, 1#8]>`에 둔다. 두 weight DM은 Main `begin_interleaved`에서 하나의 `fetch_table_lookup::<bf16>()`와 BF16 DPE로 이어진다.

Contraction의 Time은 `[Dummy2, Ps/8%4, H/32]`에서 `[Dummy2, Ps/8%4]`로 줄인다. **H만 합산하며 K와 V는 합산하지 않는다.** 완성된 F32 dot에 해당 matrix/row의 FP32 scale을 곱한 뒤 첫 BF16 cast를 한다. 기존 네 행 commit을 유지해 `[Dummy2, Ps/8%4]` BF16 DM에 쓴다. `tile::<m![Dummy2],1,m![1#2,Ps/8%4]>(i).unpad::<m![Ps/8%4]>()`로 각각의 view를 얻고, `finish_one_kv_matrix`가 원래 BF16 row Switch와 typed head gather를 그대로 실행한다.

Source의 Q prefix/O suffix, 다른42개 source, Cargo 입력3개가 기준과 byte-identical이다. `ops::sliding_project_qkv`와 공식 harness도 변경하지 않았다. 따라서 source상 후속 K head RMS/gamma/RoPE → `k_cache`와 V head RMS → `v_cache`, 동일 `kv_offset` 호출은 기존 경로다. 공개 entry-point signature를 바꾸지 않았으며 새 fixture 분기는 없다.

Root가 독립 작성한 [verify_layout.py](verify_layout.py)의 [결과](coordinate-verification.json)는 두 matrix의 전체4,096 output 좌표와15,728,640 weight 원소를 연속 H 구간으로 검사한다. 행 `p=1024*c+32*g+8*r+s`는 raw slice `8*g+s`, local row `r<4`다. Scale VRF scalar byte 주소는 `256*i+32*r`, BF16 commit은 `8*i+2*r`, split base는 `8*i`다. VRF512B/slice 중 유효 scalar8개만 대응하며 row padding을 유효 값으로 바꾸지 않는다. **이 검사는 의도한 source 좌표 모델이다. Lowered graph 검증이나 hardware 실행이 아니다.**

## 확인된 compiler 제약

[실패 로그](../../artifacts/qkv-bf16-interleaved-kv-e55/compile-attempt1.log)는 Rust source 검사 후 다음 native 오류를 낸다.

```text
visa: while lowering TensorUnit
contraction: input does not match pipeline
kernel-declared input shape does not match the shape the pipeline delivers.
declared: … InSlice: [Dummy2_1=2, Ps_8=4, H_1=3840] … (1)
pipeline: … InSlice: [interleave_1=2, Ps_8=4, H_1=3840] … (16)
```

실제 설치 SDK0.8.1의 공개 `context.rs:95–111`은 `begin_interleaved<I>`가 `Symbol<I>`를 반환하도록 선언한다. 그러나 이 composition의 native pipeline에는 `Dummy2`와 다른 `interleave` 축이 나타나 contraction 경계의 동일성 검사를 통과하지 못했다. Backend 내부 alias 구현의 원인을 공개 소스에서 확인하지 못했으므로 이름 불일치 관측 이상으로 원인을 확정하지 않는다.

`furiosa-mapping-types-0.8.1/src/lib.rs:39–57`의 `Ident::new`는 첫 글자를 대문자 ASCII로 제한하고, `furiosa-mapping-macro-0.8.1/src/lib.rs:105–106`의 `axes!` 확장은 `Ident::new(stringify!(name))`를 쓴다. 따라서 `axes![interleave=2]`로 내부 축 이름을 맞추는 것은 지원되는 수정이 아니다. 공개 TuTensor stream rename/reshape도 찾지 못했다. Private 구조를 위조하거나 padding을 재해석하지 않았고, 근거 없는 두 번째 compile은 하지 않았다.

결과 split에는 SDK `tensor/memory.rs:1217–1238`의 공개 `unpad`를 사용했다. `tensor/view.rs:369–381`은 outermost padding 제거가 주소를 보존한다고 명시한다. 반면 같은 파일의 `reshape:345–346`은 physical size equality를 assert하므로 `1#2`를 단순 unsafe reshape로 삭제하는 방식은 쓰지 않았다. 조사한 SDK8개 파일의 실제 SHA는 [evidence.json](evidence.json)에 있다.

이 실패는 **해당 SDK 버전에서 이 K/V interleaved contraction composition이 막힌 것**이다. 모든 `begin_interleaved` 사용 또는 하드웨어의 K/V 결합이 불가능하다는 증거는 아니다. 재개 근거는 공개 `I` 축을 이 contraction 경로까지 올바르게 보존하는 공식 compiler 지원·수정 증거다.

## 비용 판단과 재현

기준 E55 QKV의 실제 전체 정적 makespan은51,293이다. 기존 K/V table 준비는 각각 DMA838 + Sub1,293 cycles다. 이 source는 lookup 호출을 하나로 줄이려 했지만 compile 실패로 실제 table node 수 감소조차 확인하지 못했다. 후보 makespan은 **미측정**이며, 준비 작업2,131 cycles를 전체 시간에서 빼지 않는다.

기준 schedule에서 K weight 완료32,494 → dot35,157, V weight 완료43,041 → dot45,704다. K 후단 RMS/RoPE Main2,911 cycles는 V weight DMA와 겹친다. 결합 op는 두 weight와 두 scale을 모두 요구하므로 K를 일찍 소비하던 overlap을 잃을 수 있다. 기존 두 FP8 DM의 allocation 일부가 재사용되므로 결합하면 동시 lifetime과 주소 배치도 달라진다. Table 준비 감소만으로 개선을 예고하지 않는다.

첫 실행은 새 Docker `/tmp/moa-qkv-bf16-interleaved-kv-e55`와 존재하지 않던 전용 target을 사용했다. Cargo cache/toolchain은 설치된 것을 재사용했다.

```bash
cd /tmp/moa-qkv-bf16-interleaved-kv-e55
CARGO_HOME=/home/moa/.cargo \
CARGO_TARGET_DIR=/tmp/moa-qkv-bf16-interleaved-kv-e55/target \
FURIOSA_OPT_OUT_DIR=/tmp/moa-qkv-bf16-interleaved-kv-e55/target/furiosa-opt/kernel \
cargo furiosa-opt compile ops::sliding_project_qkv --exact \
  --dump-schedule candidate.json --dump-graph candidate-graph.json
```

[prepare_source.py](prepare_source.py)는 frozen E55 crate에서 source/patch를 재생성하며 [reproduce.sh](reproduce.sh)는 같은 실패를 재현하는 명령이다. [실제 Docker 입력46개 해시](compiled-source-manifest.json)를 compile 전후 대조했다. Lock SHA는 `aaff8dd41f9c643dd8c9faa358b9bc780e88a42c09172c1a5d7a0726d07efe09`로 유지됐다. Toolchain은 cargo-furiosa-opt0.8.1, rustc1.97.0-nightly(f53b654a8)다.

- 후보 source SHA: `8428c339a053c3835862f6e68a544a24c220f241b2469e5b7a20f6c01f143fdc`
- 실패 로그 SHA: `65385a35ef55d999307856cc1ca803375199380b096b94a0f02687e00d0ecf81`
- [Evidence](evidence.json), [manifest](manifest.json), [patch](candidate.patch)

Exact compile1회만 실행했다. Full host build, NPU, live source/target, Git 변경은 없다.
