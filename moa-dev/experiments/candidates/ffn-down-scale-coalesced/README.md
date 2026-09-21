# Down local-scale 연속 HBM staging

E37 safe BF16 FFN에서 **down scale의 HBM 배치만** 바꾼 후보다. **E45 job70641은 공식15개 비교를 모두 통과했지만 FFN394,044cycles로 느려져 미채택했다.** [E37 반복 중앙값379,741](../../../../docs/experiments/e37-combined-repeats.json)보다3.77% 높다. E45는 별도 job 한 번의3-input 중앙값이며 추가 입력이나 반복 측정은 하지 않았다.

Exact compile의 전체 정적 makespan은 **162,209 → 161,517 cycles(-0.43%)**였지만, scale 전송 합은 **9,774 → 11,672(+19.42%)**, 전체 DMA 점유는 **135,607 → 137,505**로 늘었다. HBM의 짧은 분할을 SRAM scatter로 옮긴 효과와 추가 전송 비용이 함께 있다. DMA 증가와 실측 회귀의 방향은 일치하지만, 이 비교만으로 실제 회귀 원인을 전부 규명한 것은 아니다.

[mlp.rs](mlp.rs)는 SHA256 `0315d47ff355c1ce9becfa943ea9bf04bd49e44d3874a24f4e7058f77c0431b3`로 동결했다. [원본](mlp.baseline.rs), [3줄 patch](candidate.patch), [정적 근거와 파일 해시](evidence.json), [좌표 검사](verify_layout.py)를 함께 보관한다. 원본 exact device image는 E37 FFN image `61ad0c50a03902de9208447951086df930669ded080c94e4c637bffac3c414b6`와 byte-identical하다. 후보 image는 `78869dc870464afebd1208dd3b85add2984195e8277d8ec35be50d3c099fd310`이다.

[E45 측정 기록](../../../../docs/experiments/e45-down-scale-coalesced.json)의 QKV/O/FFN 중앙값은94,492 /45,842 /394,044다. 공식 fixture의15개 비교(QKV9, O3, FFN3)만 PASS 범위에 해당한다. E45 snapshot MLP가 이 후보와 동일하고, package host ELF `7d38c2cf102ef4836e57a46646289c0397e371166ccefd257da5f1e9be90ed21`에서 위 exact FFN image 전체 byte가 한 번 존재함을 별도로 확인했다. [실측 로그](../../../logs/e45-down-scale-coalesced-job-70641.log)를 보존하며 소스와 static dump는 바꾸지 않았다.

## 실제 경로와 변경 범위

E37의 [ops.rs](../../artifacts/e37-kernel-specific-rms/src/ops.rs) `decoder_feedforward`(203행)는 pre-FF RMS → `feedforward_sharded` → post-FF RMS → residual/layer gate → HBM write를 실행한다. `feedforward_sharded`(MLP 167행)는 up/gate의 FP4→FP8 DM→BF16 LUT DM, BF16 blockdot/FP32 scale·sum, GeGLU를 거쳐 `project_down_sharded`(373행)를 부른다.

새 코드는 MLP 389–394행에만 있다. 기존 `HbmTensorView<Dummy2,H%1920,L/16>`를 바로 8-column layout으로 읽던 경로 사이에 `DmTensor<DownCluster,DownRows,[H%60,L/16]>`를 넣었다. `DownRows=[H/60%32,1#8]`의 각 row group에서 column-zero slice 하나가 60행×960개 scale을 연속 수신한 뒤, typed `to_dm`으로 기존 `DownRowsByColumns=[H/60%32,L/1920]`에 분배한다.

SDK 0.8.1 `src/tensor/memory.rs`의 `HbmTensorView::to_dm`(612행)은 `assert_dma_layout` 후 `read().transpose(true)`, `DmTensor::to_dm`(957행)은 `assert_dm_dma_layout` 후 `transpose(true)`로 lowering된다. 새 reshape, arithmetic, cast, LUT, branch는 없다. 12행 tile, 16-element dot, FP32 scale·local sum·8-column sum, BF16 cast/4값 commit 경계를 모두 그대로 둔다. 좌표의 수학적 보존과 compiler의 실제 DMA 실행 정확성은 별도 검증 대상이다.

## DMA descriptor에서 확인한 병목

원본 [schedule](../../artifacts/ffn-down-scale-coalesced/baseline.json)과 [graph](../../artifacts/ffn-down-scale-coalesced/baseline-graph.json)를 함께 추적했다. 아래 graph 번호는 `operators` key이며 schedule instruction 번호와 다르다. `axis=count:stride`의 stride와 끝 packet은 byte 단위 descriptor다. 이 정보로 실제 HBM transaction 수나 bank 충돌 횟수를 단정하지 않는다.

| 원본 경로 | Descriptor / 데이터 배치 | 정적 cycles |
| --- | --- | ---: |
| Up/gate packed weight | 각각 32행×1,920B=61,440B 연속, `None=240:256`로 묶인 segment. 각각 10개의 src descriptor | 각각 26,475 |
| Down packed weight, graph148 | `H_1=60:7680`, `L_1920=8:960`, 끝 960B. 전체 7,680B HBM row를 8개 column으로 나누어 SRAM에서는 row stride960B로 저장. 8 src descriptor, source stride0 복제 없음 | 29,427 |
| Up/gate local scales | 각각 32행×240B=7,680B 연속, `None=30:256`. 각각 10 src descriptor | 각각 3,789 |
| Down local scales, graph179 | `H_1=60:960`, `L_1920=8:120`, 끝 120B. row마다 8개의 120B 논리 segment, 전체30,720개. 8 src descriptor | 9,774 |
| Down activation, graph185 | SRAM→SRAM 32 src descriptor. 16개×240B source chunk를 1,920 BF16 segment로 모으고, source stride0의 row/cluster 복제를 실행 | 9,392 |

Down activation의 원래 논리 데이터는 `15360×2=30,720B`, 목적지는 `512×1920×2=1,966,080B`로 64배 복제된다. Source tensor의 padded allocation122,880B를 유효 입력 byte 수로 세면 안 된다. 이번 후보는 이 복제나 activation 배치를 건드리지 않는다. 원본 weight/scale HBM 작업 합99,729와 activation DMA 합17,744는 [기존 전체 병목 분석](../ffn-parallel-epilogue/remaining-bottlenecks.md)과 일치한다.

후보 [graph179](../../artifacts/ffn-down-scale-coalesced/candidate-graph.json)의 HBM→DM descriptor는 `None=225:256`으로 바뀐다. 한 row group의60×960=57,600B가 연속이다. Graph187의 새 SRAM scatter는 다시 `H_1=60:960`, `L_1920=8:120`, 끝120B를 사용한다. 따라서 HBM segment는 길어졌지만 기존 분할 전송을 없앤 것은 아니다. FP4 weight까지 같은 representative-slice staging을 적용하면 slice당460,800B를 먼저 보유해야 해512KiB DM 중 다른 동시 live buffer의 여유가 작다. 이 weight 변형은 구현·컴파일하지 않았으며, 실제 allocator 실패를 확인했다는 뜻도 아니다.

## 같은 compiler에서의 정적 비교

| 측정 | E37 exact | 후보 exact | 차이 |
| --- | ---: | ---: | ---: |
| Makespan | 162,209 | 161,517 | -692 |
| Instruction 수 | 242 | 243 | +1 |
| Down scale HBM DMA | 9,774 | 3,557 | -6,217 |
| 새 scale SRAM scatter | 0 | 8,115 | +8,115 |
| Scale 전송 합 | 9,774 | 11,672 | +1,898 |
| 전체 DMA interval union | 135,607 | 137,505 | +1,898 |
| Main interval union | 125,274 | 125,274 | 0 |
| Sub interval union | 19,378 | 19,378 | 0 |
| Vector interval union | 34,748 | 34,748 | 0 |
| Main∩DMA | 104,921 | 107,459 | +2,538 |

Main/Sub/Vector instruction duration의 multiset도 각각 동일하다. 메모리 주소와 instruction 배치가 바뀌므로 이 사실이 머신 코드 전체 동일성을 뜻하지는 않는다. Down scale HBM은 `[118130,127904)`에서 `[118060,121617)`로 이동했고, activation DMA는 `[127904,137296)`에서 `[125246,134638)`로 앞당겨졌다. 새 scatter는 `[134638,142753)`다. Down BF16 decode는 후보143,509에 끝나며 원본144,122보다613 빠르다. 작은 makespan 감소는 전체 scheduling·overlap 변화까지 포함한 값이다.

중간 scale tensor는64개 유효 slice×57,600B=3,686,400B를 보유한다. Schedule tensor201의 padded physical extent는29,491,200B, lifetime은 `[118060,142753)`, physical page list는 `[11,12]`다. 유효 byte 수, tensor extent, allocator page reservation과 실제 bus 전송량은 서로 다른 값이다. Compile 통과는 이 후보의 allocation이 허용됨을 보이지만 hardware correctness를 보장하지 않는다.

## 좌표 보존과 재현

`h∈[0,3840)`, 16-element block `b∈[0,960)`에서 HBM offset은 `960h+b`다. `c=h//1920`, `g=(h%1920)//60`, `r=h%60`, `j=b//120`, `t=b%120`로 두면 원래 목적지는 `(cluster=c,slice=8g+j,element=120r+t)`다. 새 중간 주소 `(c,8g,960r+b)`에서 typed scatter한 결과가 이 목적지와 일치한다. 중간 tensor의 `1#8`에서 padding slice를 유효 데이터로 재해석하지 않는다. [verify_layout.py](verify_layout.py)는3,686,400개 모든 좌표의 HBM offset 복원, 목적지 동일성, 중복 없음과 전체 coverage를 검사한다. 이 CPU 검사는 SDK/NPU를 실행하지 않는다.

Docker `moa2026-dev`의 `/tmp/moa-ffn-down-scale-coalesced`에 E37 frozen source 전체를 복사하고 MLP만 바꿨다. 다른 독립 scratch의 compiler cache를 사용했으며 main `/workspace/target`은 사용하지 않았다. 환경은 Rust `1.97.0-nightly (f53b654a8 2026-04-30)`, Cargo `1.97.0-nightly (eb9b60f1f 2026-04-24)`, `cargo-furiosa-opt 0.8.1`이다. 완전 빈 target 재현 실험은 기존 [E37 clean-build audit](../../../../docs/experiments/e37-clean-build-audit.md)에 분리되어 있다.

```bash
export CARGO_HOME=/home/moa/.cargo
export CARGO_TARGET_DIR=/tmp/moa-ffn-down-scale-coalesced/target
export FURIOSA_OPT_OUT_DIR=/tmp/moa-ffn-down-scale-coalesced/target/furiosa-opt/kernel
export CARGO_NET_OFFLINE=true
cd /tmp/moa-ffn-down-scale-coalesced
# src/device/shared/mlp.rs = frozen E37 baseline
cargo furiosa-opt compile ops::decoder_feedforward --exact \
  --dump-schedule schedules/baseline.json --dump-graph schedules/baseline-graph.json
# src/device/shared/mlp.rs = frozen candidate; touch it after replacing
cargo furiosa-opt compile ops::decoder_feedforward --exact \
  --dump-schedule schedules/candidate.json --dump-graph schedules/candidate-graph.json
```

두 exact compile 모두 exit0, 기존 unused helper warning5개다. 변형1개와 원본1회로 총2회만 컴파일했다. [원본 로그](../../artifacts/ffn-down-scale-coalesced/baseline.log), [후보 로그](../../artifacts/ffn-down-scale-coalesced/candidate.log), schedule/graph/image의 해시는 evidence에 있다. 이 독립 구현·분석에서는 fixture/oracle/tolerance, public entrypoint, 라이브 소스, Git, NPU 작업을 바꾸지 않았다. 이후 root가 수행한 E45 결과를 위에 연결했고 성능 회귀로 채택하지 않았다.
