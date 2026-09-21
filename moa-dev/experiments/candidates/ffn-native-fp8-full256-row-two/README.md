# Full256 native FFN의 2-way 행 배치 — compile PASS, NPU 미검증

2026-09-21. 출발점은 `../ffn-native-fp8-full256-directcommit/mlp.direct-scales.rs`, SHA256 `7c363a7f116ff9ebf4027f6a9957faa5cb3ce74aac85b3c5f7b6579ff628b267`이다. 각 slice의 실제32행 buffer와 마지막2행 zero tail, direct LUT commit/direct scale copy, native FP8 DPE helper를 유지하고 **HBM row mapping과 출력 복원만** 바꿨다.

동일 base의 exact FFN static **191,836 → 182,868 cycles(-8,968, -4.67%)**다. HBM weight/scale DMA는 줄지 않았고, 개선은 작은 Switch와 final typed gather에서 나온다. NPU 제출은 하지 않았다. E35 full256의 공식 통과나 다른 후보의 통과를 이 mapping에 승계하지 않는다.

## 원소가 이동하는 경로

원래 full256은 `L=7680*c+256*r+slice`로30개의 행마다256 slices를 오갔다. 새 배치는 다음과 같다.

```text
L = 7680*c + 60*g + 2*r + s
c=0..1, g=0..127, r=0..29, s=0..1
physical slice = 2*g+s
```

Packed weight HBM view를 `[Dummy2,FfnSlice/2,FfnRow=30,FfnSlice%2,H]`, local scale은 같은 앞축 뒤에 `H/16`으로 읽는다. 이 shape의 연속 index는 원래 `L*H+h`, `L*(H/16)+block`과 같다. 모든256 slice가 실제30개 행을 가지며, dot이 처리하는32행 weight/scale buffer는 기존처럼 **실제 memset(0) 뒤 첫30행만 덮어쓴다**. Top padding을 실제 zero로 가정하지 않는다.

`compact_full256_rows`는 first30-row read tile만 읽고 `SwitchConfig::Broadcast1{slice1:2,slice0:1}`로 같은 g의 인접 두 slice를 모은다. 출력의 논리 순서는 `[r30,s2]`, 즉 row60개다. 이어 transpose가 인접한2개 r×2개 s를 **BF16 4개, 8B**의 연속 packet으로 만든다.

```rust
.switch::<m![FfnSlice / 2, 1 # 2], m![FfnRow = 30, FfnSlice % 2]>(
    SwitchConfig::Broadcast1 { slice1: 2, slice0: 1 })
.collect::<m![FfnRow = 30, FfnSlice % 2], m![1 # 16]>()
.transpose::<m![FfnRow = 30 / 2], m![(FfnRow = 30 % 2, FfnSlice % 2) # 16]>()
.commit_trim::<m![FfnRow = 30 % 2, FfnSlice % 2]>()
.commit()
```

Commit 뒤의 실제 연속60 BF16 값을 `L%60`으로, Slice group을 `[L/60%128,1#2]`, cluster를 `L/7680`으로 rename한다. 그 다음 typed DMA가 기존 `Cluster/UpGateRows/L%60`으로 옮긴다. Rename 전에 Switch/transpose가 실제 복사를 수행하며, padded30-row buffer를 unsafe reshape만으로60행이라고 해석하지 않는다. 끝2개의 zero row는 gather·GeGLU에 들어가지 않는다.

## 보존한 계산과 검증 범위

[validate_layout.py](validate_layout.py)는 다음을 확인한다.

- 15,360개 L행의 중복·누락 없는 대응, weight/scale 행 주소와 canonical group 순서.
- 전체32행 zero-fill → 첫30행 write → first30만 read 모델. Output은 projection당3,840개의8B packet이다.
- `project_rows_with_scale_contraction` 전체 byte-identical.
- `feedforward`부터 파일 끝까지 byte-identical: runtime quantizer/activation broadcast, GeGLU, down, global scales가 바뀌지 않는다.
- `project_up_and_gate`에서 네 HBM view 변경을 제외하면 기존 dense memset/LUT direct commit/scale direct copy/DPE 호출이 그대로다.

[logical-index-proof.json](logical-index-proof.json)이 결과다. 산술·dtype·DPE helper를 바꾸지 않았지만, 이전부터 있는 FP8 hi/lo 근사와 BF16 block rounding을 없앤 후보는 아니다. CPU 좌표 검사는 compiler/NPU padding·allocator·Switch 실행을 증명하지 않는다.

## 실제 정적 비용

| 항목 | 256-way base | 2-way 후보 |
| --- | ---: | ---: |
| 전체 makespan | 191,836 | **182,868** |
| Main union | 115,325 | 99,061 |
| DMA union | 145,727 | 144,055 |
| Sub union | 54,281 | 54,281 |
| VectorEngine union | 40,741 | 40,741 |
| Up/gate packed HBM DMA, 각각 | 29,427 | 29,427 |
| Up/gate local-scale HBM DMA, 각각 | 6,565 | 6,565 |
| Output Switch gather, 각각 | 8,457 | **325** |
| Output typed gather, 각각 | 1,366 | **530** |
| Native 첫 blockdot, 각각 | 7,964 | 7,964 |

Main duration은16,264, DMAStos는1,672 감소했지만 overlap 때문에 makespan은8,968 줄었다. 특히 gate 결과의 첫 gather는 다음 up zero-fill/weight DMA와 겹친다. Compiler의 HBM DMA 모델에서는 row2가 bandwidth 비용을 줄이지 않았으며, 실제 HBM bank 효과도 측정하지 않았다.

Full32 weight memset15,626×2와 scale memset1,226×2는 그대로 남아 있다. 이 후보는 tail-only memset와 독립인 변경이지만 결합했을 때의 schedule·NPU 통과는 별도로 확인해야 한다. 현재 source 자체가 safe BF16 E36보다 빠르다는 주장도 하지 않는다.

## 실패 이력과 재현

첫 compile는 packet을 `[r%2,s%2#8]`로 만들었다. 이것은 내부에 padding hole이 있어 `commit_trim`이 제거할 수 없었다. [실패 source](mlp.attempt1.rs), [로그](attempt1-internal-packet-padding.log)를 보존했다. 두 번째는 전체4-value packet에 바깥 padding만 둔 `[(r%2,s%2)#16]`으로 수정해 PASS했다. 총2회 compile,1회 오류이며 추가 시도는 하지 않았다.

Scratch `/tmp/moa-ffn-full256-row-two`의 전용 target/cache에서 `cargo furiosa-opt compile ops::decoder_feedforward --exact`를 실행했다. Base와 Cargo.toml/Cargo.lock/rust-toolchain hash는 같으며, 컴파일 source는 이 폴더 source와 일치한다. [evidence.json](evidence.json)에 전체 source manifest, [baseline.json](baseline.json)/[candidate.json](candidate.json) hash, 비용과 한계를 기록했다. [candidate.patch](candidate.patch)는 원본 MLP 하나에 대한 diff다.

실제 적용 파일은 [mlp.rs](mlp.rs)다. `ops.rs`와 `residual.snapshot.rs`는 exact 비교 조건을 보존한 참고본이므로 다른 커널의 최신 caller를 덮어쓰는 파일이 아니다. Live source·공유 target·Git·NPU를 변경하지 않았다.
