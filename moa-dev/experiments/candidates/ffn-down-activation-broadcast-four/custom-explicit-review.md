# Anonymous4 + CustomBroadcast32 후속 검토

요청된 조합 하나를 scratch에서 exact compile했다. **Compile PASS, makespan162,209→166,181(+3,972,+2.45%)**다. Down activation TDMA는 줄지만 Main의 Switch가 BF16 decode 뒤에 직렬로 배치된다. NPU 정확성·성능 결과는 없다. 이 결과로 원본을 교체하거나 추가 tile sweep을 하지 않았다.

소스는 [mlp.custom-explicit.rs](mlp.custom-explicit.rs), SHA256 `bde02fcad8e40d6a3f210cf91b3bbb59079cd8456dff22af97962a542ddfd134`다. [E37 대비 patch](custom-explicit.patch)에는 `project_down_sharded`의 activation copy만 바뀐다. [직전 실패 mlp.rs](mlp.rs)와의 차이는 `Broadcast1 { slice1: 4, slice0: 8 }`를 `CustomBroadcast { ring_size: 32 }`로 바꾼 한 곳뿐이다. E45 scale-coalesced 변경은 포함하지 않았다.

## 지원된 실제 매핑

입력 stage는 `[Dummy256/32,1#4,L/1920]`, Switch 출력은 `[Dummy256/32,4,L/1920]`, element는 둘 다 `L%1920`이다. 익명 `4`를 실제 복제 축으로 먼저 만들고 기존 `DownRowsByColumns=[H/60%32,L/1920]`로 같은 물리 순서의 axis rename을 한다.

[Compiler graph](../../artifacts/ffn-down-activation-custom-explicit/custom-explicit-graph.json)의 operator195는 `CustomWithBitmap`, ring32, BF16 packet16을 기록한다. 256개 output slice 각각의 source는 정확히 `32*(s//32)+(s%8)`다. 각32-slice ring의 유효 source0–7이 같은 column의 네 행으로 복제된다. 모든 bitmap256개를 이 식과 대조했다. 출력 `s=32g+8r+j`의 입력 slice는 `32g+j`, element `l`은 동일하여 global activation index `1920j+l`이 유지된다. `g∈[0,8)`, `r∈[0,4)`, `j∈[0,8)`이고 마지막 rename은 row group을 `4g+r`로 해석한다. Padding source8–31을 유효 데이터로 읽는 bitmap은 없다.

이는 compiler가 내보낸 mapping과 논리 좌표의 검증이다. 실제 장치에서의 load/snoop/commit 실행이나 정확성까지 입증하지는 않는다. Dot, weight/scale 배치, FP32/BF16 경계는 소스상 그대로다.

## 전체 비용

| 측정 | E37 exact | Custom32 explicit4 | 차이 |
| --- | ---: | ---: | ---: |
| Makespan | 162,209 | 166,181 | +3,972 |
| Instruction 수 | 242 | 245 | +3 |
| Down activation DMA | 9,392 | 4,784 | -4,608 |
| 새 snoop bitmap DMA | 0 | 719 | +719 |
| 전체 DMA interval union | 135,607 | 131,718 | -3,889 |
| Main interval union | 125,274 | 129,377 | +4,103 |
| Sub interval union | 19,378 | 19,378 | 0 |
| Vector interval union | 34,748 | 34,748 | 0 |

Candidate schedule의 activation DMA node434는 `[125264,130048)`, down scales node435는 `[130048,139822)`다. BF16 weight decode가143,527에 끝난 뒤 Main Switch node466이 `[143527,147630)`에 실행된다. 따라서 줄어든 DMA 작업량만 보고 빨라졌다고 말할 수 없다. 새 bitmap load node287은 `[81280,81999)`이며 Switch의 추가 SRAM input157로 연결된다. Main instruction duration multiset은 원본에4,103 한 개만 추가됐고, DMA multiset은9,392 한 개 대신4,784와719가 추가됐다.

기존 네 실패와 이번 결과는 구분한다. Named output 축을 유지한 Custom32는 snoop enumeration으로 실패했고, explicit anonymous4라도 Broadcast1은 backend modulo panic으로 실패했다. 두 조건을 조합한 이번 explicit anonymous4+Custom32는 통과했다. 따라서 이 복제 자체나 Custom32 전체가 미지원이라는 결론은 성립하지 않는다. 각 실패의 compiler 내부 근본 원인을 모두 규명했다는 뜻은 아니다.

## 재현과 동결

독립 Docker scratch `/tmp/moa-ffn-down-activation-custom-explicit`에서 E37 source 전체를 사용했다. `mlp.rs`만 위 파일로 교체·touch했다. 다른 독립 scratch의 cache를 복사했으며 live source/main target, Git, NPU는 건드리지 않았다. SDK/CLI0.8.1, Rust1.97.0-nightly다. 후보1개, 아래 exact compile1회만 수행했다. 기준 schedule은 [E37 exact 재현](../ffn-down-scale-coalesced/evidence.json)의162,209이며 원본 image가 E37 image와 byte-identical함을 이미 확인했다.

```bash
cd /tmp/moa-ffn-down-activation-custom-explicit
CARGO_HOME=/home/moa/.cargo \
CARGO_TARGET_DIR=/tmp/moa-ffn-down-activation-custom-explicit/target \
FURIOSA_OPT_OUT_DIR=/tmp/moa-ffn-down-activation-custom-explicit/target/furiosa-opt/kernel \
CARGO_NET_OFFLINE=true \
cargo furiosa-opt compile ops::decoder_feedforward --exact \
  --dump-schedule schedules/custom-explicit.json \
  --dump-graph schedules/custom-explicit-graph.json
```

[Compile log](custom-explicit-compile.log)은 exit0, 기존 unused helper warning5개다. Candidate image222,976B의 SHA256은 `bca2ed2682ec74232e951a4ca4c6d0980f73e381251ee5fb1a420e109599e84e`다. [Evidence](custom-explicit-evidence.json)에 source/log/schedule/graph/image 해시와 비교 결과를 보관한다. 추가 compile은 하지 않았다.
