이 후보는 **컴파일만 통과했으며 미채택·NPU 미실행**이다. E47의 FFN(E37과 동일)에서 up packed weight를 8행씩 읽고 fused BF16 decode하는 변형이다. 두 번의 정확 컴파일 중 in-place slot 재사용은 backend 오류로 실패했고, owned tensor를 반환하는 최종본은 통과했다. 그러나 정적 makespan이 **162,209 → 162,644 cycles(+0.27%)**로 늘어 여기서 중단했다.

최종 [mlp.rs](mlp.rs)의 SHA-256은 `19362bb3f421b3295905825378748ab26d52fd45f6fa53400c5873e7b520d755`다. 실패한 [in-place 소스](mlp.in-place-owner-failed.rs)는 별도로 보존했다. [원본](mlp.original.rs), [patch](candidate.patch), [evidence](evidence.json)에 변경 범위와 두 시도의 로그·해시를 기록했다.

## 변경 범위와 좌표

동결 `e47-output-sixteen-columns/src/` 43개 파일과 격리 crate를 대조하면 MLP만 다르다. public ops·harness·axes·QKV/O는 그대로이며, Cargo.toml/lock/toolchain도 동일하다. MLP에서 gate 이후의 소스 전체는 원본과 byte-identical이다. up 전용 `project_staged_eight_rows_scaled_after_dot`은 weight 인자를 compact 8행 DM으로 받고 full-matrix tile을 제거하는 두 변경을 제외하면 원래 dot helper와 같다.

`decoder_feedforward` → `feedforward_sharded` → `project_up_and_gate`의 up 경로에서 다음을 실행한다.

1. FP4 HBM 전체 view를 동일 wire order인 `[Dummy2,L/32%240,L%32,H]`로 먼저 나눈다. 이미 tiled view를 reshape하는 알려진 실패 경로를 사용하지 않는다.
2. 각 cluster의 유효 slice 0–239에서 local row offset 0,8,16,24의 8행을 typed HBM→DM으로 가져온다. 240–255는 기존 padding이다.
3. `FP4 lookup→FP8 fetch_cast F32→BF16 cast→commit_trim8`로 해당 8행 BF16 matrix를 만든다. weight의 BF16 경계는 dot 전에 유지된다.
4. 기존 8행 dot을 그대로 적용한다. 각 16-element FP32 block dot → FP32 local scale → H block sum → BF16 4값 packet 두 번의 output commit 순서를 바꾸지 않는다.

HBM의 global row는 `r_global=7680*c+32*s+offset+r`이며 `c∈[0,2)`, `s∈[0,240)`, `r∈[0,8)`이다. nibble offset은 `3840*r_global+h`, byte offset은 그 값을 2로 나눈 몫이다. Scale은 같은 global row의 240개 block을 사용하고 output은 기존 local row `offset+r`에 쓴다.

[verify_layout.py](verify_layout.py)는 15,360행의 역변환·중복 없음·전체 coverage와 3,840열의 block/packet 분해를 각각 전수검사한다. 두 축의 Cartesian product로 58,982,400개 weight 좌표를 증명하며, 58M 원소를 NPU에서 비교했다는 뜻은 아니다. Helper 수치 연산과 gate/down suffix 불변 검사도 포함한다. 결과는 [verification.json](verification.json)에 있다.

## 두 시도와 실제 버퍼 수

[SDK scheduling 검토](../ffn-up-gate-fused-decode-e37/scheduling-options.md)의 `HbmTensorView::to_dm_view`와 `commit_view`로 FP4 DM slot 두 개, BF16 tile 하나를 번갈아 채우는 첫 시도를 만들었다. fake dependency나 barrier는 추가하지 않았다. Rust/type 단계 후 backend `abstract_buffer.rs:309`에서 다음과 같이 실패했다.

```text
T44 has an inconsistent state ... owner mismatch:
self [0,3686400)=T43 vs other [0,3686400)=T61
```

양쪽 shape는 모두 `(L%32=8,H)`인 FP4 DM이었다. 이 오류가 특정 padding 또는 API 전반의 결함 때문이라고 단정하지 않는다. 실패 소스와 [attempt-1.log](../../artifacts/ffn-up-tiled-fused-decode-e47/attempt-1.log)를 보존했다.

최종본은 `to_dm`과 `commit`이 새 owned tensor를 반환하도록 바꾸었다. 코드에 두 개의 prefetch handle을 교대로 shadow하지만, 이것이 물리 버퍼 두 개를 강제하지 않는다. 실제 [schedule](../../artifacts/ffn-up-tiled-fused-decode-e47/candidate.json)은 다음을 보인다.

| FP4 tile tensor | lifetime | padded extent |
|---|---|---:|
| 121 | [44896,79586) | 7,864,320B |
| 127 | [56553,87323) | 7,864,320B |
| 140 | [64453,94112) | 7,864,320B |
| 169 | [77553,98704) | 7,864,320B |

따라서 77,553–79,586에는 **FP4 tile 네 개가 동시에 살아 있다**. BF16 tile도 마지막 두 개가 겹쳐 최대 두 개가 동시에 살아 있다. 앞의 세 BF16 tensor는 주소 91,750,400을 시간에 따라 재사용하고, 네 번째는 209,190,912다. 이 최종본을 “물리 2-slot double buffer”로 부를 수 없다.

반면 up/gate 8행 scale VRF tensor 8개는 모두 주소 262,144를 사용하고 lifetime이 서로 겹치지 않는다. 유효 slice당 현재 scale은 `8×240×4=7,680B` 하나이며 VRF double buffering은 발생하지 않는다. Scale FP8 DM은 기존 32행 전체를 유지한다.

## 비용과 메모리 결과

Baseline은 E37 FFN image와 byte-identical한 [기존 exact schedule](../../artifacts/ffn-down-scale-coalesced/baseline.json)이다. E47의 MLP와 FFN 연산은 이 baseline과 같다. Context 수치는 구간 합집합이며 서로 겹치므로 합산하면 elapsed cycles가 되지 않는다.

| 항목 | Baseline | 후보 | 차이 |
|---|---:|---:|---:|
| makespan | 162,209 | 162,644 | +435 |
| Main | 125,274 | 114,280 | −10,994 |
| DMA | 135,607 | 138,928 | +3,321 |
| Sub | 19,378 | 20,928 | +1,550 |
| Vector | 34,748 | 53,080 | +18,332 |
| PeCore | 29,352 | 4,676 | −24,676 |
| Main∩DMA overlap | 104,921 | 97,581 | −7,340 |
| instruction 수 | 242 | 251 | +9 |

Up decode는 `9,863+19,463=29,326`에서 `4×4,583=18,332` cycles로 줄었다. 그러나 up HBM DMA는 26,475에서 `4×7,030=28,120`으로 1,645 늘었다. 상수 load도 838 cycles 두 개가 추가되어 전체 DMA 증가 3,321과 정확히 일치한다. Lookup Sub setup은 `2×1,293`에서 `4×1,034`로 1,550 늘었다. Gate/down dot 비용과 scale VRF load의 duration multiset은 그대로다.

[graph](../../artifacts/ffn-up-tiled-fused-decode-e47/candidate-graph.json)의 네 tile DMA node(108,113,125,151)는 각각 10개의 source descriptor를 사용한다. 기존 32행 chunk의 `None=240:256`이 tile마다 `None=60:256`으로 바뀌며, row-group source stride 61,440B와 최내부 256B packet은 같다. 전체 FP4 HBM payload는 줄지 않고 descriptor 구성이 나뉜다. 이를 실제 bus transaction 수나 bank 충돌 횟수로 환산하지 않았다.

후보에서 네 번째 tile DMA는 첫 fused decode와 일부 겹친다. 하지만 kernel 전체 Main/DMA overlap은 줄었다. Down BF16 decode가 143,332에 끝나도 scale DMA가 144,235까지 남아 첫 down dot은 144,836에 시작한다. Baseline 첫 dot 144,401보다 435 늦고, 최종 makespan 차이도 435다. Main 단독 절감량으로 성능 개선을 주장할 수 없는 경우다.

[analyze_schedule.py](analyze_schedule.py)는 모든 Sram tensor의 half-open lifetime과 `physical_pages`를 주소 구간으로 펴서, 같은 시점의 물리 주소 합집합을 계산한다. 이 **스케줄상의 메모리 점유 지표**는 250,052,608B(86,273 cycle) → 227,627,008B(56,553 cycle), 약 8.97% 감소다. Alias 중복을 제거하지만 기록된 padding을 포함한다. 실제 runtime telemetry, 최악 slice 점유, 별도 allocator 보장의 측정값은 아니다. 중간 working set을 줄인 효과는 관측되지만 정적 실행시간을 개선하지 못했다. 전체 분석은 [schedule-summary.json](schedule-summary.json)에 있다.

## 재현과 검증 한계

동결 E47의 `src/`, Cargo.toml/lock/toolchain을 Docker의 `/tmp/moa-ffn-up-tiled-fused-decode-e47`에 복사했다. 첫 시도 이전 target은 없었고, installed toolchain/cache를 재사용했다. `/workspace/src`와 `/workspace/target`, Git, fixture/oracle/tolerance는 변경하지 않았다.

```bash
export CARGO_HOME=/home/moa/.cargo
export CARGO_TARGET_DIR=/tmp/moa-ffn-up-tiled-fused-decode-e47/target
export FURIOSA_OPT_OUT_DIR=/tmp/moa-ffn-up-tiled-fused-decode-e47/target/furiosa-opt/kernel
export CARGO_NET_OFFLINE=true
cd /tmp/moa-ffn-up-tiled-fused-decode-e47
cargo furiosa-opt compile ops::decoder_feedforward --exact \
  --dump-schedule schedules/candidate.json --dump-graph schedules/candidate-graph.json
```

[attempt-2.log](../../artifacts/ffn-up-tiled-fused-decode-e47/attempt-2.log)는 성공했다. 정확 FFN image는 229,376B, SHA-256 `85fb3721867c8d0901f69a0ee8b8d2b25545f7c2faf343815262c0fe61d307cd`다. Full host 실행 파일은 만들지 않았고 NPU 실행도 하지 않았다. Scalar FP4 decode의 수학적 동등성·좌표 증명·compile PASS는 이 통합 pipeline의 하드웨어 정확성을 대신하지 않는다. 명시된 두 번의 컴파일 후 추가 변형은 시도하지 않았다.
