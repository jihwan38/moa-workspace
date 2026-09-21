# E47 O projection: 32×128 column 분할 — 정적 회귀로 제외

E47의 O16×256 배치를 O32×128로 바꾼 단일 후보다. `ops::sliding_attention_output --exact` 첫 compile은 PASS였지만 **24,621 → 37,344 cycles(+12,723, +51.68%)**로 늘었다. 추가 compile이나 NPU screen 없이 동결했다. 입력 payload 감소보다 weight DMA의 정적 비용 증가가 훨씬 컸다. 이것은 실제 NPU 성능이나 정확도 측정 결과가 아니다.

[projection.rs](projection.rs) SHA256은 `0358b2443825fc55412b050f54f3a69216202da5b4500e6cf858d1f1861897f7`이다. [동결 kernel image](../../artifacts/output-fp8-thirtytwo-columns-static/sliding_attention_output.bin)는 89,600 bytes, SHA256 `dd6472ab1f232fa0a8b026a5904200f6d5488fd329f98d79d6b5c5642f93854b`다. [candidate.patch](candidate.patch), [evidence.json](evidence.json), schedule/graph/log를 함께 보존했다.

## 변경 범위와 실제 배치

[E47 snapshot](../../artifacts/e47-output-sixteen-columns/provenance.json)의 source 43개와 manifest 3개를 별도 scratch `/tmp/moa-output-fp8-thirtytwo-columns`로 복사했다. 기준 projection SHA는 `db048b93d9b768bb4cb9cb6f8fd603783789d4a5eb609e42a00a23c69a330718`이다. 기준 schedule은 [O16 exact compile](../output-fp8-sixteen-columns/candidate.json)을 재사용했다. 실제 compile 입력 [manifest](candidate-source-manifest.json)는 projection 하나만 달라졌음을 확인한다.

`project_output` 내부와 private O quantizer의 차원만 바꿨다. Q/K/V 함수, 공개 signature, `ops.rs`, H480×8 channel-scale/RMS/residual epilogue, FFN 및 공식 harness는 그대로다. 공개 caller는 기존 반환 배치를 받는다.

output row는 `h=1920*c+240*g+r`, input column은 `q=128*k+t`로 매핑한다. c∈[0,2), g∈[0,8), r∈[0,240), k∈[0,32), t∈[0,128)이다. physical slice는 `32*g+k`이고 weight의 local 좌표는 `[r,t]`다. 2 clusters×8 rowgroups×32 columns가 512개 slice를 전부 사용한다. local weight는 240×128=30,720 bytes로 O16의 120×256과 같다.

각 128-input column에서 amax → power-of-two scale → FP8 hi/lo를 만든다. native paired dot은 row마다 64-element packet 2개를 합산한다. hi+lo/16에 해당 column의 input scale을 곱한 F32 partial을 DM에 보존하고, 다음 stream에서 32개 column slice를 F32로 합산한 뒤 BF16로 cast한다. 4-row transpose를 60번 적용해 240 BF16 값을 8-byte 단위로 commit한다. `OutputRows=[H/240%8,1#32]`의 물리 padding을 그대로 둔 채 cluster 축만 `H/1920`으로 rename하고, typed DMA로 기존 H480×8 epilogue 배치를 복원한다. padding을 유효 row로 바꾸는 reshape는 없다.

[verify_layout.py](verify_layout.py)는 3,840 rows와 4,096 columns의 bijection, 122,880 physical row/column tiles, 모든 compile 입력의 범위를 검사한다. 두 bijection의 곱은 15,728,640 weights를 정확히 덮는다. graph의 `GReduceAddF [ratio=32]`, BF16 cast, `commit_in_size: 8B`, `commit_projection: 0+[None=60:4, None=4:1]`도 확인했다. [verification.json](verification.json)은 PASS다.

## 정적 회귀의 위치

| slice당 payload | O16 기준 | O32 후보 |
|---|---:|---:|
| BF16 입력 | 512B | 256B |
| paired FP8 및 TRF 유효 입력 | 512B | 256B |
| hi F32 VRF | 1,024B | 512B |
| input-scale VRF | 32B | 32B |
| FP8 weight | 30,720B | 30,720B |
| F32 partial, `1#8` padding 포함 | 3,840B | 7,680B |
| reduced BF16 physical local 저장량 | 240B | 480B |

TRF payload 감소는 고정 TRF bank 예약량 감소를 뜻하지 않는다. unique HBM input/weight 총량도 같다. 같은 입력을 복제하는 rowgroup 수와 각 slice의 shape가 달라진다.

| instruction duration | O16 | O32 | 차이 |
|---|---:|---:|---:|
| 입력 HBM→DM | 933 | 738 | −195 |
| 입력 DM copy | 691 | 563 | −128 |
| weight HBM→DM | 13,383 | 26,218 | +12,835 |
| native pair Main | 1,242 | 1,245 | +3 |
| column reduction Main | 522 | 786 | +264 |
| H480 epilogue gather | 446 | 447 | +1 |
| 전체 makespan | 24,621 | 37,344 | +12,723 |

amax/hi/lo Main은 각각 345→313 cycles, hi-VRF Sub는 295→279, TRF Sub는 327→295로 줄었다. 하지만 weight DMA는 3,326–16,709에서 3,099–29,317로 길어졌고 paired dot은 그 완료까지 기다린다. pair 완료는 17,951→30,562, reduction 완료는 18,473→31,348, gather 완료는 19,072→31,795다. 바뀌지 않은 channel-scale epilogue 시작이 20,672→33,395로 12,723 cycles 밀려 전체 회귀와 같다. 중첩되는 stage duration을 모두 더해 전체 지연으로 해석하지 않는다.

[weight-dma-mapping.json](weight-dma-mapping.json)에 실제 compiler command를 보존했다. O16 최내부 전송은 `Qs_1=1:256`, local row stride는 256이다. O32에서는 `Qs_1=1:128`, local row stride 128이며 local row 수가 120→240으로 늘었다. 이 후보에서는 compiler가 나타내는 연속 전송 구간이 실제로 256→128 bytes로 변했다. weight량이 같은데 해당 DMA의 정적 비용은 1.959배다. **실제 하드웨어 bus transaction 수나 최소 효율 단위는 측정하지 않았으므로, 이 자료만으로 128-byte 전송의 하드웨어 비효율을 확정하지 않는다.**

## 수치 변화와 재사용 조건

quantizer의 floor `2^-119`, exponent mask, input-scale 계수 `1/128`, hi/lo 계수는 같다. finite BF16 input에 대한 scale 범위 `2^-126..2^120`, normalized input 절댓값 <256, RN E4M3의 low-term 입력 절댓값 ≤128이라는 기존 경계도 유지한다. 유한 FP8 weight의 최대 절댓값 448로 잡으면, scale 적용 전 128-term hi dot의 보수적 절댓값 상계는 14,680,064, lo dot은 7,340,032다. 이후 외부 scale/RMS를 포함한 모든 연산의 overflow-freedom을 뜻하지 않는다.

128-group은 256-group의 부분집합이므로 input scale이 커지지는 않는다. 그러나 그룹별 quantization이 바뀌고 F32 column 합산도 16→32로 바뀐다. BF16 cast의 위치와 후단 순서는 유지해도 bitwise 동치, 모든 입력의 tolerance 통과, 출력 오차 감소를 보장하지 않는다. NPU 정확도 검사는 수행하지 않았다.

현 상태에서는 O16 채택본을 유지한다. 이 O32 source를 다시 시험하려면 정적 비용을 거의 두 배로 만든 weight DMA 배치를 먼저 바꾼 별도 가설이 필요하다. 단순히 column 수를 더 늘리는 후속 sweep의 근거로 사용하지 않는다.
