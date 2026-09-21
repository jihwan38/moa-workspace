# FFN 2-cluster의 8행 up/gate · 12행 down 타일

상태: up/gate-only, down-only, 조합을 각각 독립 컴파일했고 최종 조합 정적 makespan은 **204460**이다. 기존 all-two-cluster 4행 타일 213950 대비 4.44% 감소한다. **NPU 정확성·성능은 미검증**이다. `mlp.rs`는 정적 결과가 가장 좋은 조합이며 `mlp.combined.rs`와 같다.

## 연산과 출력 순서

Up/gate는 slice당 32행을 8행씩 4회 처리한다. scale VRF는 8 × 240 × 4 = 7680 bytes다. Down은 slice당 60행을 12행씩 5회 처리하며 scale VRF는 12 × 120 × 4 = 5760 bytes다. 둘 다 8192-byte 한도 안에 있다. HBM 배치·전송, local/global scale 위치, BF16 dequant/output 경계, 각 행의 dot-product 입력 순서, down의 기존 8-way F32 reduction은 유지한다. Block-scale reassociation은 포함하지 않는다.

큰 타일을 바로 transpose하는 데에는 두 제약이 있었다. 먼저 8개 BF16을 한 packet으로 만들면 compiler가 transpose의 `max_in_rows (4)` 제한으로 거절한다. 16-byte commit packet 자체가 허용되더라도 이 입력 transpose는 4행까지만 packet으로 배치할 수 있다. 또 `L%32=8` 또는 `H%60=12`처럼 낮은 행 축을 clip한 이름은 4행 packet/time 분해와 맞지 않는다.

해법은 dequantized BF16 타일의 연속 메모리를 canonical `L%8` 또는 `H%12` 행 축으로 reshape하고, 출력의 상위 group 축 하나를 직접 선택하는 것이다.

| 경로 | canonical 계산 행 축 | output time | output packet | mutable output tile |
| --- | --- | --- | --- | --- |
| up/gate | `L%8` | `L/4%2` | `L%4#16`, trim 4 | `L/8%4`에서 group i 하나 |
| down | `H%12` | `H/4%3` | `H%4#16`, trim 4 | `H/12%5`에서 group i 하나 |

따라서 up/gate는 4개 BF16의 8B packet을 2회, down은 3회 직접 commit한다. 별도 결과 복사나 output padding buffer를 추가하지 않는다. 출력 행 주소는 각각 `8*i + 4*time + lane`, `12*i + 4*time + lane`이다. 기존 32행/60행을 각각 한 번씩 순서대로 채운다. Dequantized tile reshape는 dense row-major scalar offset `row*columns+column`을 바꾸지 않는다.

Mutable output view를 reshape하는 시도는 Rust type-check 뒤 NPU MIR에서 `expected a DmTensorViewMut`로 거절됐다. 최종본은 `view_mut().tile()`을 직접 commit에 전달한다. Compiler 오류 로그와 시도 소스는 보존했다. `mlp.down-twelve-viewmut-failed.rs`·`mlp.down-twelve-commitaxis-failed.rs`는 up/gate에서 확인한 실패 형태의 구성 snapshot이며 개별 compile 실패를 확인한 파일이라는 뜻은 아니다.

## 독립 정적 비교

| 항목 | 4행 기준 | up/gate 8행만 | down 12행만 | 최종 조합 |
| --- | ---: | ---: | ---: | ---: |
| makespan | 213950 | 210305 | 208164 | **204460** |
| instructions | 525 | 445 | 420 | **340** |
| DMA union | 132947 | 132947 | 132947 | 132947 |
| Main union | 178468 | 174100 | 172938 | 168570 |
| Vector union | 145131 | 140763 | 139601 | 135233 |
| Sub union | 22914 | 20810 | 20284 | 18180 |

큰 타일의 scale VRF는 half-bank 크기보다 커질 수 있어 staging 겹침이 나빠질 가능성을 우려했지만, 이 compiler schedule에서는 두 변경이 각각 makespan을 낮췄고 조합이 가장 좋았다. 이는 하드웨어에서 같은 효과를 보장하지 않는다. DMA 작업량은 같고, tile당 반복 비용과 main/sub 작업이 줄어든다. NPU 실측 차이가 작으면 run 간 변동과 함께 비교해야 한다.

## 파일과 검증

- `mlp.rs`, `change.patch`: 최종 선택 후보와 4행 기준 대비 patch.
- `mlp.upgate-eight.rs`, `mlp.down-twelve.rs`, `mlp.combined.rs`: 세 독립 비교군.
- `mlp.four-row.rs`: all-two-cluster 4행 snapshot.
- `validate_layout.py`, `logical-index-proof.json`: row/packet coverage, dense reshape 주소, VRF 크기, feedforward/GeGLU 불변 검사.
- `static-comparison.json`: 세 schedule과 기준 비교, source/artifact SHA256.

Scratch `/tmp/moa-ffn-larger-tiles`, 독립 target `/tmp/moa-ffn-larger-tiles-target`에서 `ops::decoder_feedforward --exact`만 컴파일했다. [Up/gate 8행 schedule](../../artifacts/ffn-larger-tiles/decoder_feedforward-upgate-eight.json), [down 12행 schedule](../../artifacts/ffn-larger-tiles/decoder_feedforward-down-twelve.json), [최종 schedule](../../artifacts/ffn-larger-tiles/decoder_feedforward-combined.json), [최종 compile log](../../artifacts/ffn-larger-tiles/compile-combined.log). live source, 공유 target, Git, NPU job은 변경하지 않았다.

All-two-cluster 후보에서 MLP 한 파일 교체로 적용할 수 있다. 이전 입력-layout 연결 지침은 [all-two-cluster README](../ffn-all-two-cluster/README.md)를 따른다. 공식 3개 kernel × 3개 seed hardware harness를 통과하기 전에는 이전 4행 구현의 PASS를 이 후보의 정확성으로 간주하지 않는다.
