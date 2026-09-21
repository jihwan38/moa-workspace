# Block-dot 결과를 packet에 묶는 API 실험: 미채택

2026-09-21, SDK 0.8.1. E15의 안전한 두 번 lookup 경로와 BF16/F32 정밀도 경계를 유지하면서, 16원소 dot 결과 여러 개에 block scale을 동시에 곱하려는 실험이다. **다섯 변형 모두 exact compile에서 거절됐다. 실행 가능한 후보나 NPU 측정 결과는 없다.**

기준은 `../../artifacts/e15-all-blockscale-safe-decode/src`이다. `mlp.original.rs`의 SHA-256은 `8717b2b450e4eba1d6fa8ad22a184d1fd86f76852c892295ee1d6c75574b3450`이다. 별도 scratch `/tmp/moa-ffn-blockscale-packed-lanes`에서만 컴파일했고, source tree의 다른 파일들은 capture와 같은 hash임을 확인했다. 공식 source, fixture, live binary는 변경하지 않았다.

| 파일 | 변경 | exact compile 결과 |
|---|---|---|
| `mlp.rs` | 8개 block을 `contract_lane` 출력 packet으로 이동. Up/gate Time `[row8,H/128]`, packet `[H/16%8]`; down도 같은 방식 | `OutPacket mismatch. Expected: 1 # 8, got: H / 16 % 8` |
| `mlp.two-block.rs` | BF16 64B input packet 안에 16원소 block 두 개를 유지. `contract_packet<[H/16%2]>`, `LaneMode::Sequential`, narrow packet `H/16%2#4` | contraction 단계는 통과. `vector_fp_binary`에서 scale VRF가 broadcast나 4연속 주소 읽기에 대응하지 않아 거절 |
| `mlp.two-block-four-row.rs` | up/gate와 down을 4행으로 줄이고, scale VRF에 두 개의 padding lane을 넣어 packet `2#4` 구성 | `collect`는 기존 fetch stream의 연속 packet을 padding/순열로 바꿀 수 없어 거절 |
| `mlp.two-block-four-row-padded-fetch.rs` | scale `fetch` 단계부터 packet `2#4`를 지정 | fetch cast 뒤 실제 packet은 `2#8`로 정규화됨. 요청한 두 개 `2#4` 그룹으로 수집할 수 없어 거절 |
| `mlp.two-block-padded-dm.rs` | Main F8→F32의 `2#8` packet을 `commit_trim<2#4>`로 실제 padded DM에 저장. Sub는 두 슬롯씩 32B fetch | MIR과 VRF operand 검증은 통과했지만 TensorUnit `lower_fetch_unit`에서 `No valid fetch size found: pv=4 consec=4 in=Float8e4m3 out=Float32`로 거절 |

각 변형의 compile log, patch, SHA-256은 같은 폴더와 `source-hashes.json`에 보존했다. 모든 컴파일 명령은 `cargo furiosa-opt compile ops::decoder_feedforward --exact`를 사용했다. 실패 시에도 도구가 `Finished 1 compiled`를 출력하므로 성공 여부는 exit code 1과 실제 `error:` 진단으로 판단했다.

## 왜 8개 Time을 Lane으로 바로 이동할 수 없는가

컨테이너의 SDK 구현 `furiosa-opt-lower-0.8.1/src/verify/engine/contraction/lane.rs`, `config_contract_lane`을 확인했다. `Interleaved` 모드의 출력 packet은 **입력 Lane을 8원소로 padding한 결과**와 같아야 한다. 원래 Lane이 `1`이면 결과는 `1#8`이다. `OutTime`의 바깥 부분도 원래 Time과 같아야 한다. 따라서 이름만 바꿔 block Time 축을 유효 Lane으로 승격하는 것은 지원하지 않는다.

`Sequential` 모드는 packet reduction 뒤 남긴 축을 출력 packet에 유지할 수 있다. 그래서 64B BF16 packet의 32원소를 두 개 16원소 block으로 나누는 contraction 자체는 verifier를 통과했다. 그러나 VE는 한 명령에서 operand를 scalar broadcast하거나 F32 네 개의 연속 주소로 읽는다. 실제 block 둘과 padding 둘이 섞인 출력에는 동일한 padded scale layout이 필요하다.

4행 up/gate에 `2#4` scale layout을 만들 수 있다면 VRF는 `4 × 120 × 4 × 4 = 7,680B`다. 실제 padded fetch의 `2#8` layout은 같은 4행에 15,360B이므로 이 방식으로는 8KiB VRF 안에 들어가지 않는다. `collect`는 fetch adapter가 결정한 packet을 임의로 압축하는 API가 아니다.

다섯 번째 변형은 실제 scale DM의 각 16B 슬롯에 값 둘과 padding 둘을 저장하도록 했다. 이 배치라면 Sub가 `[block/2%2,block%2#4]`를 32B로 읽고 VE의 4연속 VRF operand와 대응시킬 수 있다. 그러나 F8→F32 변환과 padding 삽입을 한 번에 수행하는 fetch에 지원되는 hardware size가 없어 lower 단계에서 실패했다. 이 결과를 받은 뒤 추가 시도를 중단했다. 먼저 전체 scale을 연속 F32 DM에 변환하고 별도 재배치하는 경로, 2행, 추가 vector pipeline은 구현하지 않았다. 이 결과는 테스트한 다섯 경로가 거절되었다는 증거이며, 모든 가능한 packet packing의 불가능성을 증명하지 않는다.

## 수치 경계

제안은 각 16원소의 BF16 dot을 F32로 받고 block scale을 F32로 곱한 뒤 F32로 누적하는 E15 경계를 보존한다. 두 block을 packet에 담는 것 자체는 추가 양자화를 요구하지 않는다. 다만 vector lane reduction의 순서가 달라질 수 있어 성공적으로 구현하더라도 bitwise 동일성을 주장할 수 없다. 정확도와 성능은 별도의 공식 seed 및 추가 seed NPU 검사 전에는 확인되지 않는다.
