# Fused FP4 decode 실패의 제한적 IR 조사

확인된 결과는 job70162의 FFN run0 FAIL(max delta 0.63281, tolerance 내 30.96%)와 다음 launch timeout이다. 반면 두 단계 BF16 decode E12는 job70157의 official 15 comparisons를 통과했다. Fused 경로가 수학적으로 표현 가능한 값을 사용하고 compile된다는 사실만으로 NPU 동작을 보장할 수 없었다. 정확한 고장 지점은 아직 확정하지 않았다.

SDK 0.8.1 `engine/fetch_adapter.rs:190-245`는 `fetch_table_lookup::<f8e4m3>()` 뒤 `fetch_cast::<f32>()`를 연결하라고 명시적으로 안내한다. Functional 구현은 각각 `inner.map(|v| v.lookup())`, `inner.map(|v| v.cast())`이며 Mapping을 보존한다. 따라서 현재 증거로는 해당 API 조합이 금지되었다고 결론낼 수 없다. source path/hash와 실제 LIR operator/tensor를 `fused-decode-ir-evidence.json`에 보존했다.

두 후보의 대표적인 32행×3840열 projection을 비교했다:

| 항목 | 통과한 두 단계 경로의 첫 단계 | 실패한 fused 경로 |
|---|---|---|
| 입력 | FP4 DM, 동일 논리 Mapping | FP4 DM, 동일 논리 Mapping |
| fetch size | 32 | 32 |
| fetch network dtype / flit 원소수 | F8 / 32 | F32 / 8 |
| lookup table layout | table_lookup stride8, 128×dup8, tail8 | stride16, 64×dup4, tail16 |
| lookup SRAM, 전체 512 slices | 8388608B (slice당16384B) | 4194304B (slice당8192B) |
| cast/commit | F8 32개, 32B commit | BF16 8개, 16B commit |
| commit projection | 3840 packets×32 원소 | 15360 packets×8 원소 |

두 commit은 모두 122880개 논리 원소를 기록한다. Fused output의 BF16 byte size와 원래 BF16 matrix Mapping에도 겉으로 드러나는 누락은 없다. Lookup table의 반복 수·packing이 달라지는 것은 관측된 차이이며, 이것이 버그라는 증거는 아니다. SDK 공개 crate는 일부 lowering을 prebuilt private implementation에 위임하므로 이 조사에서 register-level table packing/FP8 cast mode 동작까지 확인하지 못했다. Timeout 원인 역시 잘못된 numerical decode와 동일한 원인이라고 단정하지 않는다.

향후 필요하면 별도 scratch diagnostic으로 16개 FP4 code를 서로 다른 adjacent nibble 순서로 반복한 행을 두 단계/fused 각각 BF16 HBM으로 출력해 bitwise 비교하면 decode 실패 여부를 FFN에서 분리할 수 있다. 한 행과 32행, padded240/256 slice 배치를 차례로 늘리면 packing과 allocation 문제를 나눠 볼 수 있다. 현재 이 diagnostic을 만들거나 NPU 제출하지 않았고, 통과한 두 단계 경로를 사용한다.
