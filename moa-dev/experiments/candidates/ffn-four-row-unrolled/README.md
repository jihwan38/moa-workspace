# FFN four-row kernel의 고정 반복 전개

상태: MLP만 전개한 버전과 residual까지 전개한 버전 모두 독립 컴파일 성공. **MLP + residual 전개는 Arena job 70070에서 3개 kernel × 3개 독립 seed 전부 PASS**다. FFN 중앙값 **820226 cycles**, run별 822292 / 820107 / 820226이다. 원본 실측 4264297 대비 5.199×, four-row rolled 실측 대비 1.389×다. [하드웨어 로그](../../../logs/e6-all-unrolled-job-70070.log). MLP-only의 독립 NPU 측정은 없다. 출발점 four-row direct-layout는 Arena job 70029에서 모두 통과했고 FFN 중앙값 1139366 cycles였다.

## 변경

`mlp.rs`는 up 15회, gate 15회, down 30회의 four-row loop를 module-local `macro_rules! static_passes`로 펼친다. 입력 tile, local/global scale, 연산 순서, BF16 경계, four-row contraction/reduction/direct output commit은 그대로다. 고정 index 목록은 원래 `0..15`, `0..30`과 같다.

`residual.rs`는 `shared::residual::add`의 480-element tile 8개를 function-local macro로 전개한다. `add_vision`은 변경하지 않는다. 이 helper는 attention-output에서도 사용하므로 두 파일을 적용하면 FFN 외 attention-output도 공식 harness로 확인해야 한다.

공식 [Tuning 문서](https://developer.furiosa.ai/furiosa-opt/book/scheduling/tuning.html)는 반복을 노출하면 staging과 계산을 겹칠 수 있고, kernel에 loop가 남지 않아야 register reuse가 활성화된다고 설명한다. `#[unroll]`은 crate-level `proc_macro_hygiene`, `stmt_expr_attributes` feature가 필요하다. 현재 `lib.rs`를 바꾸지 않고 허용된 device 파일 안에서 같은 고정 반복 전개를 표현하기 위해 Rust 선언형 macro를 사용했다. SDK private marker API는 호출하지 않는다.

## 정적 비교

| 항목 | four-row rolled | MLP만 전개 | MLP + residual 전개 |
| --- | ---: | ---: | ---: |
| makespan | 562630 | 403376 | **392312** |
| DmaEngine union | 263227 | 263227 | 263227 |
| MainContext union | 319304 | 319304 | 319304 |
| VectorEngine union | 264270 | 264270 | 264270 |
| SubContext union | 33121 | 33121 | 33121 |
| instructions | 876 | 816 | 808 |

DMA와 vector/contraction 작업량은 그대로이고 schedule 겹침과 제어 경로가 달라졌다. 최종 정적 makespan은 출발점 대비 약 30.27% 감소, 1.4341×다. 정적 값에서 실제 NPU speedup을 확정하지 않는다.

전체 전개 schedule의 `PeCore` interval 합이 커 보이는 이유는 `DramReuse(72)` lifetime `[10671,390243]`이 추가되기 때문이다. 이 lifetime을 CPU가 379572 cycles 동안 산술 연산으로 바빴다는 뜻으로 해석하지 않는다. Raw JSON과 detailed source/context 합을 함께 확인한다.

남은 큰 정적 작업은 down packed weight DMA 58302, activation replication DMA 53136, up/gate packed DMA 각각 49157, up/gate scale dequant 각각 61815 cycles다. 단순히 FFN의 절대 사이클이 크다는 이유보다 이 critical interval이 실제로 겹치는지를 기준으로 다음 변경을 고른다.

## 파일과 적용

- `mlp.rs` + `residual.rs`: 최종 두 파일 후보.
- `mlp.rs`만 적용: MLP-only 전개를 별도 측정할 수 있다.
- `mlp.rolled.rs`, `residual.rolled.rs`: 출발점 snapshot.
- `change-mlp.patch`, `change-residual.patch`: 각 snapshot 대비 patch.
- `static-comparison.json`: 세 버전 비교, 최종 source와 artifact SHA256.

Scratch `/tmp/moa-ffn-four-row-unrolled`, 독립 target `/tmp/moa-ffn-four-row-unrolled-target`에서 `ops::decoder_feedforward --exact`만 컴파일했다. live source, 공유 target, NPU jobs, Git에는 변경하지 않았다.

- [MLP-only schedule](../../artifacts/ffn-four-row-unrolled/decoder_feedforward-mlp.json)
- [전체 전개 schedule](../../artifacts/ffn-four-row-unrolled/decoder_feedforward-all.json)
- [MLP-only 컴파일 로그](../../artifacts/ffn-four-row-unrolled/compile-mlp.log)
- [전체 전개 컴파일 로그](../../artifacts/ffn-four-row-unrolled/compile-all.log)

Job 70070에서 FFN run별 max absolute error는 0.00781 / 0.01172 / 0.00781, mean은 0.000270 / 0.000380 / 0.000400으로 four-row rolled와 표시 정밀도에서 동일하다. QKV 중앙값 165134, attention-output 113136도 모두 PASS였다. 하드웨어 실행은 root가 수행했고 이 후보 작성자는 scratch 컴파일만 수행했다.
