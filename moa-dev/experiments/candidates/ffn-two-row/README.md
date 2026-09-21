# FFN down: 2-row 즉시 contraction 후보

상태: `cargo-furiosa-opt 0.8.1`에서 대상 커널 컴파일 성공, **Arena job 70000에서 FFN 정확성 실패**. 현재 채택하면 안 된다. live `src/device/shared/mlp.rs`, 공유 `/workspace/target`, Git에는 변경하지 않았다.

## 변경과 보존한 계산

기존 `project_down`은 2-row weight half를 dequant·column 재배치한 뒤 두 half를 4-row `down_weight` tensor로 복사하고 한 번 contraction한다. 후보는 각 2-row half를 즉시 contraction한다. 큰 weight assembly copy 대신 작은 2-row 결과 copy만 수행한다.

- packed FP4 해석, FP8 local scale, BF16 dequant 결과를 보존한다.
- activation TRF, 1920-element column partition, 8-way inter-slice sum을 보존한다.
- contraction 후 BF16 변환, 상위 함수의 global scale·RMSNorm·residual·layer scalar 처리를 보존한다.
- 기존 4-row pass의 두 half는 각각 output offset `4*i + 2*j`에 쓴다. 전체 120 rows를 중복 없이 채운다.
- device entrypoint와 내부 함수 signature를 변경하지 않는다.

처음 시도한 2×BF16 direct commit은 4-byte packet이 되어 컴파일러가 거절했다. 지원 commit packet 크기는 `[8,16,24,32]` bytes다. 최종 후보는 2개 논리 원소를 4-slot padding으로 8-byte commit한 뒤, 논리 값 두 개만 TDMA로 destination에 복사한다. padding값을 계산 결과로 사용하는 방식이 아니다.

## 정적 비교

기준: `furiosa-opt-gemma4-12B/target/schedules/baseline/decoder_feedforward.json`.
후보: [decoder_feedforward.json](../../artifacts/ffn-two-row/decoder_feedforward.json).

| 항목 | 기준 | 후보 | 변화 |
| --- | ---: | ---: | ---: |
| Makespan | 1918472 | 1474463 | -23.14%, 정적 비율 1.3011× |
| DmaEngine interval union | 1066334 | 606494 | -459840 |
| MainContext interval union | 872451 | 880611 | +8160 |
| SubContext interval union | 129898 | 129898 | 0 |
| VectorEngine interval union | 675900 | 684060 | +8160 |
| Instructions | 1742 | 1622 | -120 |

기준 `mlp.rs`:412의 weight assembly DMA는 60회, 486900 cycles였다. 이 노드들이 사라졌고 후보 `mlp.rs`:429의 작은 output DMA 60회, 27060 cycles로 대체됐다. contraction은 기준 30회/22560 cycles에서 후보 60회/30720 cycles로 증가했다. 따라서 순수하게 copy 비용을 덜어낸 만큼 얻고, contraction 호출 증가 비용을 지불하는 예상과 일치한다.

남아 있는 BF16 tile의 column redistribution은 양쪽 모두 256500 cycles이고 dequant scale 적용은 477660 cycles다. 이번 후보는 이 두 경로를 변경하지 않는다. 세부 source/context 합과 source SHA256은 `static-comparison.json`에 있다. Context별 시간은 서로 겹치므로 합산하지 않는다.

정적 결과를 기존 Arena baseline 4264297 cycles에 곱해 실측 예측치로 제시하지 않는다. 동일 binary build와 공식 fixture를 사용해 RNGD에서 3개 커널 모두 정확성 통과와 FFN 중앙값 개선을 확인한 뒤 채택해야 한다.

## 재현과 파일

완성 파일은 `mlp.rs`, baseline 대비 unified patch는 `change.patch`다. `mlp.original.rs`는 실험 시작 시 live 원본을 복사한 파일이다. `mlp.direct-commit-failed.rs`와 [실패 컴파일 로그](../../artifacts/ffn-two-row/compile-direct-commit-failed.log)는 최소 packet 제약을 확인한 실패 시도의 기록이며 적용 대상이 아니다.

복사한 crate `/tmp/moa-ffn-two-row`에 후보 `src/device/shared/mlp.rs`를 배치하고 아래 명령으로 컴파일했다. 다른 커널과 Arena job은 실행하지 않았다.

```bash
docker exec \
  --workdir /tmp/moa-ffn-two-row \
  --env CARGO_TARGET_DIR=/tmp/moa-ffn-two-row-target \
  moa2026-dev \
  cargo furiosa-opt compile ops::decoder_feedforward --exact \
  --dump-schedule /tmp/moa-ffn-two-row/decoder_feedforward.json
```

컴파일 로그: [compile.log](../../artifacts/ffn-two-row/compile.log). 성공 출력은 `Finished 1 compiled, 14 not matching ops::decoder_feedforward`다. 스크래치 경로는 Docker 내부 임시 파일이고, 후보·patch·분석 파일은 이 디렉터리에, raw schedule과 로그는 Git에서 제외되는 `../../artifacts/ffn-two-row/`에 보존했다. artifact 경로와 SHA256은 `static-comparison.json`에 기록했다.

추가 검증: root의 job 70000에서 FFN 세 입력 모두 실패했다. 최종 direct-layout job 69992와 출력 오차 통계가 동일하므로 공통인 two-row contraction/reduction/padded commit/tiny DMA copy 경로가 원인 후보로 좁혀졌다. 정확한 원인은 아직 확정하지 않았고, 원본의 four-row contraction/direct commit을 복원하는 독립 후보를 검사 중이다.
