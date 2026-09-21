# Down의 8-column 분할을 2-column으로 변경

**채택하지 않는다.** Safe two-pass FP4 decode 기준에서 정적 makespan이 **171429 → 190591**, 11.18% 증가했다. 컴파일은 성공했지만 NPU 미검증이고 F32 합산 그룹도 바뀌었다. `mlp.rs`는 실험 보존용이다.

초기 작업 중 fused FP4→FP8→F32→BF16 경로의 E13 NPU 실패가 보고돼, compile 전에 기준을 `ffn-blockscale-all-two-cluster-bf16-reuse`의 **FP4→FP8 DM→BF16 LUT DM**으로 바꿨다. Up/gate 및 GeGLU 전체 prefix는 그 safe-decode 기준과 byte-identical이다. 두 단계 변환이 E12 up/gate에서 검증됐다는 사실이 이 새 down layout이나 전체 block-scale 후보의 정확성을 대신하지 않는다.

## 배치와 수치 변화

Down H3840행은 2개의 Dummy2 cluster로 나누며 HBM view는 기존과 같은 `[Dummy2,H%1920,L]`이다. 각 cluster는 120개 유효 row group을 128개 physical group으로 pad하고 column group 2개를 사용한다.

- `DownRowsByColumns = m![H/16%120#128, L/7680]`
- `DownRows = m![H/16%120#128, 1#2]`
- slice별 H16 × L7680, 4행씩 4패스.
- scale VRF는 4 × 480 × 4 = 7680 bytes, activation TRF는 7680 × 2 = 15360 bytes.
- scalar FP4/scale 주소는 기존 행과 열의 주소 그대로다. 마지막 8개 padded row group은 typed output compaction에서 제외한 뒤 Dummy2를 H/1920으로 rename한다.

16개 원소의 BF16 dot→F32 block scale multiplication→F32 합산→최종 BF16 경계를 유지한다. 다만 기존은 120개 block의 local F32 sum 8개를 inter-slice sum했고 새 후보는 480개 block의 local F32 sum 2개를 더한다. 동일한 960개 항이지만 **부동소수점 합산 순서가 달라져 bitwise 동일성을 주장할 수 없다**. 이를 테스트 없이 안전한 치환으로 취급하지 않는다.

## 실제 비용

| 항목 | 8 columns, H60 | 2 columns, H16 |
| --- | ---: | ---: |
| makespan | **171429** | 190591 |
| instructions | 323 | 313 |
| DMA union | 135461 | 157920 |
| Main union | 132375 | 134094 |
| Vector union | 41849 | 41768 |
| Sub union | 22059 | 23296 |
| packed down HBM→DM | 29427 | 26218 |
| down scale HBM→DM | 9774 | 5255 |
| GeGLU activation→down layout | 9392 | **39136** |
| down FP4→FP8 | 9263 | 9863 |
| down FP8→BF16 | 18263 | 19463 |
| down blockdot/scale/reduce 합 | 8925 | 8844 |

긴 연속 HBM column segment는 목표대로 weight와 scale DMA를 줄였다. 그러나 더 많은 row group에 더 긴 activation segment를 복제하는 비용이 크게 늘어 전체가 느려졌다. 480개 유효 physical slices는 기존 512개보다 적어 slice별 decode 작업도 6.67% 많아졌다.

메모리의 단순 보수 합은 slice별 BF16 245760 + FP8 122880 + FP4 61440 + activation 15360 + scale 7680 = 453120 bytes다. 이것만으로 liveness를 보장하지는 않는다. 실제 compiler allocation도 성공했으며 schedule의 살아 있는 SRAM tensor physical-page union을 계산하면 기준/후보 모두 최대 16/16 pages, 268435456 bytes였다. Alias는 중복 집계하지 않았다. 따라서 이 수치는 free headroom이 있다는 주장도 아니며 compiler가 생성한 할당의 관측값이다.

## 재현과 근거

`mlp.eight-columns.rs`는 기준 snapshot, `ops.baseline.rs`는 같은 compiler 환경의 entrypoint snapshot이다. `change.patch`, `static-comparison.json`, `validate_layout.py`, `logical-index-proof.json`에 변환과 근거를 남겼다. Ops 전체를 live에 덮어쓰는 파일이 아니다.

Competition agent의 safe-decode scratch를 `/tmp/moa-ffn-down-two-columns`로 복사하고 별도 `/tmp/moa-ffn-down-two-columns/target`에서 `ops::decoder_feedforward --exact`만 컴파일했다. 공용 SDK cache를 읽는 `CARGO_HOME`은 `/tmp/moa-extra-seeds-e5-four-row-extra/cargo-home`이다. [기준 schedule](../../artifacts/ffn-down-two-columns/decoder_feedforward-baseline.json), [후보 schedule](../../artifacts/ffn-down-two-columns/decoder_feedforward.json), [컴파일 로그](../../artifacts/ffn-down-two-columns/compile.log). Live source, 다른 agent scratch, 공유 target, NPU, Git에는 변경하지 않았다.

후속 연구를 한다면 activation multicast 비용을 먼저 해결해야 한다. 더 긴 column segment의 HBM 개선만으로 추가 NPU 후보로 추천할 근거는 없다.
