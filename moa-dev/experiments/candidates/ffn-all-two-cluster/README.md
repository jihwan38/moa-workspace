# FFN up/gate/down 모두 2-cluster 분배

상태: `ops::decoder_feedforward --exact` 독립 컴파일 성공, 정적 makespan **213950**. **Arena job 70135에서 모든 공식 kernel × 3개 seed PASS**다. FFN 중앙값 **479755 cycles**, run별 485585 / 479755 / 479429. Up/gate-only 통합 실행 657198 대비 27.0% 감소했다. [E10 증거](../../../../docs/experiments/e10-ffn-all-two-cluster.json), [하드웨어 로그](../../../logs/e10-ffn-all-two-cluster-job-70135.log). 출발점은 [up/gate 2-cluster 후보](../ffn-upgate-two-cluster/README.md)의 최종 `mlp.rs`이며 up/gate projection, hier16 input 복제와 GeGLU는 유지한다. Local-scale multiplication을 이동시키는 block-scale 알고리즘은 포함하지 않는다.

## Down 분배

3840개 출력 H행을 `2 clusters × 32 row groups × 60 rows`로 나눈다. 각 row group은 기존과 같이 L15360을 8개 column group으로 나눠 slice별 L1920 dot product를 계산한다.

- `DownCluster = m![Dummy2]`
- `DownRowsByColumns = m![H / 60 % 32, L / 1920]`: cluster별 32 × 8 = 256 slices.
- `DownRows = m![H / 60 % 32, 1 # 8]`: 각 8-slice group의 F32 reduction 결과 위치.
- HBM weight `[H,L]`→`[Dummy2,H%1920,L]`, scale `[H,L/16]`→`[Dummy2,H%1920,L/16]`은 wire order를 유지하는 view reshape다.
- local H60행을 4행씩 15패스로 처리한다. 기존 H120행의 30패스와 같은 4행 BF16 dequant/contraction/F32 inter-slice reduction/BF16 direct commit을 사용한다.

행 `h=cluster*1920+row_group*60+local_row`, 열 `l=column_group*1920+local_column`의 packed nibble 주소는 원래 `h*15360+l`과 같다. local scale 주소는 `h*960+l/16`이고 `l/16=column_group*120+local_column/16`이다. 8개의 L1920 구간 순서와 cluster 내부 F32 reduction은 유지한다. BF16 dequant 및 output 경계도 유지하며 scale VRF는 4 × 120 × 4 = 1920 bytes다.

GeGLU는 기존 `Cluster/UpGateRowsPaired/L%120` 출력을 유지한다. 이를 곧바로 2-cluster `DownRowsByColumns/L%1920`에 typed DMA하여, 이전의 full-L 중간 복제를 없앴다. private `project_down`의 인자 layout만 바뀌며 device entrypoint 서명은 그대로다.

Down 출력은 cluster별 canonical `Slice/H%1920`으로 typed DMA한 뒤 Dummy2를 `H/1920`으로 reshape하고 기존 단일-cluster `Slice/H`로 typed copy한다. 이후 global-scale, post-RMSNorm, residual 경로는 그대로다. 이 출력 compaction은 2행 padding/copy 실패 후보와 다른 기존 4행 direct-commit 경로를 유지한다.

## 정적 비교

| 항목 | up/gate만 2-cluster | up/gate/down 2-cluster |
| --- | ---: | ---: |
| makespan | 299361 | **213950** |
| DMA union | 169594 | 132947 |
| Main union | 231763 | 178468 |
| Vector union | 189426 | 145131 |
| Sub union | 27759 | 22914 |
| instructions | 679 | 525 |
| down packed DMA | 58302 | 29427 |
| down FP4→FP8 LUT | 18263 | 9263 |
| down local-scale dequant | 66030 | 33015 |
| down contraction + 8-way reduction | 22560 | 11280 |
| down scale DMA | 18998 | 9774 |

Makespan은 28.53% 감소한다. 새 GeGLU→down input 분배 DMA 9392 cycles와 최종 출력 compaction 비용을 포함한 값이다. 정적 makespan을 실측 cycles나 speedup으로 간주하지 않는다. Main과 DMA 구간은 겹치므로 위 작업량을 단순 합산하지 않는다.

## 적용 및 검증

up/gate 2-cluster 후보가 이미 적용돼 있으면 `mlp.rs` 한 파일만 바꾸면 된다. `change.patch`는 정확히 그 후보 대비 diff이고 `mlp.upgate-only.rs`는 원본 snapshot이다. 다른 출발점에 적용할 경우 [기존 후보의 ops 연결 지침](../ffn-upgate-two-cluster/README.md)을 함께 따른다: FFN RMSNorm 출력을 private `feedforward`에 직접 전달하고 기존 ops의 activation 복제 블록을 제거해야 한다.

`validate_layout.py`는 3840개 H행 전체, 8개 column group, 대표 local column의 weight/scale index와 15개 four-row tile coverage를 확인한다. up/gate 및 GeGLU 소스가 출발점과 동일한지도 확인한다. 이 정적 검사는 NPU 정확성을 대신하지 않는다.

Scratch `/tmp/moa-ffn-all-two-cluster`, 별도 target `/tmp/moa-ffn-all-two-cluster-target`에서 선택한 FFN entrypoint만 컴파일했다. [Schedule](../../artifacts/ffn-all-two-cluster/decoder_feedforward.json), [컴파일 로그](../../artifacts/ffn-all-two-cluster/compile.log), 비교와 SHA256은 `static-comparison.json`에 기록했다. live source, 공유 target, Git, NPU job은 변경하지 않았다.

Job 70135에서 FFN max absolute error는 0.00781 / 0.01172 / 0.00781, mean은 0.000270 / 0.000380 / 0.000400으로 이전 four-row 경로와 표시 정밀도에서 동일하다. 이후 변경은 다시 공식 hardware harness로 검증한다. 특히 두 cluster의 H1920 경계, L1920 구간별 scale, cluster 내부 8-way reduction, output compaction을 확인한 뒤 FFN 중앙값을 비교해야 한다.
