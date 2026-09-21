E50은 E37의 down 가중치 decode만 `FP4 → FP8 DM → BF16 DM`에서 `FP4 LUT → FP8 → F32 → BF16 DM`으로 합친 후보다. 정확 컴파일의 makespan은 **162,209 → 157,638 cycles (−2.82%)**다. [Arena 70700 결과](../../../../docs/experiments/e50-down-fused-decode.json)는 공식 3입력의 15개 비교를 모두 통과했고, Q/O/FFN 중앙값은 **94,030 / 46,072 / 376,184 cycles**다. E37의 반복 FFN 중앙값 379,741보다 0.94% 빠른 첫 측정이며, E50 반복 측정·추가 입력 검증은 아직 없다. 이 결과만으로 채택 또는 보편적인 수치 동등성을 주장하지 않는다.

## 변경과 실행 경로

[candidate.patch](candidate.patch)는 `project_down_sharded`의 decode 블록 하나만 바꾼다. 동결 E37/E50의 `src/` 43파일을 대조하면 `device/shared/mlp.rs`만 다르며, 이 폴더의 [mlp.rs](mlp.rs)는 E50 캡처 소스와 byte-identical이다. up/gate의 두 단계 decode와 down dot·scale·reduction·epilogue는 그대로다. E48 up-only 또는 E49 up+gate fusion을 포함하지 않는다.

실제 호출은 동결 [ops.rs](../../artifacts/e50-down-fused-decode/src/ops.rs)의 `decoder_feedforward`(203행) → `feedforward_sharded`의 up/gate → `geglu` → `project_down_sharded`(373행) → down global scale → post-FF RMS → residual/layer scale → HBM write다. 이 후보가 바꾸는 부분은 다음뿐이다.

1. HBM packed weights를 기존 `DownCluster=Dummy2`, `DownRowsByColumns=[H/60%32,L/1920]`, Element `[H%60,L%1920]`에 적재한다.
2. FP4→FP8 lookup 결과를 DM에 쓰지 않고 `fetch_cast::<f32>()`로 넘긴다.
3. 8개의 논리 원소를 BF16의 물리 16원소 packet으로 cast하고 `commit_trim::<L%8>()`으로 유효 8개만 저장한다. 기존 BF16 weight matrix가 만들어지는 경계는 유지된다.
4. 기존 12행 × 5회 dot이 BF16 weights/activation을 사용한다. 16원소 dot → local FP32 scale → 120 block FP32 sum → 8 column-slice FP32 sum → BF16 4원소 commit 순서도 같다.

`h=1920*cluster+60*row_group+local_row`, `l=1920*column_group+local_column`이며, 2클러스터 각각 32행 그룹 × 8열 그룹 = 256개 slice가 모두 유효하다. 기존 HBM reshape를 그대로 사용하며 새 reshape 또는 padding을 값으로 읽는 변환은 추가하지 않았다.

FP4의 ±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6은 F8/F32/BF16에 정확히 표현된다. 이는 scalar 변환의 성질이며, LUT와 fetch cast를 연결한 실제 NPU pipeline의 정확성을 단독으로 증명하지는 않는다. 70700은 이 통합 경로가 해당 공식 3입력에서 허용오차를 만족한다는 직접 근거다.

## 정적 비용과 남은 대기

E37 baseline은 [baseline.json](../../artifacts/ffn-down-scale-coalesced/baseline.json), 후보는 [candidate.json](../../artifacts/ffn-down-fused-decode-e37/candidate.json)이다. baseline FFN image는 E37 clean-build image와 byte-identical이다. 표의 엔진 비용은 해당 context의 실행 구간 합집합이다. Main/Vector 등의 구간은 겹치므로 세로로 합산하면 안 된다.

| 항목 | E37 | E50 | 차이 |
|---|---:|---:|---:|
| 전체 makespan | 162,209 | 157,638 | −4,571 |
| Main context | 125,274 | 114,211 | −11,063 |
| DMA engine | 135,607 | 134,769 | −838 |
| Sub context | 19,378 | 17,826 | −1,552 |
| Vector engine | 34,748 | 51,211 | +16,463 |
| PeCore | 29,352 | 29,352 | 0 |
| 스케줄 instruction 수 | 242 | 239 | −3 |

두 decode Main instruction 9,263 + 18,263 cycles가 Main/Vector를 동시에 쓰는 16,463 cycles instruction 하나로 바뀌었다. DMA에서는 838 cycles의 상수 load 하나가 사라진다. 큰 down packed weight DMA **29,427**, local scale DMA **9,774**, activation copy **9,392 cycles**는 각각 그대로다. 이 후보는 큰 HBM weight/scale 전송량을 줄이는 변경이 아니다.

후보 decode node 419는 `[120063,136526)`에 실행된다. 그런데 scale DMA node 421이 `[129455,139229)`까지 계속되어 첫 down dot node 460은 139,830에 시작한다. baseline 첫 dot은 144,401에 시작한다. Main 작업 11,063 cycles를 줄여도 전체 단축이 4,571 cycles인 이유를 이 일정에서 확인할 수 있다. 실제 하드웨어의 모든 대기 원인을 이 정적 일정만으로 단정하지 않는다.

slice당 115,200B, 전체 58,982,400B의 FP8 중간 tensor가 사라진다. packed FP4 57,600B/slice와 BF16 230,400B/slice는 남는다. 스케줄에서 packed tensor 수명은 baseline `[84977,124922)`에서 후보 `[84913,136526)`으로 길어진다. 중간 tensor 제거량을 peak DM 감소량으로 그대로 해석할 수는 없다.

## probe와 통합 검증의 범위

[tagged full-shape probe 70658](../../../../docs/experiments/fp4-decode-probe-fullshape-tagged.json)는 2클러스터 × **240 유효/256 물리 slice × 32행 × 3840열**이다. E50 down은 2클러스터 × **256 유효 slice × 60행 × 1920열**이다. 총 FP4 원소 수는 둘 다 58,982,400개지만 row/slice mapping, packet 진행, allocation과 lifetime이 다르다. probe 통과를 down shape 또는 full-FFN의 증명으로 옮겨 쓰지 않는다. probe는 DPE/local scale과 전체 FFN 실행 이력도 포함하지 않는다.

70700의 FFN 세 실행은 390,867 / 375,881 / 376,184 cycles이며 모두 PASS다. QKV/O는 E37 이미지와 같다. 이전 E13 fused 경로 실패 원인은 이 결과로 규명되지 않았다. 후속 채택 판단에는 E50 반복과 추가 입력을 별도로 확인해야 한다.

## 동결 파일과 감사 범위

해시·주요 스케줄 node·tensor·로그는 [evidence.json](evidence.json)에 기록했다.

| 파일 | SHA-256 |
|---|---|
| `mlp.rs` | `f4f541747934620d0a617549c79d392bc9f75bc7cb4e633ddfc91ddc042bcd34` |
| FFN image, 200,960B | `47ad23c027e27e255f2b0a0e1208d653c3de3594de26630b67f8dd1766700ad6` |
| 공식 host executable, 3,666,752B | `8bfc029fe361b017dd8b2e8bc8a7e9dc372451b7c4c5a4cbf08237517d2b0c48` |

동결 host ELF에 후보 FFN image와 E37 QKV/O image가 각각 정확히 1회 포함됨을 직접 검사했다. 70700에서 다운로드한 실행 파일·fixture 해시도 캡처와 일치한다. root의 [정확 컴파일 로그](../../../logs/e50-down-fused-decode-compile.log)와 [full-build 로그](../../../logs/e50-down-fused-decode-build.log)의 성공을 확인했으며, 이 감사에서는 재컴파일·NPU 제출·Git mutation·live source 변경을 하지 않았다. 캡처 도구 자체의 `source_binary_relationship.verified=false`는 유지한다. 캡처만으로 빌드 연결을 증명하는 대신 동결 소스 대조, 빌드 완료 로그, image byte 비교를 별도로 사용했다.

원본 큰 artifact는 Git-ignored `artifacts/`에 있다. E50 graph dump는 제공되지 않아 physical DMA descriptor 변경에 대한 주장은 하지 않는다. 현재 working tree의 후속 실험 상태와 이 동결 후보는 독립이다.
