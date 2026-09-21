# FFN up/gate 2-cluster row distribution

상태: 최종 `mlp.rs`는 독립 scratch에서 `ops::decoder_feedforward --exact` 컴파일 성공. 정적 makespan **299361**이다. **Arena job 70118 통합 실행에서 공식 모든 kernel × 3개 seed PASS**다. FFN 중앙값은 657198 cycles(run별 658639 / 657198 / 655590). 이 실행에는 root의 norm32·parallel residual 변경도 포함돼 있으므로 FFN 속도 차이를 up/gate 분배만의 효과로 해석하지 않는다. [하드웨어 로그](../../../logs/e9-upgate2-norm32-parallel-residual-job-70118.log). 출발점은 job 70070에서 PASS한 four-row down + MLP/residual 전체 반복 전개 버전이다.

## 행 분배와 수치 경계

up/gate의 15360개 출력 행을 `2 clusters × 240 valid slices × 32 rows`로 나눈다. 물리 slice는 각 cluster 256개이며 마지막 16개는 padding이다. 각 slice는 4행씩 8패스를 수행한다. 원래 1 cluster × 256 slices × 60행의 15패스보다 per-slice 작업을 줄이면서 기존 4개 BF16 출력의 8-byte commit을 유지한다.

- `UpGateCluster = m![Dummy2]`, `UpGateSlices = m![L / 32 % 240 # 256]`.
- weight HBM `[L,H]` view를 `[Dummy2,L%7680,H]`로 reshape하고 typed DMA한다. row `r=c*7680+s*32+t`의 packed FP4 nibble 위치는 `r*3840+h`로 같다. scale도 `[Dummy2,L%7680,H/16]`로 같은 행과 column block을 유지한다.
- slice-local scale VRF는 4 × 240 × 4 = 3840 bytes로 8192-byte 한도 이하다.
- FP4→FP8 LUT, F32 local-scale multiplication, BF16 dequant, 전체 H contraction, BF16 결과 경계는 유지한다. H 축을 나누지 않아 각 행의 dot-product reduction 구조를 바꾸지 않는다.
- up/gate 출력을 typed DMA로 cluster별 연속 `[L%7680]`에 모아 padding을 제거한 뒤, Dummy2를 `L/7680`으로 reshape하고 기존 `Cluster/UpGateRows`에 typed copy한다. `geglu`와 `project_down` 소스는 출발점과 byte 단위로 동일하다.

원안인 2 clusters × 256 slices × 30행, 6행씩 5패스는 BF16 12-byte output packet을 요구한다. SDK packet은 8/16/24/32-byte이므로 이 후보에서는 4행 commit을 유지했다. 과거 2행 down 후보의 padding/copy 경로가 실측 FAIL한 이력 때문에 6행 padding을 정확한 대체로 가정하지 않았다.

`validate_layout.py`와 `logical-index-proof.json`은 모든 15360개 행의 일대일 분배, representative column들의 packed/scale index, 4행 타일의 연속 coverage, GeGLU/down 불변을 확인한다. 정적 증명은 RNGD correctness test를 대신하지 않는다.

## Activation 복제와 API 연결

private `feedforward`의 x 인자는 `Cluster/Slice` 단일 slice로 바뀐다. 함수 내부에서 각 cluster의 16-slice ring leader로 TDMA한 뒤 `CustomBroadcast { ring_size: 16 }`으로 activation을 채운다. 모든 physical slice가 같은 x를 가지므로 마지막 16개 padding slice를 제외한 view로 안전하게 해석할 수 있다. `fetch::<m![1],m![H]>()`로 H 전체를 switch packet에 두어야 한다. H/16을 time에 두면 compiler가 `Switch read is not sequenceable`로 거절한다.

**적용 시 `ops::decoder_feedforward`의 기존 x replication 블록을 제거하고 RMSNorm 출력을 직접 넘긴다. device entrypoint의 외부 서명은 바꾸지 않는다.** `change-ops.patch`는 원래 TDMA 한 줄 삭제를 보여준다. 현재 root의 `ops.rs`에는 다른 QKV/O/FFN broadcast 변경이 있으므로 candidate의 전체 `ops.rs`를 덮어쓰지 않는다. 이미 hier16 블록이 있다면 해당 FFN 블록 전체를 제거해야 한다.

## 정적 측정

| 항목 | 1-cluster 전체 전개 + 기존 direct copy | 2-cluster + direct copy | 최종 2-cluster + hier16 |
| --- | ---: | ---: | ---: |
| makespan | 392312 | 414078 | **299361** |
| DmaEngine union | 263227 | 266795 | 169594 |
| MainContext union | 319304 | 227660 | 231763 |
| VectorEngine union | 264270 | 189426 | 189426 |
| SubContext union | 33121 | 27759 | 27759 |
| instructions | 808 | 675 | 679 |

2-cluster direct copy는 x 복제만 106272 cycles를 써 전체가 느려졌다. hier16은 이를 줄여 최종 makespan을 낮춘다. 따라서 392312→299361의 23.7% 감소에는 row 분배와 activation 복제 두 효과가 포함된다. 이 값을 NPU speedup으로 해석하지 않는다.

같은 hier16 activation 복제를 적용한 단일 cluster 비교군은 makespan **382949**, DMA 214986, Main 323407, Vector 264270 cycles다. 동일 복제 전략에서 2-cluster 후보는 makespan이 21.83% 감소한다. [단일 cluster + hier16 schedule](../../artifacts/ffn-upgate-two-cluster/decoder_feedforward-single-hier16.json).

Up/gate 각각 packed weight DMA 49157→26475, LUT 18263→9863, local-scale dequant 61815→32968, contraction 18375→9800 cycles다. Down dequant 합 66030과 down packed DMA 58302는 남는다. Padding 제거와 원래 layout으로 출력 복원하는 작은 typed DMA도 schedule에 포함돼 있다.

## 실패 기록과 재현

- `mlp.slice-tile-failed.rs`: 기존 Replicated x에서 `slice_tile`로 leader를 선택하려 했으나 NPU lowering이 non-in-slice axis tiling을 지원하지 않는다. private 입력을 단일 slice로 바꿨다.
- `mlp.same-axis-failed.rs`: Cluster=`L/7680`, Slice=`L/32%240#256`은 HBM→DM 시 `cannot factorize common axis L`로 실패한다. high axis를 Dummy2로 분리한 동일 wire-order reshape가 해결한다.
- `mlp.direct-replication.rs`: axis rename 후 컴파일 성공한 direct-copy 비교군.
- `mlp.single-cluster.rs`: 출발점 snapshot. `mlp.single-cluster-hier16.rs`는 같은 hier16을 단일 cluster에 적용한 별도 비교용이다.

Scratch `/tmp/moa-ffn-upgate-two-cluster`, target `/tmp/moa-ffn-upgate-two-cluster-target`만 사용했다. [최종 schedule](../../artifacts/ffn-upgate-two-cluster/decoder_feedforward.json), [컴파일 로그](../../artifacts/ffn-upgate-two-cluster/compile.log), failure 로그와 SHA256은 `static-comparison.json`에 기록했다. live source, 공유 target, Git, NPU job은 변경하지 않았다.

Job 70118의 FFN max absolute error는 run별 0.00781 / 0.01172 / 0.00781, mean은 0.000270 / 0.000380 / 0.000400으로 기존 four-row와 표시 정밀도에서 같다. 이후 변경도 정확한 후보 SHA를 사용한 공식 3개 kernel × 3개 seed harness와 FFN run별 cycle로 검증한다. 특히 두 cluster의 행 경계 7680, slice padding 제거, 4행 commit, up/gate의 기존 layout 복원이 하드웨어에서 맞는지 확인한다. 여기 기록된 통합 PASS가 이후 수정본의 정확성을 대신하지 않는다.
