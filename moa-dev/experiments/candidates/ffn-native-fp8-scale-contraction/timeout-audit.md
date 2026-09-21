# E22/E25 native FFN timeout 공통 경로 감사

2026-09-21. E22 job70321과 E25 job70354의 frozen source, provenance, 원시 로그, SDK/std/lower 0.8.1과 scratch schedule을 읽었다. E25 source를 수정하지 않았고 추가 NPU 실행은 하지 않았다. 근본 원인은 아직 확인하지 못했다.

두 로그 모두 Q/O 검증을 끝낸 뒤 첫 `decoder_feedforward` launch에서 `rank 1 ... launch ran past its time limit`를 반환하고 host가 exit101로 종료했다. FFN output/cycle은 없다. `rank 1`을 특정 cluster 또는 특정 instruction의 실패라고 해석할 근거는 로그에 없다. E22/E25의 ops hash와 shared RMSNorm hash는 같고 MLP hash는 각 frozen 후보와 일치한다. 첫 kernel 호출 내부 어느 단계까지 진행했는지는 알 수 없다.

## 확인한 공통 경로

E25 quantizer와 ring32 broadcast 전체 source는 `FfnTerm→Dummy2` 및 helper/type 이름 치환 후 기존 QKV two-lane 후보의 quantizer와 byte-identical하다. 두 clusters의 canonical input에서 amax와 power-of-two s를 구하고, 8 slices에 H480씩 배분해 hi/lo를 만든 뒤 packed `[term,H]`를 512 slices로 복제한다. E22도 같은 경로를 사용하며 마지막 runtime scalar만 reciprocal `1/s`로 반환한다. E25는 reciprocal pass를 사용하지 않아 두 timeout을 reciprocal 하나로 설명할 수 없다.

FFN 소비 layout은 `Cluster=Dummy2`, `Slice=L/32%240#256`, Element=`FfnTerm,H`다. 원래 512 physical copies를 두 named clusters와 240 valid+16 padded slices로 바꾸며, 유효 slice가 읽는 term/H 순서는 변하지 않는다. cluster c, valid slice s, local row r의 weight row는 `7680*c+32*s+r`다. 15,360 rows의 전수 대응을 확인했다. cluster `Dummy2`와 lane `FfnTerm`을 구분하여 축 이름 충돌도 없다.

FP4 HBM→FP8 DM lookup, 같은 padded 240-slice row mapping, 마지막 compaction은 E23 streamed BF16 scale-contraction에서 통과했다. 따라서 이들 각각이 단독으로 잘못됐다고 결론내릴 수 없다. 다만 native FP8 pipeline과 결합했을 때의 lowering은 별도 검증 대상이다.

두 실패 후보의 공통 DPE 경로는 `contract_packet<H/16%4>`로 64-element packet에서 네 block 결과를 유지하고, `contract_time<row,H/64>`가 60개 packet time을 유지한 뒤 two-lane Sequential→pair unzip으로 보낸다. `H=64*t+16*b+i`의 전수 대응은 정확하다. E22는 row4, E25는 row32지만 Packet=`H/16%4#8`인 점은 같다. 반면 launch가 완료한 native FP8 QKV와 oracle를 통과한 native O는 packet reduction을 scalar까지 끝내고 pair에 전달한다. **네 block을 유지하는 native DPE→Sequential→pair 조합은 더 좁은 공통 미검증 구간**이다. 이것이 timeout 원인이라는 증거는 아직 없다.

SDK lower의 `config_contract_lane` 검증상 Sequential의 accumulator 요구량은 lane2 × inner-time1 × padded packet32 = 64 cells로 1024-cell 한도 이내다. 전체 row×60 Time 길이를 이 accumulator 용량과 곱해 overflow라고 주장하면 잘못이다. `config_vector_intra_slice_unzip`와 narrow/widen 타입 검증도 두 후보에서 통과했다. 실제 lowering/실행 정확성까지 보장하는 검사는 아니다. pair의 narrow_trim/widen_pad는 금지되어 있으며 두 후보는 허용된 buffered narrow_split/widen_concat을 쓴다.

## Schedule 확인과 한계

E22 packed ring broadcast는 79,437→87,380, activation TRF 적재는 87,380→88,603이다. 첫 native block contraction은 88,603→89,847, 마지막은 128,456→129,700이다. activation TRF는 그 전체 구간 동안 주소0에서 유지되고 이후 down TRF는 다른 주소에서 146,322 이후 시작한다.

E25 broadcast는 79,414→87,357, TRF 적재는 87,357→88,580이다. 첫 native block contraction은 88,820→96,784, 두 번째는 117,826→125,790이다. activation TRF 주소0은 125,790까지 유지되고 down TRF는 142,635 이후 같은 주소를 재사용한다. s VRF는 75,428→125,790 유지된다. 주요 producer/consumer와 register lifetime에서 겹치는 입력 덮어쓰기를 확인하지 못했다. metadata alias join에 표시된 짧은 overlap을 실제 data hazard라고 해석하지 않았다.

이 수치는 compile schedule이며 hardware가 해당 시점까지 도달했다는 trace가 아니다. 스케줄 그래프의 DAG와 finite makespan은 runtime stream handshake가 끝난다는 증거도 아니다. E22와 E25는 F32 partial+VE reduce / BF16 partial+second DPE라는 서로 다른 후반 경로를 사용하므로 후반 reduction 하나만 원인으로 지목하기는 어렵지만, 실행 위치를 모르므로 완전히 제외할 수도 없다.

## 최소 진단 제안

추가 대형 최적화 sweep 대신, 먼저 packet4를 유지하면서 **Sequential two-lane/pair만 없앤 두 lane1 dot**을 별도 scratch에서 compile하는 것이 구분력 있는 진단이다. 정확한 Interleaved lane-folder 규칙상 lane1의 output Packet은 `1#8`이며 기존 packet4 block 축은 OutTime으로 옮겨야 한다. `Packet=block4#8`을 그대로 선언하면 잘못된 mapping이다. scalar block을 compact F32 DM/VRF에 보관하는 경로가 SDK와 용량 내에서 성립하는지 먼저 확인해야 한다. 컴파일 실패 시 실패 근거를 남기고 멈춘다.

그 경로도 hang하면 native FP8 packet4 또는 padded-slice 연결이 남고, 통과하면 제거한 Sequential/pair 또는 바뀐 scheduling을 더 조사할 수 있다. 어느 결과든 원인을 하나로 단정하지 않는다. instruction-level trace가 없고 추가 분리가 필요하면 별도 diagnostic kernel에서 quantizer+broadcast 출력을 HBM으로 반환하는 단계를 먼저 검증하는 방법이 있다. 이 경우 실제 전체 512 copies와 padding 경계를 관찰해야 하며, 일부 sample slice만 읽어 전체 정상으로 일반화하면 안 된다. 이 문서 작성 중 해당 NPU 진단은 실행하지 않았다.

확인한 primary sources: SDK `src/engine/contraction/lane.rs`, `src/engine/vector/tensor/vector_tensor_pair.rs`, `src/engine/vector/tensor/verify.rs`; lower `src/verify/engine/contraction/lane.rs`, `src/verify/engine/vector/tensor.rs`, `src/verify/mod.rs`. 정확한 source/schedule 증거는 `timeout-audit-evidence.json`에 있다.
