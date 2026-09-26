# E124 corrected-v2 독립 전송 계약 검토

2026-09-27. 후보 source/graph와 `analyze_corrected.py corrected-v2 --check`를
별도로 검토했다. 구체 구현 오류는 발견하지 못했다. 아래는 NPU 결과와 구별되는
source/descriptor 검토이며 새 compile·Arena·후보 수정은 하지 않았다.

- Weight O24/O31은 K/V IO(2)/IO(3)의 compact DM을 읽어 joint slot2/3에 쓴다.
  Slice당480×32 FP8=15360B를 읽고 쓰며 destination은 각각
  [30720,46080) / [46080,61440)B다.
- Scale O45/O52는 IO(5)/IO(6)의4BF16=8B만 fetch·commit한다. Collect는
  16BF16의32B packet으로 padding하지만 commit_trim이 유효4개만 쓰므로
  destination은 [16,24) / [24,32)B다.
- 여섯 weight/scale HBM source의 coverage와 마지막 주소가 allocation에 맞다.
  Analyzer에서 직접 assert하지 않는 HBM→DM destination row 대응도 별도로
  packet을 열거해 확인했다. Source bounds만으로 destination mapping을 추정하지 않았다.
- Joint weight T50 / scale T71의 Q→K→V IndexWrite 의존성이 유지된다.
  Scale VRF O56→dot O59이며 모든 복사가 완료된 뒤 소비한다. Q/K/V output
  gather의 byte 대응도 통과했다.
- corrected-v2/projection.rs의 산술 부분(:322 이후)과 ops.rs/rmsnorm.rs/rope.rs는
  frozen-v2와 바이트 단위로 같다. F32 dot→channel scale→첫 BF16 cast와
  head/RoPE/cache 경계는 유지된다.

이 검토 시점에는 Arena105093이 대기 중이었다. 정적 검토의 PASS를 NPU
정확도·성능 PASS로 해석하지 않는다. 실제 실행 결과는 별도의 E124 보고서를 따른다.
