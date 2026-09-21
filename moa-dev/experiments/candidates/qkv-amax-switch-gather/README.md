# QKV sharded amax의 Switch gather 대조 후보

**Exact QKV compile PASS, 전체 정적 시간은 E29의50,364에서50,853 cycles(+489, +0.97%)로 늘었다.** 국소 amax 비용 감소를 전체 개선으로 해석하지 않는다. NPU는 아직 실행하지 않았다.

Frozen `e29-sharded-qkv-packed-output` 전체 source를 독립 scratch에 복사하고 기준 QKV를 다시 compile했다. 기준 kernel SHA는 E29의 `91b232…`와 같다. 수정은 `projection.rs::prepare_fp8_split`의 전역 amax 블록 하나뿐이다. ops/shared RMS와 Q/K pair·V two-dot·head RMS/RoPE는 그대로다. **E29의 O direct-packed quantizer도 byte-identical하게 포함된다.** E31 RMS gather는 섞지 않았다.

각 cluster 입력 H3840은 slice0–7에480개씩 들어 있다. 기존 local abs/max/floor 결과 scalar8개를 typed `switch::<Slice,m![H/480]>(Broadcast1 { slice1:8, slice0:1 })`로 slice0의 Time0–7에 모은 뒤 F32 intra-slice Max로 하나를 선택한다. 기존 exponent mask, ×1/128 scale, scalar replication과 packed quantization은 그대로다. per-shard scale을 쓰지 않는다.

새 unsafe reshape는 없고 padding을 유효 Time 축으로 해석하지 않는다. 양수 floor를 적용한 유한 scalar8개의 Max는 입력을 선택하므로 순서에 따른 반올림이 없다. signed zero도 floor 때문에 사라진다. 유한 BF16의65,280개 bit pattern을 독립 CPU로 확인했다. 범위는 [math-layout-verification.json](math-layout-verification.json)에 있다. NaN 처리 순서 및 장치 구현을 CPU 검사로 보증하지 않는다.

| 정적 항목 | E29 기준 | 후보 |
| --- | ---: | ---: |
| 전체 QKV |50,364|50,853|
| instructions |171|171|
| amax node |2,562|289|
| scale scalar 완성 |8,516|6,243|
| replicated scale 완성 |13,933|13,933|

Node 본문/type/duration multiset은 amax 한 개 외 모두 같다. scale 계산이 빨라져도 TDMA가 weight load를 수행하는 동안 scalar copy가 기다려, 최종 replicated scale은 둘 다13,933에 준비된다. 이후 cos/sin DMA 순서와 Q/K epilogue가 재배치된다. 기준 cos gather32,757–33,691이 후보에서는43,394–44,328로 이동하고 전체 끝은489 cycles 늦어진다. compiler가 이 순서를 선택한 내부 이유까지 증명한 것은 아니다.

[Evidence](evidence.json)에 source/kernel/schedule hashes를 기록했고 최소 diff는 `candidate.patch`다. prerequisite ops/rmsnorm 사본은 E29 재현용이다. 더 최신 소스에 적용할 때는 전체 파일 덮어쓰기보다 amax 블록 diff만 반영해야 후속 RMS/O 개선을 되돌리지 않는다. Raw schedule/kernel/log는 `../../artifacts/qkv-amax-switch-gather/`, 독립 Docker scratch/target은 `/tmp/moa-qkv-amax-switch-gather`다. live source·Git·NPU는 변경하지 않았다. 정적 성능만으로 채택을 추천하지 않는다.
