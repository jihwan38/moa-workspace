# 전체 blockscale-after-dot, 두 단계 BF16 decode

`mlp.rs`는 E12에서 NPU 통과한 `ffn-blockscale-two-cluster-bf16-reuse`의 up/gate, feedforward, GeGLU prefix를 그대로 유지하고 down에도 blockscale-after-dot을 적용한 별도 후보다. Up/gate/down 모두 packed FP4 → FP8 DM → BF16 table lookup → unscaled BF16 DM의 **두 단계 decode**를 사용한다. Fused LUT+fetch_cast 경로가 parent job70162에서 FFN FAIL/다음 launch timeout을 일으켜 그 경로를 쓰지 않는다. 실패 후보는 별도 폴더에 원본과 로그 증거를 보존했다.

Down 배치는 2클러스터×32 row group×60행이며, 각 row group의 8개 column slice가 1920개 열씩 처리한다. Full BF16 weight를 한 번 만들고 12행 tile 5회에 재사용한다. 16원소 FP32 block dot → FP32 local scale 곱 → 120개 block FP32 합 → 8 slice FP32 합 → BF16 cast 순서다. 12행 출력은 4행 packet 3개로 commit한다. Slice당 packed 57600B + FP8 115200B + BF16 230400B =403200B(다른 tensor 및 liveness 제외), scale VRF는 5760B다. Compiler 실제 배치는 spill 없이 통과했다.

Exact `ops::decoder_feedforward` compile 통과, 정적 **171429 cycles**. E12의 181155보다 5.37% 낮다. Main 132375, Vector 41849, DMA 135461, Sub 22059다. 전체 lookup DRAM 상수 6개, spill I/O 0. Root가 확인한 E12 공식 15 comparisons PASS/FFN 436380 cycles는 **원본 up/gate 경로의 결과**이며 이 후보의 새 down 정확도·성능을 증명하지 않는다. Raw schedule은 ignored `../../artifacts/ffn-blockscale-schedules/all-bf16-reuse*`에 있다.

`validate_candidate.py --output NEW.json`은 up/gate/GeGLU byte-identical, 3840행·15360열·960개 scale block coverage, local FP32→8 slice FP32→BF16 순서, 전체 유한 FP4×FP8 4064개 곱의 BF16 exactness를 검사한다. 결과는 `numerical-layout-proof.json`이며 이 검사도 실제 컴파일된 packet 경로의 정확성을 대신하지 않는다. FP32 누산 순서가 바뀌므로 bitwise identity는 주장하지 않는다.

Overflow는 조건부 증명만 가능하다. 전체 유한 FP4/FP8 코드 범위에서 down 입력 `|x|<=2^101`이면 보수적인 rounding 여유를 포함한 모든 중간값과 BF16 출력이 유한 범위다(global scale 이전). 그러나 16개 x=`2^124`, q=1, scale=1/512이면 기존 scaled dot은 `2^119`로 유한한 반면 unscaled dot은 `2^128`로 FP32 범위를 넘는다. 이런 extreme 입력에 대한 무조건적 동치는 성립하지 않는다. NaN/Inf, flush-to-zero, global scale 후 overflow는 이 증명 범위 밖이며 runtime guard나 fixture 특수 분기는 추가하지 않았다.

`mlp.e12-parent.rs`는 통과한 up/gate 원본, `mlp.failed-fused-parent.rs`는 변경 전 전체 fused 원본이다. Source/도구/proof hash와 compiler 기록은 `source-hashes.json`, `evidence.json`, `compile.log`, `static-comparison.json`에 있다. 동일 ops를 scratch에 포함했으며 실제 반영은 `change-ops.patch`의 기존 입력 복제 한 줄만 제거하면 된다. 이 에이전트는 공식 source/live binary/Git/NPU 제출을 변경하지 않았다.
