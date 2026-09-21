# E47 shared helper의 실제 호출 범위

2026-09-21. [기계 판독 기록](e47-shared-caller-audit.json)은 source43개를 비교하고 `ops::full_attention_output`을 두 번 exact compile한 결과다. 전체 모델 또는 full-attention NPU 정확성 검사를 수행한 기록은 아니다.

[ops.rs](../../furiosa-opt-gemma4-12B/src/ops.rs)의 `full_attention_output`은 full attention의 softmax 합으로 나눈 입력을 projection한 뒤 `shared::rmsnorm::normalize` → `shared::residual::add` → HBM write를 호출한다. [shared/rmsnorm.rs](../../furiosa-opt-gemma4-12B/src/device/shared/rmsnorm.rs)는 원본 전체가 그대로 prefix로 남고 새 sharded helper만 추가됐다. 이 호출자의 normalize는 변경되지 않았다.

반면 [shared/residual.rs](../../furiosa-opt-gemma4-12B/src/device/shared/residual.rs)의 `add`는 H=3840의 480원소 타일8개를 처리하는 기존 loop를 정적 매크로 호출8개로 펼쳤다. 타일 주소, F32 add와 BF16 cast는 그대로다. 독립 타일을 compiler가 다르게 배치할 수 있으므로 full-attention image도 달라진다. 대조 소스는 E47에서 **이 함수만** 원본 loop로 복원했으며 새 helper와 나머지42개 source는 유지했다.

| exact compile | 정적 makespan | device image bytes |
|---|---:|---:|
| E47 현재 shared add | 462,598 | 142,336 |
| 원본 loop만 복원한 대조 | 464,953 | 126,976 |

차이는 −2,355 cycles(−0.507%)다. 이는 schedule 비교이며 실제 장치 속도 또는 정확성의 증거로 사용하지 않는다. 두 image/schedule/source SHA256은 JSON에 기록했다.

[host/runtime.rs](../../furiosa-opt-gemma4-12B/src/host/runtime.rs)의 `Decode::sliding`과 `Decode::full`은 모두 `self.mlp`를 호출하고, `mlp`는 `ops::decoder_feedforward`를 launch한다. 따라서 E47의 FFN 변경은 text decoder의 두 attention 종류 모두에 적용된다. Stage1 harness는 해당 커널의 별도 입력만 검사하며 모델 전체의 상태와 입력 분포를 검사하지 않는다.

직접적인 vision shared 호출은 [ops_vision.rs](../../furiosa-opt-gemma4-12B/src/ops_vision.rs)의 `shared::residual::add_vision`이며 이 함수 본문은 원본과 같다. `ops_audio.rs`, `ops_vision.rs`, `host/runtime.rs`도 원본과 byte-identical하다. 다만 vision/audio embedding 이후 text decoder로 들어가므로 이를 멀티모달 전체 영향 없음으로 해석하지 않는다. Stage2의 실제 workload가 나오면 이 호출 흐름을 따라 상태 보존과 끝까지의 모델 출력을 검사해야 한다.
