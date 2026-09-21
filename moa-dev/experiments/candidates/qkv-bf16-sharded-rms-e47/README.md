# QKV BF16 입력 보존 + E47 sharded RMS

현재 공식 입력 실패를 고친다는 주장이 아니라, 더 넓은 모델 입력 범위를 위한 정밀도 대안이다. E47 FP8 입력 quantizer는 아직 공개 QKV9개 구성에서 모두 통과했다. 별도 CPU 상쇄 반례는 공식 QKV generator가 생성하는 x 범위 밖이며 NPU에서 실행하지 않았다.

`projection.rs`의 QKV helper는 E18의 BF16 lookup+DPE 경로를 그대로 재사용한다. `ops.rs::sliding_project_qkv`는 E47의 H480×8 pre-RMS를 유지한 뒤 E18의 두 cluster/ring32 계층 BF16 broadcast를 실행한다. O와 FFN은 E47이며 QKV head RMS/RoPE/cache write도 그대로다. 공개 device signature는 바꾸지 않았다. 원본/patch를 함께 저장했다.

E18 helper는 raw dot을 BF16으로 반올림한 뒤 channel scale을 적용한다. BF16 입력 소실은 제거하지만 이 반올림 경계와 FP32 합산 순서가 원래 Python reference와 완전히 같다는 보장은 없다. 향후 channel-scale-before-cast를 별도로 검토할 수 있다.

Isolated exact compile은 Dummy256 import 누락1회를 수정한 뒤 통과했다. Static makespan52,282는 E47의50,364보다3.81% 크다. Image/소스/schedule hash는 [manifest](manifest.json)에 있다. **NPU 정확성·성능은 미검증이며 채택하지 않았다.** Live source를 바꾸지 않고 scratch에서만 compile했다.

통합 주의: 이 frozen 후보의 O suffix는 E47이다. 파일 전체를 E53 위에 복사하면 O 정확성 수정을 되돌린다. 향후 실험은 QKV prefix와 ops의 QKV 본문만 E53에 결합하고 다시 소스·image·정확성을 검증해야 한다.

후속 [E54](../qkv-bf16-f32-scale-e53/README.md)는 QKV 부분만 E53에 통합하고 raw dot·행 재배치를 F32로 유지해 channel scale 뒤 첫 BF16 cast를 둔다. 이 후속 구현의 NPU 결과를 위 early-cast 초안의 검증 결과로 대신하지 않는다.
