# E28 identity / E30 skip 대조군 검증

E30의 QKV는 E28과 두 group0 Mul0 인자만 다르다. E28 `1.0f32`를 E30 `()`로 바꾼 두 줄 외에 `projection.rs` 전체가 같고, `ops.rs`도 byte-identical이다. 전체 frozen source에서 추가 차이는 FFN `mlp.rs`를 E27 안전 버전으로 복원한 것뿐이다. E29의 sharded input/amax 및 direct-packed O 변경은 들어가지 않았다.

Q 디바이스 커널은 둘 다183,296 bytes이며 **정확히24 bytes**가 다르다. E30은 frozen `qkv-fp8-direct-packed-quant`의 skip 커널과 byte-identical이다. immutable 전체 test ELF에서 E28 identity Q가 offset1,276,353, E30 skip Q가1,263,041에 각각 포함된 것을 직접 확인했다. E28/E30 O 커널도 byte-identical이다. 정확한 hashes와 offsets는 [control-audit.json](control-audit.json)에 있다.

정적 스케줄은51,574 cycles,165 instructions로 같다. instruction의 source description 외 모든 필드가 같고 tensor도 생성된 name 외 shape, address, lifetime, fingerprint가 같다. 이는 같은 스케줄에서 다른 명령 상수가 생성됐음을 확인한 결과이며, backend binary24 bytes의 ISA 의미를 완전히 해독한 것은 아니다.

E28 QKV 공식9 comparisons PASS만으로 identity가 원인을 수정했다고 볼 수 없다. 이전 실패 E20/E21은 direct-packed quantizer도 달랐다. E30은 이 혼입을 제거한 직접 대조군이다. 전체 job의 FFN은 다르므로 전체 실행 상태까지 동일한 실험이라고 서술하지 않는다. Q/O 비교에는 각 커널 payload의 동일성이 확인되어 있다.

Raw frozen kernel/schedule 사본은 `../../artifacts/qkv-pair-control-audit/`에 보존했다. live source, Git, NPU는 변경하지 않았다.

후속 [E30 job70440](../../../logs/e30-pair-skip-control-job-70440.log)는 공식15 comparisons 모두 PASS했다. QKV median98,599, O52,113, FFN389,137이다. E28의 QKV median98,428과 E30 모두 통과했으므로 **명시적 hi×1은 관측된 E28 통과에 필수가 아니었다.** direct-packed 배치/스케줄의 영향과 이전 V 오류의 근본 원인은 아직 분리되지 않았다. 이번 결과를 skip-state 버그 수정의 증거로 쓰지 않는다.
