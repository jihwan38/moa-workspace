# C10 — FP8 lookup에서 weight TRF로 직접 전달

**현재 compiler target의 StoTrf lowering에서 실패하여 종료.** 후보 compile 1회,
수정 재시도·host build·Arena·공식 제출 0회다. [판정](decision.json) ·
[정확한 로그](evidence/c10-direct-v1/compile.log) · [빌드 입력](build-c10-direct-v1-repro.json).

[direct-v1/projection.rs](direct-v1/projection.rs)는 E104 `7f891f9`에서 출발한다.
Q는 slice당 연속 8행, K/V는 4행의 FP8 weight를 DM에 읽고 다음 경로를 구현했다.

```text
Main FP8 weight fetch → BF16 lookup → collect → weight TRF
BF16 activation stream × weight TRF → F32 dot → channel scale → 첫 BF16
→ 현재 E104의 head 배치로 typed DMA → 기존 RMS/RoPE/cache 경로
```

이전 `qkv-lane-parallel`에 있던 BF16 weight DM commit과 Sub의 재읽기를 제거한다.
Q의 TRF payload는 61,440B, K/V는 30,720B로 SDK의 논리 한도 65,536B 이내다.
입력 activation의 전체 복제는 E104 그대로이며, weight 행 배치를 DPE lane에 맞춰
연속 배치로 변경했다. 이전 후보의 dot 직후 BF16 cast는 사용하지 않고 channel scale
뒤에 처음 반올림한다. O suffix와 `ops.rs`는 E104 그대로다.

Q/KV의 논리 행→head index 대응은 [logical-layout.json](logical-layout.json)에
전수 확인했다. 이 검사는 compiler lowering·NPU 값의 정확성을 증명하지 않는다.

Rust frontend를 통과한 뒤 Q의 `.to_trf()`에서 다음 오류가 발생했다.

```text
visa: while lowering StoTrf
caused by: lower_fetch_unit
caused by: renegade feature is enabled in non renegade-s config
src/device/sliding/projection.rs:23:10
```

Frontend API가 존재해도 설치된 compiler의 현재 target 설정이 해당 lookup→TRF
경로를 허용하지 않는 사례다. 이 결과를 RNGD 모든 TRF 기능의 부재로 확대하지 않는다.
Q에서 full QKV compile이 중단됐으므로 K/V 경로를 별도로 검증했다고 표시하지 않는다.
단순 shape/API 수정으로 해결할 근거가 없어, 다른 chip target을 강제하거나 이미
느렸던 BF16 DM 중간 저장을 되살리지 않고 종료한다. 성공 image·schedule·성능값은 없다.

```bash
python3 moa-dev/experiments/structural_build.py c10-direct-weight-trf direct-v1 --label NEW-ATTEMPT
```

실제 실패 소스와 로그는 `evidence/`에 동결했다. 나머지 compile 입력은 build JSON의
E104 commit·lock·파일 해시로 재구성한다. 채택 runtime은 E104를 유지한다.
