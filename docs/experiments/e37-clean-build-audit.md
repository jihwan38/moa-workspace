# E37 빈 target 재빌드 감사

**E37 고정 소스를 빈 target에서 offline 빌드해 기존 공식 테스트 바이너리 전체를 바이트 단위로 재현했다.** SDK0.8.1과 `nightly-2026-05-01`, 기존 Cargo registry 경로를 사용한 결과다. 별도 cache 경로를 사용한 첫 빌드에서도 QKV/O/FFN 장치 커널은 모두 동일했으나, host 바이너리에 포함된 dependency 경로 문자열 때문에 전체 ELF가 달랐다. 원래 cache 경로와 또 다른 빈 target으로 원인을 분리했다.

| 항목 | 결과 |
| --- | --- |
| 입력 | `artifacts/e37-kernel-specific-rms`의 source·Cargo·toolchain·scripts·ref49개 파일 |
| 원본/최종 test binary | **3,669,824B**, SHA256 `5527080606282675b04ad56deae87b3efd704b8e043b260e9dfbd3127c604035` |
| 원본/최종 ELF BuildID | `cf86a182e0ff410425252990fb9492cef4663fc5` |
| 첫 clean build | 별도 Cargo home, exit0, 61.46초; 3개 device image 동일, host ELF 상이 |
| 두 번째 clean build | 원래 Cargo home, exit0, 59.58초; **전체 ELF 동일** |
| 입력/lock 변경 | 두 빌드 전후49개 파일 모두 동일 |
| NPU 실행 | 없음. 이번 작업은 빌드 재현 검사다. |

환경·명령·해시·두 시도의 기록은 [e37-clean-build-audit.json](e37-clean-build-audit.json)에 있다. E37의 기존 [반복 실측](e37-combined-repeats.json)과 [추가 입력 검사](e37-combined-extra-correctness.json)는 별도 결과다.

## 정확히 무엇을 새로 빌드했는가

고정 snapshot의 `src`, `scripts`, `ref`, `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`만 `/tmp/moa-e37-clean-build-audit-20260921`에 복사했다. 기존 target은 복사하지 않았다. 43개 source 파일과 lock/fixture를 snapshot provenance의 hash와 대조했고, 전체49개 입력의 전후 manifest를 보존했다. lock SHA256은 `aaff8dd41f9c643dd8c9faa358b9bc780e88a42c09172c1a5d7a0726d07efe09`다.

각 빌드 직전에 해당 target 경로가 **존재하지 않는 것**을 assertion으로 확인했다. 기존 `moa2026-dev`의 설치된 도구와 registry source cache는 사용했지만 이전 compiled target을 가져오지 않았다. Cargo는 dependency부터 host test binary와 세 device kernel을 새로 빌드했다. 최초 측정 보조 명령 `/usr/bin/time`이 설치되어 있지 않아 Cargo 호출 전에127로 종료한 로그도 보존했다. target 부재를 다시 확인한 뒤 Bash 내장 `time`을 사용했으며 설치 변경은 하지 않았다.

전체 ELF가 일치한 두 번째 빌드의 핵심 명령은 다음과 같다. 같은 감사 snapshot이 scratch에 복사돼 있어야 하며, 재실행 시 target은 새로운 빈 경로를 사용한다.

```sh
cd /tmp/moa-e37-clean-build-audit-20260921
export CARGO_HOME=/home/moa/.cargo
export CARGO_TARGET_DIR=/tmp/moa-e37-clean-build-audit-20260921/target-canonical-cargo-home
export FURIOSA_OPT_OUT_DIR="$CARGO_TARGET_DIR/furiosa-opt/kernel"
export CARGO_NET_OFFLINE=true
export RUSTUP_TOOLCHAIN=nightly-2026-05-01
unset RUSTC_WRAPPER RUSTC_WORKSPACE_WRAPPER
test ! -e "$CARGO_TARGET_DIR"
cargo furiosa-opt build --release --locked --offline --bin test_kernels -j 8
```

첫 빌드는 `CARGO_HOME=/tmp/moa-extra-seeds-e31-rms-extra/cargo-home`, target은 scratch의 `target`을 사용했다. 두 Cargo home 모두 추가 `config`/`config.toml`이 없고 `RUSTFLAGS`도 설정되지 않았다. 두 빌드에 새로 나타난 오류는 없으며 기존 unused helper5개 경고만 있었다.

## 첫 host ELF 차이와 device image 확인

첫 바이너리는 3,672,320B, SHA256 `ea165d75ae5b8636fd2ec540e6e3ebf4273951ed0c57eceed34fec7b6fa27885`다. 원본의 `/home/moa/.cargo` dependency 경로85개가 더 긴 임시 Cargo home으로 바뀌어 `.rodata`가 198,440→200,936B로 증가했다. 파일 크기의 차이도 같은2,496B다. `.text` 및 relocation/address 관련 section, BuildID도 달라져 단순히 BuildID만 무시하면 같은 파일이라는 주장을 하지 않는다. 문자열·ELF section 비교를 저장했고, 원래 Cargo home에서 다시 clean build한 전체 파일이 원본과 일치함을 확인했다.

세 device image는 **두 clean build 모두** E37의 고정 hash와 일치했다. 각 새 image의 실제 bytes가 원본 E37 ELF와 각 재빌드 ELF에 정확히 한 번씩 포함되는 것도 검사했다.

| device kernel | 크기 | SHA256 |
| --- | ---: | --- |
| `sliding_project_qkv` | 187,648B | `91b232f0179b37b126f2ba8755b6125e532364418516948fd98dadcc9d7e77ea` |
| `sliding_attention_output` | 88,576B | `e9d3402c62fd866f417053e618e7a7af1d5cb60485c04cc292f3a702f5996cde` |
| `decoder_feedforward` | 204,032B | `61ad0c50a03902de9208447951086df930669ded080c94e4c637bffac3c414b6` |

## 환경과 범위

Rust `1.97.0-nightly (f53b654a8 2026-04-30)`, Cargo `1.97.0-nightly (eb9b60f1f 2026-04-24)`, cargo-furiosa-opt0.8.1을 사용했다. 컨테이너 이미지 ID는 `sha256:518d7b7ddaca5a97344d8bbd60e04f5e87ca7c50d883bfb5930a64a0c6ff382f`이며 컨테이너 ID·이미지·시작 시간이 감사 전후 같았다. 실행 파일 hash와 linker/glibc 정보도 JSON/raw manifest에 있다.

이 결과는 **기록한 설치 환경과 Cargo home 경로에서 E37을 빈 target으로 재현한 증거**다. 새 OS/image 설치 재현은 이전 [E18 fresh Docker 감사](repro-audit-e18.md)와 구분한다. 임의 Cargo home 경로에서 host ELF까지 같은지는 보장하지 않는다. offline Cargo는 기존 dependency source cache와 설치된 toolchain을 사용했으며 network packet 감사를 수행한 것은 아니다. Golden fixture는 그대로 복사했고 다시 생성하거나 NPU/full-model 실행을 하지 않았다.

두 binary, kernel images, 원본/재빌드 BuildID, build logs, 49개 입력 manifest 및 ELF 차이는 Git-ignored [raw artifacts](../../moa-dev/experiments/artifacts/e37-clean-build-audit)에 보존했다. live source/target, Git index/commit은 변경하지 않았다.
