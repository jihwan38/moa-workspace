# Arena 드라이버 / 펌웨어 버전 조회

`probe_versions.sh`는 독립 Arena shell job용 읽기 전용 probe다. NPU 커널을 실행하지 않고 실제 실행 환경이 보여 주는 드라이버·펌웨어 버전을 기록한다. 공식 test runner와 benchmark 측정 경로는 변경하지 않는다. 작성일은 2026-09-21이다.

조회는 두 가지뿐이다.

1. 공식 CLI의 `furiosa-smi info` 기본 형식을 실행한다. 이 표에는 firmware와 장치 이름, 온도·전력·PCI 주소 등이 표시된다. `--full`이나 UUID/serial, process/status 조회는 사용하지 않는다. CLI 자체 package 버전을 driver 버전으로 취급하지 않는다. [공식 SMI CLI 문서](https://developer.furiosa.ai/latest/en/device_management/system_management_interface/furiosa_smi_cli.html#furiosa-smi-info), [공식 upgrade 확인 절차](https://developer.furiosa.ai/latest/en/get_started/upgrade_guide.html).
2. `/sys/class/rngd_mgmt/rngd!npu<번호>mgmt/` 바로 아래 `version`과 `fw_version`만 읽는다. Furiosa의 공개 `device-api`는 각각 `driver_version()`과 `firmware_version()`의 원천으로 이 속성을 사용한다. 경로와 의미는 [고정 commit의 `RNGDInner::new`](https://github.com/furiosa-ai/device-api/blob/bc3cabcf7a46f4882c26ce27b1ee981dcc6cc185/device-api/src/arch/rngd.rs#L25), [`driver_version` / `firmware_version`](https://github.com/furiosa-ai/device-api/blob/bc3cabcf7a46f4882c26ce27b1ee981dcc6cc185/device-api/src/arch/rngd.rs#L112), [속성 이름](https://github.com/furiosa-ai/device-api/blob/bc3cabcf7a46f4882c26ce27b1ee981dcc6cc185/device-api/src/sysfs.rs#L23)에서 확인했다. 임의의 module 이름이나 다른 sysfs 버전 경로는 추측하지 않는다.

SMI는 최대 3초, 속성 하나는 최대 1초, 전체는 8초 뒤 종료 요청을 보내고 1초 뒤 강제 종료한다. Arena job timeout은 10초로 설정한다. `timeout` 명령이 없으면 조회하지 않고 unavailable로 끝낸다. CLI 실패·권한 제한·숨겨진 sysfs·빈 값은 unavailable로 출력하며 환경 불일치로 추론하지 않는다. 마지막 `probe_complete`가 없으면 불완전한 로그다. 성공 종료는 조회 절차 완료를 뜻하며 모든 버전 확인 성공을 뜻하지 않는다.

담당자가 quota를 확보한 뒤 아래처럼 **스크립트 파일 하나만** 제출할 수 있다. Root가 실행한 첫 결과는 아래 job70431 기록이다. `furiosa-arena`가 인증된 개발 컨테이너 안에서 실행하는 명령이다.

```sh
furiosa-arena submit /tmp/arena-version-probe/probe_versions.sh \
  --name arena_version_probe \
  --entrypoint probe_versions.sh \
  --timeout 10
```

사전에 host의 `probe_versions.sh`를 위 컨테이너 경로에 복사하고 실행 권한을 보존한다. root가 job ID와 원본 로그를 일반 실험 기록처럼 저장한다. 이 job은 점수 측정이 아니며 한 서버의 관찰을 모든 Arena 서버에 일반화하지 않는다.

[9월 17일 포럼 답변](https://forums.furiosa.ai/t/furioa-opt-support-matrix-driver-version/464/4)은 opt 0.8.1 실물 카드 실행에 드라이버 2026.4.0이 필요하다고 안내했다. **firmware 버전이나 SMI CLI 버전이 2026.4.0인지로 driver 호환성을 판단하면 안 된다.** 실제 driver 문자열이 확인된 뒤 이 안내와 비교한다. 버전 문자열이 내부 build/git hash뿐이거나 조회할 수 없다면 결론은 미확인이다. 명시적으로 오래된 driver가 보여도 지금 kernel 오류의 원인까지 증명되지는 않는다.

검증: `sh -n`, NPU 없는 host의 직접 실행, 임시 가짜 SMI가 exit 7로 실패하는 경우를 확인했다. 모두 예상대로 종료했고, SMI 부재/실패와 driver 미확인을 구분했다. [로컬 smoke 로그](local-dry-run.log)는 Arena 결과가 아니다. 실제 Arena의 SMI/sysfs 가용성과 버전은 제출 후에만 확인할 수 있다.

## Arena 관찰: job70431

2026-09-21 root가 별도 probe job을 실행했고 성공을 보고했다. [원본 로그](../../../logs/arena-version-probe-70431.log)를 직접 읽어 다음을 확인했다.

| 항목 | 실제 문자열 | 조회 경로 |
| --- | --- | --- |
| Driver | `2026.4.0~rc2, 6d43c26` | `/sys/class/rngd_mgmt/rngd!npu0mgmt/version` |
| Firmware | `2026.2.0, a5db283` | SMI 기본 표와 `/sys/class/rngd_mgmt/rngd!npu0mgmt/fw_version` 모두 일치 |

SMI exit code는 0이고 driver/firmware 속성이 각각 하나씩 읽혔으며 마지막 `probe_complete`가 있다. 로그 hash와 로컬 script hash는 [evidence.json](evidence.json)에 보존했다. 업로드 archive와 script의 동일성은 이 조사에서 별도로 검사하지 않았다.

관찰된 driver는 2026.3이 아닌 **2026.4.0 RC2**다. Firmware의 2026.2.0을 driver 버전으로 읽으면 잘못이다. 포럼 답변은 2026.4.0을 명시하지만 이 RC2 build의 정확한 지원 상태나 opt 0.8.1 native FP8 실패 원인은 설명하지 않는다. 이 결과로 호환성 오류나 compiler/runtime 결함을 확정하지 않는다. 또한 job70431이 관찰한 실행 환경을 다른 모든 job/서버에 일반화하지 않는다.
