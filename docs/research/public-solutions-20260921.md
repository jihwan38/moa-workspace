# 2026-09-21 공개 구현·공식 상태 재확인

공식 main은 `8559916051162e5fd682c29ec0fe3e9cc6e420cf`(0.8.1) 그대로다. [공식 사이트](https://micro2026-moa.github.io/index.html)는 평가 기준 변경에 따른 제출 중단을 계속 공지한다. 읽기 전용 health/leaderboard API는 정상 응답했지만 인증 제출 가능 여부는 시험하지 않았다. [리더보드](https://micro2026-moa.github.io/leaderboard.html)는 참고용이며 최종 순위는 같은 조건에서 재실행한다고 명시한다. 최신 API도 58팀, 3위 Fluidstack `80,573 / 34,010 / 231,258`, score `8.398528`로 이전과 같다. [원본 응답](../experiments/leaderboard-20260921-refresh.json), [HEAD·fork·파일 hash 증거](public-solutions-20260921-evidence.json).

E15 job70181의 `115,780 / 65,476 / 397,675`를 그 API 기준에 대입한 참고 점수는 **4.993666**이다. 3위와의 비는 총점 1.682배, 커널별 QKV 1.437배/O 1.925배/FFN 1.720배다. Arena와 공식 재평가 조건이 같다는 보장은 없으며 이는 순위 달성 판정이 아니다.

## 규칙·출처 범위

[공식 README](https://github.com/micro2026-moa/furiosa-opt-gemma4-12B/blob/8559916051162e5fd682c29ec0fe3e9cc6e420cf/README.md)는 device 함수의 이름·시그니처 유지, `src/device/`와 `src/ops.rs` 함수 본문만 평가, 공식 3-seed correctness/median 평가를 명시한다. 공개 fork를 이용하는 별도의 명시적 허가는 여기서 확인하지 못했다. 공식 repo와 7개 fork의 GitHub license metadata는 null이며 확인한 tree에 LICENSE/COPYING/NOTICE가 없었다. **아래는 저자·출처를 밝힌 읽기 전용 분석이다. 외부 소스는 프로젝트에 가져오지 않았고 자체 구현·테스트도 이번 조사에서 하지 않았다.** 라이선스 부재를 참가용 공식 skeleton 사용 허가의 부정으로 해석하지 않는다.

7개 fork의 공개 branch를 조사했다. horangahor/David98Korea/michalmiotk의 main은 기존 baseline이었고, Gohing의 별도 branch는 실행 스크립트만 달랐다. KaiserRichard는 제한적인 실험 branch, hyunjun1234와 Kwon-GiHwan은 상당한 kernel 변경이 있었다. 이 계정들과 leaderboard 팀 이름의 관계는 확인하지 않았다. [공식 fork 목록](https://github.com/micro2026-moa/furiosa-opt-gemma4-12B/forks).

## 코드에서 확인한 기법

| 공개 구현 | 실제 호출 경로와 관찰 | 현재 실험에 주는 시사점 |
|---|---|---|
| [hyunjun1234: QKV](https://github.com/hyunjun1234/furiosa-opt-gemma4-12B/blob/bc9a3cc225604fff119bc6858939becb8a9ccfae/src/ops.rs#L39) | `sliding_project_qkv` → `qkv::xnorm8::normalize_everywhere_f8` → `proj8::project_query/project_key_value` → head별 norm/RoPE → HBM/cache. 두 FP8 activation 항을 TRF lane으로 넣어 같은 weight stream과 곱하고 합친다. Head는 projection한 cluster에 남긴다. | Native FP8 contraction으로 BF16 weight decode 비용을 없앨 가능성. 두 항의 표현 오차·범위와 고정 prescale을 독립 검증해야 한다. |
| [hyunjun1234: FFN](https://github.com/hyunjun1234/furiosa-opt-gemma4-12B/blob/bc9a3cc225604fff119bc6858939becb8a9ccfae/src/device/shared/ffn7.rs) | `ops::decoder_feedforward` → `ffn7::feedforward` → `normalize_quantize` → `block_sums_all` → `scales_bf16/apply_scales` → GeGLU/`quantize_gather` → `down_tile` → distributed final norm/add/gate. 16원소 block dot을 BF16 DM에 저장한 뒤, BF16 scale을 TRF에 넣는 두 번째 contraction이 scale 곱과 block 합을 수행한다. | 현재 blockscale-after-dot의 vector Mul/reduce 및 scale VRF 제한을 줄이는 별도 설계다. **Block 합의 추가 BF16 반올림**이 생기므로 현재 FP32 경계 보존 후보와 동치라고 볼 수 없다. |
| [hyunjun1234: O](https://github.com/hyunjun1234/furiosa-opt-gemma4-12B/blob/bc9a3cc225604fff119bc6858939becb8a9ccfae/src/device/sliding/output31.rs) | `sliding_attention_output` → `output31::project_normalize_add`. 작은 residual/scale/norm weight를 pool tile에 보관해 생성·읽기 의존성을 조정한다. Distributed RMSNorm의 reciprocal을 작은 tensor에서 구해 마지막 긴 vector pass의 원소별 division을 multiplication으로 바꾼다. | O의 남은 격차가 가장 크다. 먼저 자체 tail의 Div/동기화/DM 재사용 대기를 trace로 분리해 측정할 가치가 있다. Pool 자체가 빨라진다는 보장은 없다. |
| [Kwon-GiHwan: QKV](https://github.com/Kwon-GiHwan/furiosa-opt-gemma4-12B/blob/001ba1b33a4b984c0ece81d5c5ad663322848a4f/src/device/sliding/qkv_head_local.rs#L363) | [ops 호출](https://github.com/Kwon-GiHwan/furiosa-opt-gemma4-12B/blob/001ba1b33a4b984c0ece81d5c5ad663322848a4f/src/ops.rs#L39) → `normalize_native_input` → projection → head-local norm/RoPE. 실제 normalized input의 최대값에서 scale을 구하고, 큰 FP8 항과 잔차 항을 생성해 projection 후 scale을 복원한다. | 고정 gain보다 입력 크기에 적응하는 대안이다. 추가 max reduction/cast/재배치 비용이 있어 우리 shape에서 이득인지 별도 검증해야 한다. 공개 실행 로그로 현재 성능을 재현하지 않았다. |

KaiserRichard의 [FFN-002](https://github.com/KaiserRichard/furiosa-opt-gemma4-12B/blob/ef166e6fb706ed539d54a666b8848fd33ce1831f/research/experiments/FFN-002.md)는 GeGLU→down 중복 TDMA 제거 가설이고, [OUT-001](https://github.com/KaiserRichard/furiosa-opt-gemma4-12B/blob/de64039c5ca1337d57910265df4b32fac4a17868/research/experiments/OUT-001.md)은 4개 partial을 2개로 합치는 후보였다. 문서 자체가 compiler/하드웨어 미검증이라고 명시하므로 성능 근거로 쓰지 않는다. 중복 TDMA는 우리 코드에서 이미 제거했다.

### 8-way row interleave의 정확한 의미

[QKV row mapping 정의](https://github.com/hyunjun1234/furiosa-opt-gemma4-12B/blob/bc9a3cc225604fff119bc6858939becb8a9ccfae/src/device/qkv/mod.rs)와 [projection·역변환](https://github.com/hyunjun1234/furiosa-opt-gemma4-12B/blob/bc9a3cc225604fff119bc6858939becb8a9ccfae/src/device/qkv/proj8.rs#L179)을 함께 읽었다. Q head 내 행 인덱스를 `d=64g+8l+s` (`l,s=0..7`)로 분해하면 다음과 같다. 아래 식은 관찰한 배치의 대수적 설명이며 외부 구현 복사가 아니다.

| 배치 | head/query-group 내부 slice | slice 내부 행 |
|---|---:|---:|
| 연속 8행 배치 | `8g+l` | `s` |
| 8-way interleave | `8g+s` | `l` |

연속 HBM 행 8개가 같은 slice로 향하는 대신 서로 다른 8개 slice로 향한다. 각 행의 3840개 FP8 원소는 통째로 유지하므로 열 partial/reduction은 새로 생기지 않는다. Dot 이후 `InterTranspose`가 `l↔s` 자리를 되돌리고 원래 head 순서로 gather한다. 모든 slice에 실제로 준비된 전체 activation이 필요하며 padding을 replication으로 간주하면 안 된다.

저자 mapping 주석은 이 변경의 DMA throughput을 530~595 B/cycle, 이전 배치를 490~510 B/cycle로 보고한다. 우리가 재측정한 결과가 아니다. 특히 현재 FFN의 FP4 row는 1920B이고 공개 QKV의 FP8 row는 3840B이므로 정렬·packet·역변환 비용이 다르다. 자체 FFN row interleave의 안전성은 행 permutation과 역변환 coverage로, 성능은 실제 DMA timeline과 end-to-end kernel cycles로 판정해야 한다.

## 9월 21일 측정과 비교할 때

hyunjun1234의 [측정 노트](https://github.com/hyunjun1234/furiosa-opt-gemma4-12B/blob/bc9a3cc225604fff119bc6858939becb8a9ccfae/OPTIMIZATION-NOTES.md)는 SDK **0.6.0**, 로컬 RNGD, firmware 2026.3.0을 적고 있다. 9월 18일 말미에는 새 clone의 3-run median으로 QKV81995/O38870/FFN213181을 보고한다. 같은 문서가 core/NUMA affinity와 호스트 상태에 따라 결과가 달라짐을 관찰하며, 앞부분의 최소값 측정과 후반부 중앙값/짝지은 비교가 섞여 있다. 일부 과도한 순위 원인 설명은 작성자가 뒤에서 정정했다. 따라서 그 수치를 0.8.1 shared Arena의 E15와 직접 비교해 코드만의 개선량으로 주장하지 않는다.

채택할 조사 방향은 **행 배치 개선 → 작은 RMSNorm reciprocal/epilogue 축약 → FP8 두 항의 독립 수치 검증 → contraction 기반 blockscale** 순서다. 정확성 검사는 공식 tolerance와 입력 분포를 유지하고 추가 seed를 병행한다. Host pinning·최소값 선별·미초기화 tensor 읽기를 사용하는 warm-up 패턴은 이 조사에서 가져오지 않는다. Schedule상 DMA byte 감소보다 실제 명령 고정비·tail 직렬화가 더 큰지 먼저 확인한다.

공식 forum의 [VRF/stash 설명](https://forums.furiosa.ai/t/furiosa-opt-v0-5-1-8-kib-slice-vrf-operand-vector-pipeline-scheduling/451/4)은 8KiB VRF를 꽉 채운 operand와 `vector_stash()`가 겹칠 때 추가 1024B 때문에 실패할 수 있음을 설명한다(0.5.1 시점). 이는 큰 scale tile과 fused epilogue를 설계할 때 확인할 항목이며, 0.8.1의 같은 동작을 이 조사에서 실증한 것은 아니다.
