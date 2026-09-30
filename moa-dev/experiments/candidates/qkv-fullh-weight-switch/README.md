# E144 full-H weight ingress → Main Switch → native QKV

동결 `fullh-v2`는 E142 대비 QKV를 세 독립 allocation에서 **7.4094%, 8.0340%, 7.9907%** 단축했다. Paired 변화율 중앙값은 **−7.990693%**이며, aggregate 중앙값의 비율 −7.981099%와 구분한다. 원본 3입력 성능 실행마다 45개 비교, 추가 6입력에서는 30개 비교가 PASS다. 전체 QKV 경계·캐시 검사와 [독립 최종 검토](adoption/independent-final-review.md)도 모두 PASS이며 동결 후보의 채택을 권고한다. 실제 live 채택과 공식 제출은 root가 별도로 수행한다.

| Arena job | E142 before | E144 QKV | E142 after | Paired 변화율 |
|---|---:|---:|---:|---:|
| 136536 | 103,820 | 95,909 | 103,348 | −7.409445% |
| 136554 | 104,003 | 95,910 | 104,574 | −8.033963% |
| 136573 | 105,330 | 95,899 | 103,125 | −7.990693% |

세 allocation의 host·fixture·runner 및 바깥 실행 payload는 `job.yaml`을 제외하고 byte-identical하다. 각 allocation에서 E142 before → candidate → E142 after를 순차 실행했다. 앞뒤 대조나 입력을 별도 allocation으로 세지 않는다. [성능 요약](adoption/performance-summary.md), [candidate 집계](adoption/e144-adoption-performance.json), [paired control](adoption/e144-e142-paired-control.json)은 실제 업로드 tar와 원시 로그를 다시 검증한다. 첫 [screen 판단](screen-decision.json)과 [원본 정규화 보고서](normalized/e144-qkv-fullh-weight-switch.json)는 초기 증거 그대로 보존했다.

## 실행 경로와 출처

실제 시작 commit은 `b840ed4228135f1f04a45819eb7e11c7362ee9ef`, 명시적인 source base 및 성능 control은 E142 `b3fe0ab4be2a5ece6382ae5745112722200f1b66`이다. E138 `native-v2`의 projection과 QKV caller를 사용한 뒤 Q/K/V weight 소유 layout과 native 전 fetch/Switch만 교체했다. 나머지 44개 compile input은 E142 그대로다. E142 O suffix와 FFN source, 두 실제 device image도 동일하다. 과거 E138 host의 FFN을 성능 control로 사용하지 않았다.

`preRMS → 기존 ring32 입력/consumer-local four-term quantization → shared activation TRF·계수 VRF → full-H E4 weight ingress → Main InterTranspose → 기존 Term4/K4 native·F32 reduce → canonical gather·scale·first BF16 → head RMS/RoPE/cache` 전체를 측정했다. [recipe](fullh-v2/recipe.json), [source 계약](source-contract-v2.json), [실제 source/image binding](source-build-binding.json)에 출처와 수치 범위를 기록했다.

Slice는 `4g+b`, `b=Row%4`를 소유한다. Q의 global row는 `2048c+32g+4r+b` (`r<8`), K/V는 `1024c+16g+4r+b` (`r<4`)이며 각 소유 row가 전체 H3840을 보관한다. Fetch offset `3840r+960q+u`를 Switch가 destination slice `4g+q`, time `4r+b`로 보낸다. 실제 DMA 256B packet과 32,768개 row-quarter 좌표, affine `u∈[0,960)`를 [독립 audit](audit.md)으로 확인했다.

Weight HBM read·DM write·DM read는 각각 **30 MiB로 유지**된다. 세 Main Switch 경로의 setup, buffering, backpressure 비용이 추가되며 full-weight transpose DM이나 weight TRF는 없다. 따라서 이번 이득을 전송 바이트 제거, 단독 bus stall 제거 또는 D-E143 span 합계로 설명하지 않는다. 별도 내부 계측 없이 확인한 결과는 전체 QKV latency 개선이다. Static 50,510 cycles는 별도 compiler 추정치다.

## 실패 보존과 지원되는 수리

[fullh-v1](fullh-v1/recipe.json)은 MIR `Switch OutTime mismatch`로 실패했고 실행 image가 없다. 최초 연속 row 소유 방식에서는 실제 expected time `(Row%R,Row/R)`가 canonical Row와 달랐다. [fullh-v2 수리](source-v1-v2.diff)는 slice에 low-two row bits, element에 high row bits를 놓아 `(Row/4,Row%4)=Row`가 되게 한다. 소유 row는 stride4로 바뀌므로 E136 연속 row DMA와 동일하다고 주장하지 않는다. 무근거 tile sweep이나 stream 이름만 바꾸는 수리를 하지 않았으며, v1 source/log/recipe는 보존했다.

## 정확도 범위와 판정

추가 입력 job 136585는 원래 PRNG·reference·tolerance를 유지한 run3…8이며, 세 device image가 표준 host와 동일하다. [extra 결과](adoption/extra-results.json)와 actual tar를 다시 검증했다. 성능 세 allocation은 같은 원본 3입력을 반복하므로 고유 standard 입력은 3+6=9개다. Candidate 비교 총 75개 중 반복된 성능 입력도 포함한다.

[raw9 경계 검사](boundary-suite/raw-validation-summary.md)는 실제 production helper를 호출하는 별도 diagnostic image에서 73,728 BF16 출력을 정확히 통과했다(±zero 동등). 이는 head RMS 이전 projection 경계 검사이며 whole image와 동일한 image라는 뜻이 아니다. 전체 preRMS/head/RoPE/cache의 두 원본 cancellation 입력도 job 136578에서 실제 production image로 16,384 outputs 및 8,380,416 untouched cache values를 통과했다. 이 full2 입력은 공식 generator support 밖의 별도 수치 stress이며 표준 PRNG 입력으로 세지 않는다. 임의 BF16 입력, PRNG 도달 가능성 또는 end-to-end 모델 정확도를 보장하지 않는다.

[최종 판단](decision.json)은 세 결정을 분리한다. 검증을 모두 통과한 동결 후보를 채택하도록 권고하며 live 변경은 root가 담당한다. 파라미터 튜닝은 중단하며 row/stage/term sweep을 이어가지 않는다. 후속 큰 구조 연구는 채택된 전체 경로의 남은 readiness를 재평가한 뒤 선택한다. 이득 5%는 채택 최소치가 아니다.

재검증 명령:

```bash
python3 moa-dev/experiments/candidates/qkv-fullh-weight-switch/audit.py e144-fullh-v2 --check
python3 moa-dev/experiments/candidates/qkv-fullh-weight-switch/verify_graph.py e144-fullh-v2 --check
python3 moa-dev/experiments/candidates/qkv-fullh-weight-switch/validation_performance.py --check
python3 moa-dev/experiments/candidates/qkv-fullh-weight-switch/validation_extra.py --check
python3 moa-dev/experiments/candidates/qkv-fullh-weight-switch/provenance.py --check
python3 moa-dev/experiments/candidates/qkv-fullh-weight-switch/adoption/independent-final-review.py --check
```
