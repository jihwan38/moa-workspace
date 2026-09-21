# E49: up·gate fused decode — 정적 schedule 회귀로 제외

**Exact compile은 성공했지만 full host build/NPU 검사는 하지 않았다.** 정적 makespan은 E48 157,866 → **189,967(+32,101, +20.33%)**이며 E37 162,209보다도 17.11% 높다. Compiler가 다른 실행 순서를 선택해 DMA와 계산의 겹침이 줄어든 근거가 있어 compile-only 후보로 보존한다. 실제 NPU가 느리다거나 정확도에 실패했다고 판단한 결과는 아니다.

소스 SHA256은 `083e53e4dd8077a805f785fe8b7a1c8c5b0b8098a27f62bcf902c6b6059b40bd`다. Source와 patch, raw schedule을 `../../artifacts/ffn-up-gate-fused-decode-e37/`에 복사해 보존했다. 분석 근거는 `evidence.json`과 `schedule-summary.json`에 있다.

## 변경 범위와 실제 비용

E48에서 검증된 up fused chain을 gate에도 적용했다. 입력/출력 mapping은 그대로 **2clusters ×240live/256physical slices ×32rows ×3840columns**다. Local-scale 적용, 8행 BF16 DPE, GeGLU, down 두 단계 decode, post RMSNorm/residual/layer scalar는 바뀌지 않았다. Source comment의 “exact”는 FP4 값의 표현 가능성을 설명하는 의도이며 이 E49 binary의 hardware 정확성 증거가 아니다.

| 정적 항목 | E48 | E49 | 변화 |
|---|---:|---:|---:|
| Main duration 합 | 113,491 | 101,708 | −11,783 |
| DmaLoad duration 합 | 109,907 | 109,069 | −838 |
| Sub duration 합 | 17,826 | 16,274 | −1,552 |
| DmaStos duration 합 | 24,214 | 24,214 | 0 |
| DmaStore duration 합 | 648 | 648 | 0 |
| Main 미실행 구간 합 | 44,375 | 88,259 | +43,884 |
| makespan | 157,866 | 189,967 | +32,101 |

두 fused decode는 **각각 17,543 cycle**이다. 새 gate decoder는 이전의 9,863+19,463보다 11,783 cycle 짧고, up decoder도 느려지지 않았다. 개별 행렬/scale DMA duration 역시 변하지 않았다. 따라서 decode shape나 새로운 arithmetic 비용 증가로 설명되는 회귀는 아니다.

Main instruction interval이 서로 겹치지 않음을 확인하고 kernel 시간 0–makespan에서 그 구간을 뺀 결과가 표의 “Main 미실행 구간”이다. 준비/동기화와 다른 engine 실행도 포함하므로 전부 불필요한 hardware stall이라고 해석하면 안 된다. Main −11,783과 미실행 구간 +43,884의 합은 makespan +32,101과 정확히 같다.

## 겹침이 사라진 구간

E48은 gate가 먼저였지만 **E49는 up이 먼저**다. 실제 HBM source와 뒤의 consumer를 연결해 구분했다.

1. E49의 up weight DMA가 2,679–29,154에 먼저 실행된다. residual HBM DMA는 29,992–30,539까지 밀린다. E48 residual DMA는 2,679–3,226이었다.
2. 첫 Main은 E48 3,675에서 E49 **31,126**으로 늦어진다. E49는 먼저 up fused decode를 31,126–48,669에 실행한다. pre RMSNorm Main은 그 뒤에 실행되어 **53,343**에 끝난다(E48 8,349).
3. Activation 복제와 TRF 적재 준비도 69,807까지 늦어지고(E48 46,010), up 4개 dot는 80,365에 끝난다. Up output gather까지 끝나는 **82,836**에야 gate weight DMA가 시작해 **109,311**에 끝난다.
4. 기다리는 동안 일부 up 후처리가 실행되지만 **85,041–109,311에는 Main이 전혀 실행되지 않는다(24,270 cycle)**. E48의 두 번째 projection weight DMA는 44,896–71,371로, 첫 번째 projection 계산과 겹쳐졌다.
5. E49 gate fused 및 4개 dot는 137,612에 끝난다(E48 두 번째 projection 종료 103,700). 마지막 down dot 종료는 183,172로 E48 151,071보다 32,101 늦다. 이후 후단 비용은 두 후보 모두 **6,795 cycle**로 동일하다.

E49의 Core duration 합은 38,025 늘었다. 특히 `ExplicitSync(15)`가 1,206–53,529의 52,323-cycle 대기 구간을 가진다. Core duration 합을 계산량으로 간주하거나 Main 미실행 증가와 다시 더하면 중복 해석이 된다.

## 메모리 근거와 한계

E49 up BF16 T95와 gate BF16 T172는 같은 schedule address `109051904`, 같은 physical pages `[1,4,5,6,7,8,9,10]`, 같은 125,829,120B allocation을 순서대로 재사용한다. 두 FP4 input도 같은 physical pages 2/3을 재사용한다. 이는 실제 배치가 달라졌다는 근거다. 하지만 메모리 재사용 때문에 gate DMA를 반드시 82,836까지 미뤄야 한다는 인과는 이 schedule만으로 증명되지 않는다.

어떤 scheduler/allocator heuristic이 이 순서를 선택했는지, 합법적인 prefetch/lifetime 제어로 바꿀 수 있는지는 조사 범위 밖이다. 별도 lowered graph나 추가 compile 없이 보존된 schedule만 분석했다. 강제 prefetch 후보를 만들지 않았다.

E48의 up-only fused 전체 FFN은 job70683에서 all15 PASS였다. E13은 이전 구조에서 up·gate 둘 다 fused였고 실패했다. **E49는 NPU 미검증이므로 E13의 원인을 gate fusion으로 단정할 수 없다.** Tagged fullshape decode probe70658의 PASS도 이 전체 FFN 조합의 정확성을 대신하지 않는다.

이번 감사에서 추가 compile, NPU 제출, live source 수정, Git 작업은 없다. Root가 완료한 exact compile과 보존 소스/schedule만 사용했다. 확인일: 2026-09-21.
