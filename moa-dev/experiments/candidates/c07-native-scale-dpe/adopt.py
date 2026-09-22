#!/usr/bin/env python3
"""Checkpoint the validated E104 source, then create its recipe after tagging that commit."""
import argparse
import json
import re
import subprocess
from build import HERE, common

ROOT = common.ROOT
PREFIX = str(HERE.relative_to(ROOT))
PERFORMANCE = "docs/experiments/e104-c07-e5-reload-materialized.json"
EXTRA = "docs/experiments/e104-c07-extra-correctness.json"
PROBE = "docs/experiments/e104-c07-probe-correctness.json"


def read(path):
    return json.loads((ROOT / path).read_text())


def write(path, value):
    (ROOT / path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def validated():
    measured = read(PREFIX + "/build-c07-up-e5-reload-materialized-v3-repro.json")
    reviewed = read(PREFIX + "/build-c07-e104-reviewed-v1-repro.json")
    extra = read(PREFIX + "/build-c07-up-e5-reload-materialized-extra-v1-repro.json")
    assert measured["package_files"] == reviewed["package_files"]
    assert measured["images"] == reviewed["images"] == extra["images"]
    performance = read(PERFORMANCE)
    control = read(performance["reference"])
    assert performance["valid"] and performance["independent_arena_jobs"] == 3
    assert control["valid"] and control["independent_arena_jobs"] == 3
    key = "decoder_feedforward"
    assert performance["aggregate"][key]["max_cycles"] < control["aggregate"][key]["min_cycles"]
    assert performance["aggregate"][key]["max_cycles"] < min(performance["control_after_job_medians"])
    probe = read(PREFIX + "/projection-probe-reload-materialized-v1-results.json")
    additional = read(PREFIX + "/c07-screen-e104-extra-results.json")
    assert probe["valid"] and probe["complete"]
    assert additional["valid"] and additional["stages"]["candidate_0"]["valid"]
    assert additional["stages"]["candidate_0"]["output_comparisons_passed"] == 30
    return measured, reviewed, performance, probe, additional


def source_checkpoint():
    measured, reviewed, performance, probe, extra = validated()
    # Preserve the raw diagnostic build path while presenting standalone evidence.
    write(PROBE, dict(probe, build=PREFIX + "/" + probe["build"]))
    write(EXTRA, dict(extra, build=PREFIX + "/build-c07-up-e5-reload-materialized-extra-v1-repro.json",
                     results=extra["stages"]["candidate_0"], cycles_are_diagnostic_only=True,
                     device_images_identical_to_basic=True))
    candidate = common.ARTIFACTS / "c07-e104-reviewed-v1/crate"
    live = ROOT / common.CRATE
    expected = reviewed["source_inputs"]
    changed = []
    for name, value in expected.items():
        if common.digest((live / name).read_bytes()) != value:
            changed.append(name)
    assert changed in (["src/device/shared/mlp.rs"], []), changed
    ops = (candidate / "src/ops.rs").read_bytes()
    assert ops == subprocess.check_output(["git", "show", common.BASE + ":" + common.CRATE + "/src/ops.rs"], cwd=ROOT)
    signatures = sum(len(re.findall(rb"#\[device\(", (candidate / name).read_bytes()))
                     for name in expected if name.startswith("src/") and name.endswith(".rs"))
    assert signatures == 15, signatures
    (live / "src/device/shared/mlp.rs").write_bytes((candidate / "src/device/shared/mlp.rs").read_bytes())
    for name, value in expected.items():
        assert common.digest((live / name).read_bytes()) == value, name
    write(PREFIX + "/compiled-source-manifest.json", expected)
    write(PREFIX + "/source-build-audit.json", dict(
        candidate="E104", source_base=common.BASE,
        measured_build=PREFIX + "/build-c07-up-e5-reload-materialized-v3-repro.json",
        source_build=PREFIX + "/build-c07-e104-reviewed-v1-repro.json",
        source_comment_review=PREFIX + "/reviewed-source-equivalence.json",
        live_compile_inputs_match_frozen_build=True, compile_inputs=len(expected),
        public_device_signatures_preserved=signatures, changed_runtime_files=["src/device/shared/mlp.rs"],
        qkv_output_images_unchanged=True, basic_and_extra_target_images_identical=True,
        reviewed_and_measured_entire_packages_identical=True, images=reviewed["images"],
        validation=[PERFORMANCE, EXTRA, PROBE], full_model_tested=False,
        scope="Native FP8 input approximation, validated on nine standard inputs and six independent projection cases. Not an all-input equivalence or full-model result."))
    print("E104 source checkpoint ready; commit this source and tag baseline-e104 before generating the recipe")


def recipe():
    validated()
    commit = subprocess.check_output(["git", "rev-parse", "baseline-e104^{commit}"], cwd=ROOT, text=True).strip()
    manifest = read(PREFIX + "/compiled-source-manifest.json")
    for name, value in manifest.items():
        path = "moa-dev/locks/gemma4-Cargo.lock" if name == "Cargo.lock" else common.CRATE + "/" + name
        assert common.digest(subprocess.check_output(["git", "show", commit + ":" + path], cwd=ROOT)) == value
    data = read("baseline/e79.json")
    data.update(id="E104", git=dict(tag="baseline-e104", commit=commit),
                compile_inputs=PREFIX + "/compiled-source-manifest.json", source_build_audit=PREFIX + "/source-build-audit.json",
                performance_report=PERFORMANCE, recent_control_report="docs/experiments/c07-reload-e79-control.json",
                correctness_evidence=[EXTRA, PROBE, "docs/experiments/e55-qkv-boundary-probe.json", "docs/experiments/output-boundary-comparison.md"],
                scope="E104: E5M2 three-term native up/gate, materialized FP8 weights, per-projection input reload, F32 block partials and contiguous 30-row slices. E79 down/epilogue and QKV/O images preserved. Native input representation is approximate; no all-input or full-model equivalence claim.")
    mlp = common.CRATE + "/src/device/shared/mlp.rs"
    replacements = {
        "F1": ("FFN contiguous 30행 배치와 입력 방송",
               "Up/gate 15360행을 2×256 slices에30행씩 연속 배치한다. 양자화한 입력은 ring32로 복제하고 retained DM에서 up/gate TRF와 input scale을 각각 읽는다. Down은 E79의8-column leader DMA·ring32 방송을 유지한다.",
               "up row L=7680*c+30*s+r; 512 live slices; down 2×32×60 rows, 8 columns",
               "F32 30행을2원소/8B로 commit한 뒤 canonical60행 배치로 gather하고 BF16으로 바꾼다. Weight/scale의 행 대응과 실제 broadcast 좌표를 함께 검증한다.",
               ["feedforward_sharded", "project_up_and_gate", "broadcast_ffn_quantized", "cast_gathered_rows", "project_down_sharded"]),
        "F2": ("Native E5M2 up/gate와 BF16 down DPE",
               "Up/gate는 FP4 weight를 E5M2 DM으로 펼쳐 native FP8 DPE에 넣고, 세 activation term을 time축에서 합친다. F32 input scale을 복원한 block partial만 저장한다. Down은 E79 FP8 DM→BF16 stream/DPE→F32 partial을 유지한다.",
               "up/gate 30×240 F32 partial/slice; FfnTerm=3, packet64B; down60×120",
               "FP4→FP8 DM을 생략한 직접 stream은 별도 projection 진단에서 실패했다. BF16 block partial은 사용하지 않는다. 다른 구조는 새 image로 검증하며 이 제약을 일반 하드웨어 법칙으로 해석하지 않는다.",
               ["project_up_and_gate", "project_rows_with_scale_contraction", "project_down_sharded"]),
        "F3": ("F32 block partial의 VE scale·합산",
               "Up/gate의 native block dot에 input scale을 복원한 F32 partial과 해당 E4M3 weight scale을 곱하고 H block을 합친다. F32 행 gather 뒤 처음 BF16으로 바꾼다. Down은 local F32 block 합산→8-column F32 합산→BF16 경계를 유지한다.",
               "block=16; scale E4M3→F32; second scale-DPE 없음",
               "project_rows_with_scale_contraction이라는 이름과 달리 두 번째 단계는 VE다. Block scale 좌표와 F32 partial 경계를 확인하고 중간 BF16 rounding을 추가하면 상쇄 검증을 새로 한다.",
               ["project_rows_with_scale_contraction", "cast_gathered_rows", "project_down_sharded"]),
        "F4": ("30/60행 dot stream과6/12행 scale tile",
               "Up/gate는30행 native dot 뒤6행 scale/reduction을5회 수행한다. Down은60행 BF16 dot 뒤12행 scale/reduction을5회 수행한다. Up의 F32 output은2원소씩 commit하고 gather 후 BF16으로 변환한다.",
               "up scale6×240×4=5760B/slice; down scale12×120×4=5760B/slice",
               "VRF 크기뿐 아니라 operand lifetime·tile offset·입력 재로딩을 확인한다. Tile이나 materialization 제거는 별도 조합으로 재검증한다.",
               ["static_passes", "project_rows_with_scale_contraction", "project_down_sharded"]),
    }
    evidence = [PREFIX + "/README.md", PERFORMANCE, EXTRA, PROBE]
    for method in data["methods"]:
        if method["id"] in replacements:
            name, behavior, parameters, invariant, symbols = replacements[method["id"]]
            method.update(name=name, behavior=behavior, parameters=parameters, invariant=invariant,
                          implementation=[dict(path=mlp, symbols=symbols)], evidence=evidence)
            if method["id"] in {"F1", "F2", "F3"}:
                method["coupled_with"] = sorted(set(method["coupled_with"] + ["F6"]))
        elif method["id"] == "F5":
            method["evidence"] = evidence + ["moa-dev/experiments/candidates/c04-down-transfer/broadcast-proof.json"]
    data["methods"].append(dict(id="F6", name="실제 BF16 입력의 E5M2 세 항 표현", kernels=["FFN"],
        behavior="실제 pre-RMS 결과의 amax에서 power-of-two scale을 계산한다. E5M2 high, 첫 residual, 두 번째 residual을480원소 shard에서 만든 뒤 ring32로 방송한다. 고정 입력값이나 fixture 분기는 없다.",
        parameters="s=2^floor(log2(max(amax,2^-112)))/16384; hi=RN_E5(x/s), lo=RN_E5(x/s-hi), third=RN_E5(x/s-(hi+lo))",
        coupled_with=["F1", "F2", "F3"],
        invariant="이는 일반 BF16의 정확한 분해가 아니다. 표준9입력의 CPU 복원과 별도6-case NPU projection을 통과했지만2^-32 혼합 CPU 반례는 남는다. 새 dtype/scale은 작은 값·상쇄·zero·block 경계를 다시 검사한다.",
        implementation=[dict(path=mlp, symbols=["prepare_ffn_fp8_split", "broadcast_ffn_quantized"])],
        evidence=evidence + [PREFIX + "/numerics.json"]))
    write("baseline/e104.json", data)
    current = read("baseline/current.json")
    current["recipe"] = "baseline/e104.json"
    if "id" in current:
        current["id"] = "E104"
    write("baseline/current.json", current)
    print("E104 recipe points to", commit)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["source", "recipe"])
    args = p.parse_args()
    source_checkpoint() if args.action == "source" else recipe()
