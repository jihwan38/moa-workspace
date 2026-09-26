#!/usr/bin/env python3
"""Verify frozen source, metadata-only labeling, schedules, and raw Arena evidence."""
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
from image_profile import digest, inspect

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARTIFACTS = ROOT / "moa-dev/experiments/artifacts/e104-weight-path"
sys.path.insert(0, str(ROOT / "moa-dev"))
from measure import parse_log


def main():
    build = json.loads((HERE / "build-profile-v1.json").read_text())
    expected = json.loads((HERE.parent / "c07-native-scale-dpe/build-c07-e104-reviewed-v1-repro.json").read_text())
    assert build["exact_compile_exit_code"] == build["host_build_exit_code"] == 0
    for name, image in build["images"].items():
        assert image == {k:expected["images"][name][k] for k in ("bytes","sha256")}
    frozen = HERE / "evidence/profile-v1"
    assert digest((frozen / "test_kernels.rs").read_bytes()) == build["source_inputs"]["src/bin/test_kernels.rs"]
    for name, value in build["logs"].items():
        assert digest((frozen / name).read_bytes()) == value
    assert digest((frozen / "schedule.json").read_bytes()) == build["dumps"]["schedule.json"]
    base = build["source_plan"]["source_base_commit"]
    for name, value in build["source_inputs"].items():
        if name == "src/bin/test_kernels.rs":
            continue
        data = ((ROOT/"moa-dev/locks/gemma4-Cargo.lock").read_bytes() if name=="Cargo.lock" else
            subprocess.check_output(["git","show",base+":furiosa-opt-gemma4-12B/"+name],cwd=ROOT))
        assert digest(data) == value
    profile = json.loads((HERE / "profile-v1-image-profile.json").read_text())
    assert profile == json.loads((HERE / "original-image-profile.json").read_text())
    assert len(profile["spans"][0]) == 351

    artifact_checks = []
    out = ARTIFACTS / "profile-v1"
    if (out / "unlabelled-host").exists():
        host = (out / "unlabelled-host").read_bytes()
        original = next((out / "kernels").rglob("*::decoder_feedforward.bin")).read_bytes()
        labelled, metadata = inspect(original, True)
        metadata = json.loads(json.dumps(metadata))
        assert metadata == profile
        _, new = inspect(labelled)
        new = json.loads(json.dumps(new))
        allowed = {i for row in metadata["spans"] for span in row
                   for i in range(span["name_offset"],span["name_offset"]+span["name_bytes"])}
        assert all(a==b or i in allowed for i,(a,b) in enumerate(zip(original,labelled)))
        for key in ("tasks","weight","slots","inputs","outputs","stacks","profile_depth"):
            assert metadata[key] == new[key]
        assert [(s['begin_marker'],s['end_marker']) for row in metadata['spans'] for s in row] == [
            (s['begin_marker'],s['end_marker']) for row in new['spans'] for s in row]
        assert host.count(original) == 1
        assert digest(host) == build["unlabelled_host"]
        patched = (out / "package/probe").read_bytes()
        assert host.replace(original,labelled) == patched
        assert digest(patched) == build["labelled_host"]
        artifact_checks += ["Only span-name bytes changed in embedded FFN image; all task bytes/markers/bindings unchanged"]

    package = json.loads((HERE / "e104-profile-screen-v1-package.json").read_text())
    passes = 0
    jobs = []
    for path in sorted(HERE.glob("e104-profile-screen-v1-job-*-results.json")):
        result = json.loads(path.read_text())
        assert result["job"]["status"]=="SUCCEEDED" and result["job"]["exit_code"]==0
        raw = (ROOT / result["log"]).read_bytes()
        assert digest(raw) == result["log_digest"]
        log = raw.decode()
        sections = re.findall(r"^DIAG_STAGE_BEGIN (\w+)\n(.*?)^DIAG_STAGE_END \1$",log,re.M|re.S)
        assert [name for name,_ in sections] == [row[0] for row in package["stages"]]
        assert log.count("DIAG_SCREEN_COMPLETE")==1
        for name, section in sections:
            assert re.findall(r"^DIAG_STAGE_EXIT "+name+r" (\d+)$",section,re.M)==["0"]
            saved = result["stages"][name]
            if name.startswith("control"):
                assert parse_log(section) == saved["kernels"]
                passes += 15
            else:
                spans = [json.loads(line) for line in re.findall(r"^DIAG_SPAN (.+)$",section,re.M)]
                assert spans == saved["spans"]
                assert len(spans)==(768 if name=="probe_trace" else 6)
                assert "DIAG_CHECK_COMPLETE selected=1 runs=3 failures=0" in section
                assert len(re.findall(r"within tol=.*-> PASS",section))==3
                assert not re.search(r"\bFAIL\b",section)
                passes += 3
        archive_path = ARTIFACTS / f"job-{result['job']['id']}.tar"
        if archive_path.exists():
            with tarfile.open(archive_path) as tar:
                members=[m for m in tar if m.isfile()]
                assert len({m.name for m in members})==len(members)
                files={m.name:digest(tar.extractfile(m).read()) for m in members if m.name!="job.yaml"}
            assert files == package["package_files"]
            artifact_checks.append(f"Downloaded job {result['job']['id']} payload matches frozen package")
        jobs.append(result["job"]["id"])
    assert jobs==[102797,102842] and passes==78
    # Recompute derived correspondence and all 12 causal-order checks.
    before = {name:(HERE/name).read_bytes() for name in ("marker-source-map.json","profile-analysis.json")}
    subprocess.run([sys.executable,str(HERE/"analyze.py")],check=True,stdout=subprocess.DEVNULL)
    assert all((HERE/name).read_bytes()==value for name,value in before.items())
    verification={"status":"PASS","jobs":jobs,"comparison_passes":passes,
        "source_origin":base,"live_mlp_matches_source_origin":digest((ROOT/"furiosa-opt-gemma4-12B/src/device/shared/mlp.rs").read_bytes())==build["source_inputs"]["src/device/shared/mlp.rs"],
        "artifact_checks":artifact_checks,"engine_operations_mapped":120,"run_cluster_correspondences":12,
        "scope":"Diagnostic evidence integrity and output checks; no optimization speedup or adopted-runtime claim"}
    assert verification["live_mlp_matches_source_origin"]
    (HERE/"verification.json").write_text(json.dumps(verification,indent=2)+"\n")
    print(json.dumps(verification,indent=2))


if __name__=="__main__":
    main()
