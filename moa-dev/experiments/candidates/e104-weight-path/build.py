#!/usr/bin/env python3
"""Build a diagnostic host with byte-identical E104 kernels and named profile IDs."""
import argparse
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from image_profile import digest, inspect

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARTIFACTS = ROOT / "moa-dev/experiments/artifacts/e104-weight-path"


def run(args, log=None):
    if log:
        with log.open("w") as out:
            subprocess.run(args, stdout=out, stderr=subprocess.STDOUT, check=True)
    else:
        subprocess.run(args, check=True, stdout=subprocess.DEVNULL)


def host_source(source):
    replacements = {
        "#[derive(Clone, Copy)]\nstruct Span {": "#[derive(Clone)]\nstruct Span {\n    name: String,",
        "    cluster: Option<u64>,\n}": "    cluster: Option<u64>,\n    name: Option<String>,\n}",
        "    fn record_debug(&mut self, _field: &tracing::field::Field, _value: &dyn std::fmt::Debug) {}":
        '''    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "name" { self.name = Some(value.to_owned()); }
    }
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "name" { self.name = Some(format!("{value:?}").trim_matches('"').to_owned()); }
    }''',
        "push(Span { cluster, begin, end });": "push(Span { name: extractor.name.unwrap_or_default(), cluster, begin, end });",
        "    let mut failures = Vec::new();\n    for test in TESTS {":
        '''    let mut failures = Vec::new();
    let mut selected = 0usize;
    for test in TESTS {
        if std::env::var("DIAG_KERNEL").is_ok_and(|name| name != test.name) { continue; }
        selected += 1;''',
        "                collector.window_cycles()":
        '''                for span in collector.spans.lock().unwrap().iter() {
                    println!("DIAG_SPAN {}", serde_json::json!({
                        "kernel": test.name, "run": run, "name": span.name,
                        "cluster": span.cluster, "begin": span.begin, "end": span.end
                    }));
                }
                collector.window_cycles()''',
        'println!("all {} tests passed", TESTS.len());': 'println!("DIAG_CHECK_COMPLETE selected={selected} runs={RUNS} failures=0");',
    }
    for old, new in replacements.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    return source


def build(label):
    record_path = HERE / f"build-{label}.json"
    assert not record_path.exists()
    spec = importlib.util.spec_from_file_location("source_planner", ROOT / "moa-dev/experiments/structural_build.py")
    planner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(planner)
    _, _, plan = planner.resolve_plan("e104-weight-path", "v1")
    out = ARTIFACTS / label
    out.mkdir(parents=True, exist_ok=False)
    crate = out / "crate"
    base = plan["source_base_commit"]
    archive = subprocess.check_output(["git", "archive", base, "furiosa-opt-gemma4-12B"], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar:
            if member.isfile():
                path = crate / Path(member.name).relative_to("furiosa-opt-gemma4-12B")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(tar.extractfile(member).read())
    shutil.copyfile(ROOT / "moa-dev/locks/gemma4-Cargo.lock", crate / "Cargo.lock")
    host = crate / "src/bin/test_kernels.rs"
    host.write_text(host_source(host.read_text()))
    inputs = {str(p.relative_to(crate)): digest(p.read_bytes()) for p in sorted(crate.rglob("*"))
              if p.is_file() and (p.is_relative_to(crate / "src") or p.name in
                                 {"Cargo.toml", "Cargo.lock", "rust-toolchain.toml"})}
    assert {k:v for k,v in inputs.items() if k != "src/bin/test_kernels.rs"} == {
        k:v for k,v in plan["source_origin_inputs"].items() if k != "src/bin/test_kernels.rs"}
    remote = "/tmp/moa-e104-weight/" + label
    run(["docker", "exec", "moa2026-dev", "mkdir", "-p", remote])
    run(["docker", "cp", str(crate) + "/.", "moa2026-dev:" + remote])
    copied = subprocess.check_output(["docker", "exec", "--workdir", remote, "moa2026-dev", "tar", "-cf", "-", *inputs])
    with tarfile.open(fileobj=io.BytesIO(copied)) as tar:
        assert {m.name: digest(tar.extractfile(m).read()) for m in tar if m.isfile()} == inputs
    env = ["docker", "exec", "--workdir", remote, "-e", "CARGO_HOME=/home/moa/.cargo",
           "-e", "CARGO_NET_OFFLINE=true", "-e", "CARGO_TARGET_DIR=/tmp/moa-c01-upgate/target",
           "-e", f"FURIOSA_OPT_OUT_DIR={remote}/kernels", "moa2026-dev"]
    record = {"label": label, "source_plan": plan, "source_inputs": inputs,
              "source_inputs_verified_in_container": True, "builder": digest(Path(__file__).read_bytes()),
              "profile_editor": digest((HERE / "image_profile.py").read_bytes()),
              "scope": "Diagnostic host/profile names only; no adopted kernel or official submission", "commands": []}
    expected = json.loads((HERE.parent / "c07-native-scale-dpe/build-c07-e104-reviewed-v1-repro.json").read_text())
    try:
        print("Exact E104 FFN compile and schedule/IR/vISA dump", flush=True)
        command = env + ["cargo", "furiosa-opt", "compile", "ops::decoder_feedforward", "--exact",
                         "--dump-schedule", "schedule.json", "--dump-ir", "ir.txt", "--dump-visa", "visa.txt"]
        record["commands"].append(command)
        run(command, out / "compile.log")
        record["exact_compile_exit_code"] = 0
        for name in ["schedule.json", "ir.txt", "visa.txt"]:
            run(["docker", "cp", "moa2026-dev:" + remote + "/" + name, str(out / name)])
        print("Build diagnostic host with unchanged E104 arithmetic", flush=True)
        run(env + ["cargo", "clean", "--release", "-p", "furiosa-opt-gemma4"])
        command = env + ["cargo", "furiosa-opt", "build", "--release", "--locked", "--bin", "test_kernels"]
        record["commands"].append(command)
        run(command, out / "build.log")
        record["host_build_exit_code"] = 0
        run(["docker", "cp", "moa2026-dev:" + remote + "/kernels", str(out / "kernels")])
        run(["docker", "cp", "moa2026-dev:/tmp/moa-c01-upgate/target/release/test_kernels", str(out / "unlabelled-host")])
        binary = (out / "unlabelled-host").read_bytes()
        record["unlabelled_host"] = digest(binary)
        record["images"] = {}
        for p in sorted((out / "kernels").rglob("*.bin")):
            name = p.name.split("::")[-1].removesuffix(".bin")
            raw = p.read_bytes()
            assert digest(raw)["sha256"] == expected["images"][name]["sha256"]
            assert binary.count(raw) == 1
            record["images"][name] = digest(raw)
            if name == "decoder_feedforward":
                labelled, metadata = inspect(raw, True)
                _, checked = inspect(labelled)
                for key in ("tasks", "weight", "stacks", "slots", "inputs", "outputs", "profile_depth"):
                    assert metadata[key] == checked[key]
                binary = binary.replace(raw, labelled)
                (out / "labelled-ffn.bin").write_bytes(labelled)
                (HERE / f"{label}-image-profile.json").write_text(json.dumps(metadata, indent=2)+"\n")
                record["profile_metadata"] = str((HERE / f"{label}-image-profile.json").relative_to(ROOT))
        assert len(record["images"]) == 3
        package = out / "package"
        package.mkdir()
        (package / "probe").write_bytes(binary)
        (package / "probe").chmod(0o755)
        record["labelled_host"] = digest(binary)
        print("Diagnostic host ready; original kernel hashes match; task bytes unchanged", flush=True)
    except subprocess.CalledProcessError as error:
        record.update(failed_command=error.cmd, failed_exit_code=error.returncode)
        raise
    finally:
        evidence = HERE / "evidence" / label
        evidence.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(host, evidence / "test_kernels.rs")
        record["logs"] = {}
        record["dumps"] = {}
        for name in ("compile.log", "build.log"):
            if (out / name).exists():
                shutil.copyfile(out / name, evidence / name)
                record["logs"][name] = digest((out / name).read_bytes())
        for name in ("schedule.json", "ir.txt", "visa.txt"):
            if (out / name).exists():
                record["dumps"][name] = digest((out / name).read_bytes())
        record_path.write_text(json.dumps(record, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label")
    build(parser.parse_args().label)
