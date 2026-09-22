#!/usr/bin/env python3
"""QKV cache/RoPE boundaries on six seeded inputs; diagnostics, never timing evidence."""
from pathlib import Path
import hashlib
import json
import sys

POSITIONS = [0, 1023, 1024, 32767, 1763, 32766]


def edge_harness(content):
    text = content.decode()
    edits = [
        ("const RUNS: usize = 3;", "const RUNS: usize = 6;"),
        ("for run in 0..RUNS {", "for run in 3..3 + RUNS {"),
        ('    let seed = prng::name_hash(&format!("run{run}.global.pos"));\n    1 + (prng::word(seed, 0) % 2047) as usize',
         '    [0, 1023, 1024, 32767, 1763, 32766][run - 3]'),
        ('let mut k_cache: HbmTensor<bf16, Chip, m![Ts, Ns, Ds]> = zeros(device).await;',
         'let mut k_cache = sentinel_cache(device, 0x3e80).await;'),
        ('let mut v_cache: HbmTensor<bf16, Chip, m![Ts, Ns, Ds]> = zeros(device).await;',
         'let mut v_cache = sentinel_cache(device, 0xbf00).await;'),
        ('let k = read_bf16(device, &k_cache).await[slot * width..(slot + 1) * width].to_vec();',
         'let k = checked_cache(device, &k_cache, run, "k", slot, 0x3e80).await;'),
        ('let v = read_bf16(device, &v_cache).await[slot * width..(slot + 1) * width].to_vec();',
         'let v = checked_cache(device, &v_cache, run, "v", slot, 0xbf00).await;'),
        ('    let width = Ns::SIZE * Ds::SIZE;\n    let k = checked_cache', '    let k = checked_cache'),
    ]
    for old, new in edits:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    text += '''
// Nonzero sentinel catches writes to every non-target cache element.
async fn sentinel_cache(device: &mut Device, bits: u16) -> HbmTensor<bf16, Chip, m![Ts, Ns, Ds]> {
    let values = vec![bf16::from_bits(bits); Ts::SIZE * Ns::SIZE * Ds::SIZE];
    HostTensor::<bf16, m![Ts, Ns, Ds]>::from_vec(values).to_hbm(&mut device.pdma).await.unwrap()
}
async fn checked_cache(device: &mut Device, cache: &HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>, run: usize, key: &str, slot: usize, bits: u16) -> Vec<f32> {
    let values: HostTensor<bf16, m![Ts, Ns, Ds]> = cache.to_host(&mut device.pdma).await.unwrap();
    let values = values.into_vec();
    let width = Ns::SIZE * Ds::SIZE;
    let target = slot * width..(slot + 1) * width;
    let mut checked = 0usize;
    for (i, value) in values.iter().enumerate() {
        if target.contains(&i) { continue; }
        assert_eq!(value.to_bits(), bits, "run{run} {key} unexpected cache write at {i}");
        checked += 1;
    }
    println!("C03_CACHE run={run} cache={key} pos={} slot={slot} checked={checked} failures=0", pos(run));
    values[target].iter().map(|v| v.to_f32()).collect()
}
'''
    return text.encode()


def generate(scripts, output):
    import torch
    from safetensors.torch import save_file
    sys.path.insert(0, str(scripts.resolve()))
    import generate_references as official
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    torch.use_deterministic_algorithms(True)
    official._run_pos = lambda run: POSITIONS[run - 3]
    entries = {}
    for run in range(3, 9):
        for name, build in official.TESTS.items():
            outputs, checks = build(run)
            for label, tensor in outputs.items():
                entries[f"run{run}.{name}.{label}"] = tensor.detach().float().contiguous()
            for label, value in checks.items():
                entries[f"run{run}.{name}.check.{label}"] = torch.tensor([value], dtype=torch.int64)
        print(f"edge reference run{run} pos={POSITIONS[run - 3]}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(entries, str(output))
    def digest(path):
        data = path.read_bytes()
        return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    meta = {"positions": POSITIONS, "runs": list(range(3, 9)), "official_performance": False,
            "changes": "Override positions only; original six-run PRNG, references, dimensions and tolerances.",
            "fixture": digest(output), "torch": torch.__version__,
            "sources": {str(p.relative_to(scripts)): digest(p) for p in
                [scripts / "generate_references.py", scripts / "fixture_prng.py", scripts / "reference/gemma4.py"]}}
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    generate(Path(sys.argv[1]), Path(sys.argv[2]))
