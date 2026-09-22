#!/usr/bin/env python3
"""Generate coupled down layouts from the immutable E74 implementation.

Only project_down_sharded and its internal layout aliases change. Up/gate,
GeGLU, normalization, public entry points and benchmark inputs stay byte exact.
"""
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = "9e2058ec0947498d7cb693fbe98e7b5f0b5c9ea7"
SOURCE = "furiosa-opt-gemma4-12B/src/device/shared/mlp.rs"
CONFIGS = {
    "four-ring16": (4, 32, 8, 16),
    "four-ring32": (4, 32, 8, 32),
    "two-ring16": (2, 16, 4, 16),
    "eight-ring32": (8, 60, 12, 32),
    "four-ring16-sub": (4, 32, 8, 16),
    "eight-ring32-sub": (8, 60, 12, 32),
    "four-row30-ring16": (4, 30, 6, 16),
    "four-row30-ring8": (4, 30, 6, 8),
    "four-row30-gather-direct": (4, 30, 6, 16),
    "eight-gather-direct": (8, 60, 12, 32),
}


def generate(name, columns, rows, tile, ring):
    base = subprocess.check_output(["git", "show", f"{BASE}:{SOURCE}"], cwd=ROOT, text=True)
    prefix, down = base.split("pub(crate) type DownEpilogueSlices", 1)
    width, blocks = 15360 // columns, 960 // columns
    groups, physical = 1920 // rows, 256 // columns
    assert 1920 % rows == rows % tile == ring % columns == 256 % ring == 0
    assert tile * blocks * 4 <= 8192 and width * 2 <= 16384
    # Replace each semantic expression once, so larger replacement values
    # cannot accidentally get transformed a second time.
    replacements = {
        "H / 60 % 32": f"H / {rows} % {groups}" + (f" # {physical}" if groups != physical else ""),
        "H % 60 = 12": f"H % {rows} = {tile}",
        "H % 60": f"H % {rows}",
        "L / 16 / 8 % 15": f"L / 16 / 8 % {blocks // 8}",
        "L / 128 % 15": f"L / 128 % {blocks // 8}",
        "L / 64 % 30": f"L / 64 % {blocks // 4}",
        "L / 32 % 60": f"L / 32 % {blocks // 2}",
        "L / 16 % 120": f"L / 16 % {blocks}",
        "L / 1920": f"L / {width}",
        "L % 1920": f"L % {width}",
        "1 # 8];\npub(crate) type DownRowsByColumns": f"1 # {columns}];\npub(crate) type DownRowsByColumns",
        ", 12,": f", {tile},",
        "12 * i": f"{tile} * i",
        " # 60,": f" # {rows},",
        " #{!} 60": f" #{{!}} {rows}",
        "[0, 1, 2, 3, 4]": "[" + ", ".join(map(str, range(rows // tile))) + "]",
        "eight column-slice": f"{columns} column-slice",
    }
    import re
    pattern = "|".join(re.escape(k) for k in sorted(replacements, key=len, reverse=True))
    down = re.sub(pattern, lambda match: replacements[match.group()], down)
    old = f"    let x: DmTensor<bf16, Chip, DownCluster, DownRowsByColumns, m![L % {width}]> = x.to_dm(&mut device.tdma);"
    copies = ring // columns
    leader = f"m![Dummy256 / {ring}, 1 # {copies}, L / {width}]"
    replicas = f"m![Dummy256 / {ring}, {copies}, L / {width}]"
    new = f"""    // Each ring receives one segment per column; its row replicas use Switch.
    // The anonymous replica axis denotes equal values, not distinct H rows.
    let x: DmTensor<bf16, Chip, DownCluster, {leader}, m![L % {width}]> = x.to_dm(&mut device.tdma);
    let x: DmTensor<bf16, Chip, DownCluster, {replicas}, m![L % {width}]> = device.main
        .begin(x.view())
        .fetch::<m![1], m![L % {width}]>()
        .switch::<{replicas}, m![1]>(SwitchConfig::CustomBroadcast {{ ring_size: {ring} }})
        .collect::<m![L / 16 % {blocks}], m![L % 16]>()
        .commit_trim::<m![L % 16]>()
        .commit();
    // Physical slice s holds column s % {columns}; row replicas share x.
    // Padded row groups are excluded by the typed consumer mapping below.
    let x: DmTensor<bf16, Chip, DownCluster, DownRowsByColumns, m![L % {width}]> = unsafe {{ x.reshape() }};"""
    assert down.count(old) == 1
    if name.endswith("-sub"):
        new = new.replace("= device.main", "= device.sub")
    if name.endswith("-gather-direct"):
        new = f"""    // First compact the 128 distributed GeGLU chunks into one contiguous
    // activation. The next DMA replicates contiguous column segments.
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![L]> = x.to_dm(&mut device.tdma);
    let x: DmTensor<bf16, Chip, DownCluster, DownRowsByColumns, m![L % {width}]> = x.to_dm(&mut device.tdma);"""
    down = down.replace(old, new)
    start = down.index("    // Only rename the existing high-row cluster axis.")
    if groups != physical:
        down = down[:start] + """    // Drop padding through a typed copy before renaming the high row axis.
    let down: DmTensor<bf16, Chip, DownCluster, Slice, m![H % 1920]> = down.to_dm(&mut device.tdma);
    let down: DmTensor<bf16, Chip, m![H / 1920], Slice, m![H % 1920]> = unsafe { down.reshape() };
    down.to_dm(&mut device.tdma)
}
"""
    if rows == 30:
        # Six BF16 rows do not form complete four-value commit groups. Preserve
        # F32 after the column sum and cast only after typed output gathering.
        down = down.replace("let mut down: DmTensor<bf16", "let mut down: DmTensor<f32")
        old_commit = f"""            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![H % {rows} = {tile} / 4], m![H % {rows} = {tile} % 4 # 16]>()
            .commit_trim::<m![H % {rows} = {tile} % 4]>()"""
        new_commit = f"""            .transpose::<m![H % {rows} = {tile} / 2], m![H % {rows} = {tile} % 2 # 8]>()
            .commit_trim::<m![H % {rows} = {tile} % 2]>()"""
        assert down.count(old_commit) == 1
        down = down.replace(old_commit, new_commit)
        start = down.index("    // Only rename the existing high-row cluster axis.")
        down = down[:start] + """    // Every slice is live: h = 1920 * cluster + 30 * row_group + local_row.
    let down: DmTensor<f32, Chip, m![H / 1920], DownRows, m![H % 30]> = unsafe { down.reshape() };
    let down: DmTensor<f32, Chip, Cluster, DownEpilogueSlices, m![H % 480]> = down.to_dm(&mut device.tdma);
    device.main.begin(down.view())
        .fetch::<m![1], m![H % 480]>()
        .fetch_cast::<bf16>()
        .collect::<m![H / 16 % 30], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit()
}
"""
    source = prefix + "pub(crate) type DownEpilogueSlices" + down
    # Cover every logical output, block, and representative packed/scale address.
    covered = [c * 1920 + g * rows + r for c in range(2) for g in range(groups) for r in range(rows)]
    assert covered == list(range(3840))
    assert [c * blocks + b for c in range(columns) for b in range(blocks)] == list(range(960))
    for s in range(256):
        leader_s = (s // ring) * ring + s % columns
        assert leader_s % columns == s % columns and leader_s // ring == s // ring
    for h in covered:
        c, local_h = divmod(h, 1920)
        for column in range(columns):
            for local_l in (0, 1, 15, 16, width - 2, width - 1):
                l = column * width + local_l
                assert h * 15360 + l == (c * 1920 + local_h) * 15360 + column * width + local_l
                assert h * 960 + l // 16 == (c * 1920 + local_h) * 960 + column * blocks + local_l // 16
    output = HERE / name
    output.mkdir(exist_ok=True)
    path = output / "mlp.rs"
    if path.exists() and path.read_text() != source:
        raise SystemExit(f"Refusing to replace an existing candidate: {path}")
    path.write_text(source)
    return dict(name=name, columns=columns, rows_per_slice=rows, row_tile=tile,
                valid_groups_per_cluster=groups, physical_groups_per_cluster=physical,
                blocks_per_slice=blocks, ring_size=ring, replicas_per_ring=copies,
                scale_vrf_bytes=tile * blocks * 4, activation_trf_bytes=width * 2,
                leader_activation_logical_bytes=2 * (groups if name.endswith("-gather-direct") else 256 // ring) * 15360 * 2,
                post_broadcast_activation_physical_bytes=512 * width * 2,
                source_base=BASE, unchanged_upgate_geglu_prefix=True,
                activation_broadcast_engine="TDMA after compact gather" if name.endswith("-gather-direct") else "Sub" if name.endswith("-sub") else "Main",
                logical_index_checks_passed=True,
                numerical_note="New FP32 reduction grouping requires NPU validation; static address proof is insufficient.")


if __name__ == "__main__":
    result = [generate(name, *config) for name, config in CONFIGS.items()]
    (HERE / "layout-proofs.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
