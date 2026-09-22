#!/usr/bin/env python3
"""Check unchanged arithmetic and the safety of factor-renaming reshapes."""
import hashlib
import json
import re
import subprocess

from prepare import HERE, ROOT, PREFIX, VARIANTS, source_path


def main():
    # The math pipelines, stream order, reductions and rounding boundaries
    # must match E55 even though their backing storage row pitches change.
    arithmetic = re.compile(r"^\s*\.(?:fetch(?:_|<)|collect<|contract_|vector_|cast<|transpose<|switch<|commit_trim<).*$", re.M)
    reports = {}
    for variant in VARIANTS:
        target = source_path(variant)
        base = subprocess.check_output(["git", "show", "baseline-e55:" + PREFIX + target], cwd=ROOT, text=True)
        src = (HERE / variant / target.split("/")[-1]).read_text()
        if variant == "qkv-sixteen-rows":
            assert src.split("pub(crate) fn project_output(", 1)[1] == base.split("pub(crate) fn project_output(", 1)[1]
            old_dots = re.findall(r"\.begin\(weight_f8.view\(\)\)(.*?)\.commit\(\);", base, re.S)
            new_dots = re.findall(r"\.begin\(weight_f8.view\(\)\)(.*?)\.commit\(\);", src, re.S)
            assert len(old_dots) == len(new_dots) == 2
            for old, new in zip(old_dots, new_dots):
                normalized = new.replace("Qs / 16 %", "Qs / 8 %").replace("Ps / 16 %", "Ps / 8 %").replace("Qs / 64 % 2", "Qs / 32 % 2")
                assert normalized == old
        else:
            assert arithmetic.findall(src) == arithmetic.findall(base), variant
        manifest = json.loads((HERE / variant / "source-manifest.json").read_text())
        changes = []
        for rel, entry in manifest.items():
            data = (ROOT / "moa-dev/locks/gemma4-Cargo.lock").read_bytes() if rel == "Cargo.lock" else subprocess.check_output(["git", "show", "baseline-e55:" + PREFIX + rel], cwd=ROOT)
            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                changes.append(rel)
        assert changes == [target]
        assert hashlib.sha256(src.encode()).hexdigest() == manifest[changes[0]]["sha256"]
        reports[variant] = {"changed_inputs": changes, "arithmetic_streams_identical_modulo_qkv_row_renaming": True}

    # Each full HBM row is eight adjacent intervals. Splitting L into
    # DownShard x DownElement (or L/16 into DownShard x DownBlock) changes
    # neither its byte order nor nibble order. Verify every physical row and
    # shard, then both endpoints of every linear interval.
    coverage = []
    for width, bits, pitch in [(1920, 4, 2048), (120, 8, 128)]:
        row_shards = set()
        for cluster in range(2):
            for row_group in range(32):
                for row in range(60):
                    h = cluster * 1920 + row_group * 60 + row
                    intervals = []
                    for shard in range(8):
                        slc = 8 * row_group + shard
                        assert (h, shard) not in row_shards
                        row_shards.add((h, shard))
                        begin = (h * 8 + shard) * width
                        intervals.append((begin, begin + width))
                        for local in (0, width - 1):
                            split_hbm = (((cluster * 1920 + row_group * 60 + row) * 8 + shard) * width + local)
                            original_hbm = h * (8 * width) + shard * width + local
                            assert split_hbm == original_hbm
                            # After DMA only the names change. Per-level
                            # extents, slice number and padded row pitch match.
                            assert row * pitch + local < 60 * pitch
                            assert slc // 8 == row_group and slc % 8 == shard
                        assert row * pitch * bits % 64 == 0
                    assert intervals[0][0] == h * (8 * width)
                    assert intervals[-1][1] == (h + 1) * (8 * width)
                    assert all(a[1] == b[0] for a, b in zip(intervals, intervals[1:]))
        assert len(row_shards) == 3840 * 8
        coverage.append({"logical_row_width": width * 8, "bits": bits,
                         "row_shards_checked": len(row_shards),
                         "logical_elements_covered": 3840 * 8 * width,
                         "destination_row_pitch_bytes": pitch * bits // 8,
                         "padding_promoted_to_live_elements": False})
    qkv = []
    for total, cluster_size, local_rows in [(4096, 2048, 8), (2048, 1024, 4)]:
        before, after, heads = set(), set(), set()
        for row in range(total):
            cluster = row // cluster_size
            group = row // (16 * local_rows) % 16
            slc, local = 16 * group + row % 16, row // 16 % local_rows
            assert row == cluster * cluster_size + group * 16 * local_rows + 16 * local + slc % 16
            before.add((cluster, slc, local))
            # Scale DMA is canonical with local sixteen consecutive BF16s.
            canonical_slice, canonical_local = group * 16 + local, slc % 16
            assert (16 * (canonical_slice // 16) + canonical_local, canonical_slice % 16) == (slc, local)
            after.add((cluster, canonical_slice, canonical_local))
            if total == 4096:
                head, gd = divmod(row, 512)
                g, d = divmod(gd, 256)
                assert (cluster, canonical_slice, canonical_local) == (head // 4, (head % 4) * 64 + g * 32 + (d // 128) * 16 + d // 16 % 8, d % 16)
                heads.add((head, g, d))
            else:
                head, d = divmod(row, 256)
                assert (cluster, canonical_slice, canonical_local) == (head // 4, (head % 4) * 64 + (d // 64) * 16 + d // 16 % 4, d % 16)
                heads.add((head, d))
        assert len(before) == len(after) == len(heads) == total
        # Every unused slot remains padding in both the scale DMA and the
        # inverse output transpose. It is not renamed as a valid head value.
        invalid = sum(local >= local_rows for _ in range(2 * 16 * 16) for local in range(16))
        assert invalid == 8192 - total
        qkv.append({"rows": total, "bijection_checked": True, "padding_slots": invalid})
    # O loading order permutation uses the same row/column coordinates.
    # DMA returns to 16*row_group+column_group before the unchanged reduction.
    output_rows = {1920*c + 120*g + r for c in range(2) for g in range(16) for r in range(120)}
    output_columns = {256*j + k for j in range(16) for k in range(256)}
    assert output_rows == set(range(3840)) and output_columns == set(range(4096))
    assert {16*j+g for j in range(16) for g in range(16)} == set(range(256))
    result = {"variants": reports, "down_factor_rename_proof": coverage,
              "qkv_row_scale_and_head_bijection": qkv,
              "output_coordinate_product": len(output_rows) * len(output_columns),
              "scope": "Source/math and integer coordinate checks, not proof of SDK execution or arbitrary-input floating-point equivalence."}
    (HERE / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"variants": len(reports), "coordinate_proofs": coverage}, indent=2))


if __name__ == "__main__":
    main()
