"""Exhaustive integer-coordinate proof; no device/numerical correctness claim."""
from pathlib import Path


def query_coords(q):
    return q // 2048, (q // 64 % 32) * 8 + q % 8, q // 8 % 8


def kv_coords(p):
    return p // 1024, (p // 32 % 32) * 8 + p % 8, p // 8 % 4


for rows, coords, cluster_size, local_rows in (
    (4096, query_coords, 2048, 8),
    (2048, kv_coords, 1024, 4),
):
    seen = set()
    for row in range(rows):
        cluster, slc, local = coords(row)
        assert 0 <= cluster < 2 and 0 <= slc < 256 and 0 <= local < local_rows
        # Slice combines block index and the low 3 row bits. The local
        # coordinate carries the remaining high row bits within the block.
        rebuilt = cluster * cluster_size + (slc // 8) * (local_rows * 8) + local * 8 + slc % 8
        assert rebuilt == row
        assert (cluster, slc, local) not in seen
        seen.add((cluster, slc, local))
        # Weight and scale use this same logical row; columns are unchanged.
        for h in (0, 1, 31, 32, 3839):
            assert rebuilt * 3840 + h == row * 3840 + h
        # Switch swaps the low three slice bits with local row coordinates.
        # Q has eight valid local rows. KV has four padded to eight, so its
        # output slice has four valid entries followed by four padded ones.
        switched_slice = (slc // 8) * 8 + local
        switched_local = slc % 8
        if rows == 4096:
            n, gd = divmod(row, 512)
            g, d = divmod(gd, 256)
            assert (cluster, switched_slice, switched_local) == (n // 4, (n % 4) * 64 + g * 32 + d // 8, d % 8)
        else:
            n, d = divmod(row, 256)
            assert (cluster, switched_slice, switched_local) == (n // 4, (n % 4) * 64 + (d // 32) * 8 + d // 8 % 4, d % 8)
            assert switched_slice % 8 < 4
            assert row == cluster * 1024 + (switched_slice // 8) * 32 + switched_slice % 8 * 8 + switched_local
    assert len(seen) == rows

folder = Path(__file__).parent
old = (folder / "projection.original.rs").read_text()
new = (folder / "projection.rs").read_text()
assert old.split("pub(crate) fn project_output(", 1)[1] == new.split("pub(crate) fn project_output(", 1)[1]
print("PASS: Q4096/KV2048 bijection, logical row/column address, Switch head views and KV padding, unchanged O suffix")
