"""Verify row/column bijections and unchanged neighboring code for the sweep."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
baseline = (ROOT / "projection.8.rs").read_text()
for n in (8, 4, 2):
    groups, cols = 256 // n, 4096 // n
    rows = 1920 // groups
    row_indices = [
        1920 * cluster + rows * group + local
        for cluster in range(2)
        for group in range(groups)
        for local in range(rows)
    ]
    col_indices = [cols * group + local for group in range(n) for local in range(cols)]
    assert sorted(row_indices) == list(range(3840))
    assert sorted(col_indices) == list(range(4096))
    assert 2 * groups * n * rows * cols == 3840 * 4096
    assert rows * cols == 30720
    source = (ROOT / f"projection.{n}.rs").read_text()
    assert source.split("pub(crate) fn project_output(")[0] == baseline.split(
        "pub(crate) fn project_output("
    )[0]
    assert source.split("type HiddenRows =")[1] == baseline.split("type HiddenRows =")[1]
    if n != 8:
        assert "m![H % 1920, 1 # 4]> = result.to_dm" in source
        assert ".transpose::<m![H / 4 % 480], m![H % 4 # 16]>()" in source
    print(f"PASS {n} columns: 3840 rows, 4096 columns, 15728640 weight elements")
