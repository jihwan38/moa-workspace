"""Generate a fully live four-column native FP8 down projection."""
from pathlib import Path
import re


def native_down(source, pair):
    # Start from our streamed pair helper in all cases, including time-up.
    here = Path(__file__).resolve().parent
    # Generator writes the helper explicitly from its pair source.
    helper = pair[pair.index("fn project_rows_with_scale_contraction("):pair.index("pub(crate) fn project_up_and_gate(")]
    helper = helper.replace("project_rows_with_scale_contraction", "project_native_down_rows")
    helper = helper.replace("UpGateCluster", "DownCluster").replace("UpGateSlices", "DownRowsByColumns")
    mapping = {"FfnRow": "H % 30", "H / 16 / 16": "L / 256 % 15", "H / 16 % 16": "L / 16 % 16",
               "H / 16 % 8": "L / 16 % 8", "H / 16 % 4": "L / 16 % 4", "H / 16": "L / 16 % 240",
               "H / 128": "L / 128 % 30", "H / 64": "L / 64 % 60", "H % 64": "L % 64",
               "H / 32": "L / 32 % 120", "H % 32": "L % 32", "H]": "L % 3840]"}
    pattern = "|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True))
    helper = re.sub(pattern, lambda m: mapping[m.group()], helper)
    helper = helper.replace("output: &mut DmTensor<bf16, Chip, DownCluster, DownRowsByColumns, m![H % 30, 1 # 4]>",
                            "output: &mut DmTensor<f32, Chip, DownCluster, DownRowsByColumns, m![H % 30, 1 # 2]>")
    helper = helper.replace("        .cast::<bf16, m![1 # 16]>()\n        .commit_trim::<m![1 # 4]>()", "        .commit_trim::<m![1 # 2]>()")
    start = source.index("pub(crate) type DownCluster")
    end = source.index("type FfnQuantSlices", start)
    down = (here / "down.rs").read_text()
    return source[:start] + down + "\n" + helper + "\n" + source[end:]
