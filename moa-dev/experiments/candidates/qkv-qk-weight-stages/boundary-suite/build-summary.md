# E146 boundary packages ready

Both packages are bound to `e146-qk-half-v4`; no boundary NPU job has run. Raw9 exact diagnostic and host passed all63 input SHA checks; full2 host passed38 input SHA checks and embeds the selected production QKV image exactly.

Whole QKV image:213760B, `e0245eaced58992518dfecaf6b0447c107a93f942303d15b7997716fcabe371f`. Separate raw diagnostic image:141056B, `a4943e8628f93fd38494bf80fefadfcf52b5c006f67c2f48b866b182ea15b724`. The diagnostic image is not the production whole image; its48 source files retain all46 production inputs except the documented lib module append.

[Build binding](build-summary.json), [raw build](raw-build-v1.json), [full2 build](build-e146-boundary-v1.json). Compile target m03 is idle after both builds. NPU validation remains conditional on the first positive whole screen and coordinated slot permission.
