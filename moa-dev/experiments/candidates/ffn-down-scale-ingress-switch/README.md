# E151: down scale ingress and Sub routing

Actual branch start `3d4a600e068429114b7b5f49fe9a8482492c5d31`, source `baseline-e150`/ad1ecb3, independent performance control `baseline/e150.json`. Candidate branch `opt/ffn-down-scale-ingress-switch`.

Current E150 profile141326 shows same-clock median10244cycles from last down native completion to first blockscale reduce, and each scaleVRF ends exactly1cycle before its consumer in all6observations. These overlapping intervals are not a saving promise. Read 480 contiguous E4 scale values per row under row-low2 ownership, then route F32 values inside the existing Sub-to-VRF consumer with InterTranspose4. Original quantizer/native/partial writes/reduction arithmetic and five12row consumers remain.

HBM scale payload remains3,686,400B. New Switch routing carries14,745,600B F32 plus bitmap/setup; no new intermediate scale DM scatter. Check actual routing order, HBM bounds, VRF5760B lifetime and all5 consumers before early whole control/candidate/control. Prior related E45/E149 costs and separate adoption/tuning/research decisions are in ingress-v1/recipe.json. Any reproducible positive can be adopted after bounded original standard/extra/rawdown/fullFFN checks; no five-percent adoption threshold.
