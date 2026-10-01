# E148 finite-BF16 scale identity

The [portable proof](scale-proof.py) repeats E147's exhaustive scalar arithmetic over65,280 finite BF16 patterns. The [original script](e147-scale-proof.py) and [result](e147-scale-proof.json) are preserved byte-for-byte. Only the composition geometry changes: a monotone scalar function commutes with K256 local Max and K16 inter-slice Max, so each complete K4096 group has identical old/new scale bits. Signed zero, subnormals, floor, largest finite values and sign are included. F32 multiplication with RNE and intermediate-subnormal flush variants agree after the floor.

Run `python3 numerical/scale-proof.py --check` from the candidate directory. This proves the scalar expression on finite stored BF16, not full native arithmetic, nonfinite inputs, target lowering or NPU timing. Four residual terms and their F32 multiply/add order still require independent source/graph and exact-image output checks.

New Mul0 input work is512×256=131,072 values instead of32 valid group scalars. The one-Main expression removes two Main starts and their max intermediates. Ring16 Sub broadcast/bitmap and direct residual-VRF Main/Sub resource work remain real costs.
