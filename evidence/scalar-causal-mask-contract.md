Scalar attention causal maximum repair

Workspace uploads visibility masks containing 1 or 0 (`runtime.rs`), while old
kernel probes used positive infinity or 0. The original scalar attention computed
its QK maximum over every cache slot, then clamped exponentials by this mask. A
masked future key could therefore change the maximum and the BF16-rounded
attention weights/numerator, even though its own weight was zero. Request cache
reuse could expose this dependence to identical later prompts.

Both scalar full-attention page entries and scalar sliding attention now clamp
the visibility mask at 1 and convert it to `(visibility - 1) * 1e30`. They add this
mask to QK scores before the maximum. Visible BF16 scores remain unchanged;
invisible scores become a large finite negative value. Original exponentiation,
post-exp mask, sum/EPS, value accumulation and page recurrence are retained.
The public 1/0 mask and all host/cache interfaces remain unchanged; the older
infinity/0 probe masks also remain supported.

This intentionally changes arithmetic where an invisible key previously affected
the maximum. It fixes the kernel's causal dependence; whole-model request repeat
determinism still requires the root's paired identical-prompt run. The probe
changes only masked future K and V to large finite nonzero values and requires
bit-exact nontrivial outputs, including full pages after512 and1024 positions.
