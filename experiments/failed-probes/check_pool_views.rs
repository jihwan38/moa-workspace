//! Compile-only check of zero-copy host pool windows at existing kernel boundaries.
use furiosa_opt_gemma4::{Chip, axes::*, ops, ops_request_attention::Rc};
use furiosa_opt_std::prelude::*;
#[allow(dead_code)]
async fn existing_kernel_from_pool_window(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
    pool_k: &HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
    pool_v: &HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
    mask: &HbmTensor<f32, Chip, m![Ts]>,
    out: &mut HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
    slot: usize,
) -> Result<(), Error> {
    let k: HbmTensorView<'_, bf16, Chip, m![Ts, Ns, Ds]> = unsafe {
        pool_k.view().tile::<m![Rc], 1, m![Rc = 1, Ts, Ns, Ds]>(slot).reshape()
    };
    let v: HbmTensorView<'_, bf16, Chip, m![Ts, Ns, Ds]> = unsafe {
        pool_v.view().tile::<m![Rc], 1, m![Rc = 1, Ts, Ns, Ds]>(slot).reshape()
    };
    launch(ops::sliding_attention, (device, q, &k, &v, mask, out)).await
}
fn main() {}
