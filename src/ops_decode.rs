//! Independent-request attention adapters around batched projection/output tensors.
//! `row_index` is the unscaled row 0..B; KV offsets retain the original byte units.
use crate::Chip;
use crate::axes::{B, Df, Ds, Gf, Gs, Ns, Tf, Ts};
use crate::device::layout::{Cluster, Slice};
use crate::device::{full, sliding};
use furiosa_opt_std::prelude::*;

#[device(chip = 1)]
pub fn sliding_attention_row(
    device: &mut Device,
    q_batch: &HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
    k_batch: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v_batch: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    row_index: &HbmTensor<i32, Chip, m![1]>,
    kv_byte_offset: &HbmTensor<i32, Chip, m![1]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    mask: &HbmTensor<f32, Chip, m![Ts]>,
    out_batch: &mut HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>,
) {
    let row: DmTensor<i32, Chip, Cluster, Slice, m![1 # 8]> = row_index.to_dm(&mut device.tdma);
    let row: SpmTensor<i32, Chip, Cluster, m![1 # 4], m![1]> = row.to_spm(&mut device.tdma);
    let q: DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Gs, Ds]> = q_batch
        .gather::<m![B], m![Ns, Gs, Ds]>()
        .by_positions::<m![1]>(&row)
        .to_dm(&mut device.tdma);
    let k: DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]> = k_batch
        .gather::<m![B], m![Ns, Ds]>()
        .by_positions::<m![1]>(&row)
        .to_dm(&mut device.tdma);
    let v: DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]> = v_batch
        .gather::<m![B], m![Ns, Ds]>()
        .by_positions::<m![1]>(&row)
        .to_dm(&mut device.tdma);
    k_cache
        .view_mut()
        .scatter::<m![Ts], m![Ns, Ds]>()
        .by_byte_offsets(kv_byte_offset.view())
        .from_dm(&mut device.tdma, k);
    v_cache
        .view_mut()
        .scatter::<m![Ts], m![Ns, Ds]>()
        .by_byte_offsets(kv_byte_offset.view())
        .from_dm(&mut device.tdma, v);
    // The original attention helper accepts HBM rows. These tiny temporaries avoid
    // changing its arithmetic or copying the request's complete KV ring.
    let q_hbm: HbmTensor<bf16, Chip, m![Ns, Gs, Ds]> = q.to_hbm(&mut device.tdma);
    let mut out_hbm: HbmTensor<bf16, Chip, m![Ns, Gs, Ds]> = q.to_hbm(&mut device.tdma);
    sliding::attention::attend(device, &q_hbm, k_cache, v_cache, mask, &mut out_hbm);
    let out: DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Gs, Ds]> = out_hbm.to_dm(&mut device.tdma);
    out_batch
        .view_mut()
        .scatter::<m![B], m![Ns, Gs, Ds]>()
        .by_positions::<m![1]>(&row)
        .from_dm(&mut device.tdma, out);
}

/// Extract Q and commit K/V once before visiting this request's full-attention pages.
#[device(chip = 1)]
pub fn full_commit_row(
    device: &mut Device,
    q_batch: &HbmTensor<bf16, Chip, m![B, Gf, Df]>,
    k_batch: &HbmTensor<bf16, Chip, m![B, Df]>,
    v_batch: &HbmTensor<bf16, Chip, m![B, Df]>,
    row_index: &HbmTensor<i32, Chip, m![1]>,
    kv_byte_offset: &HbmTensor<i32, Chip, m![1]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
    q_row: &mut HbmTensor<bf16, Chip, m![Gf, Df]>,
) {
    let row: DmTensor<i32, Chip, Cluster, Slice, m![1 # 8]> = row_index.to_dm(&mut device.tdma);
    let row: SpmTensor<i32, Chip, Cluster, m![1 # 4], m![1]> = row.to_spm(&mut device.tdma);
    let q: DmTensor<bf16, Chip, Cluster, Slice, m![Gf, Df]> = q_batch
        .gather::<m![B], m![Gf, Df]>()
        .by_positions::<m![1]>(&row)
        .to_dm(&mut device.tdma);
    let k: DmTensor<bf16, Chip, Cluster, Slice, m![Df]> = k_batch
        .gather::<m![B], m![Df]>()
        .by_positions::<m![1]>(&row)
        .to_dm(&mut device.tdma);
    let v: DmTensor<bf16, Chip, Cluster, Slice, m![Df]> = v_batch
        .gather::<m![B], m![Df]>()
        .by_positions::<m![1]>(&row)
        .to_dm(&mut device.tdma);
    q.view().to_hbm_view(&mut device.tdma, q_row.view_mut());
    k_cache
        .view_mut()
        .scatter::<m![Tf], m![Df]>()
        .by_byte_offsets(kv_byte_offset.view())
        .from_dm(&mut device.tdma, k);
    v_cache
        .view_mut()
        .scatter::<m![Tf], m![Df]>()
        .by_byte_offsets(kv_byte_offset.view())
        .from_dm(&mut device.tdma, v);
}

#[device(chip = 1)]
pub fn full_divide_pack_row(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![Gf, Df]>,
    running_sum: &HbmTensor<f32, Chip, m![Gf]>,
    row_index: &HbmTensor<i32, Chip, m![1]>,
    out_batch: &mut HbmTensor<bf16, Chip, m![B, Gf, Df]>,
) {
    let row: DmTensor<i32, Chip, Cluster, Slice, m![1 # 8]> = row_index.to_dm(&mut device.tdma);
    let row: SpmTensor<i32, Chip, Cluster, m![1 # 4], m![1]> = row.to_spm(&mut device.tdma);
    let input: DmTensor<bf16, Chip, Cluster, Slice, m![Gf, Df]> = input.to_dm(&mut device.tdma);
    let out = full::attention::divide_by_softmax_sum(device, &input, running_sum);
    out_batch
        .view_mut()
        .scatter::<m![B], m![Gf, Df]>()
        .by_positions::<m![1]>(&row)
        .from_dm(&mut device.tdma, out);
}
