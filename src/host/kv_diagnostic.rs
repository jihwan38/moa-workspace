//! Read-only full-byte cache comparison for isolated model diagnostics.
//! Invoke between completed model calls. This allocates private copy buffers,
//! reads every reserved page including untouched padding, and never changes pools.
use super::runtime::{Blocks, KvCache, Workspace};
use crate::{Chip, axes::*};
use furiosa_opt_std::prelude::*;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
async fn raw<Mp: M>(d: &mut Device, x: &HbmTensor<bf16, Chip, Mp>) -> Result<Vec<u8>> {
    Ok(x.to_host::<Mp>(&mut d.pdma).await?.into_inner().into_buf())
}
fn equal(label: &str, a: &[u8], b: &[u8]) -> Result<usize> {
    if a != b {
        let first = a.iter().zip(b).position(|(a, b)| a != b);
        return Err(format!(
            "{label}: cache byte mismatch first={first:?}, lengths={}/{}",
            a.len(),
            b.len()
        )
        .into());
    }
    Ok(a.len())
}
/// Compare equivalent logical allocations with different physical rings/pages.
/// Returns bytes compared (one side). All sliding rows and all reserved full
/// pages are checked; this is deliberately outside model timing.
pub async fn compare_canonical_kv(
    d: &mut Device,
    ws: &Workspace,
    a: &Blocks,
    b: &Blocks,
) -> Result<usize> {
    if a.full.len() != b.full.len() {
        return Err("different reserved page counts".into());
    }
    let mut sk = HostTensor::<bf16, m![Ts, Ns, Ds]>::zero()
        .to_hbm(&mut d.pdma)
        .await?;
    let mut sv = HostTensor::<bf16, m![Ts, Ns, Ds]>::zero()
        .to_hbm(&mut d.pdma)
        .await?;
    let mut fk = HostTensor::<bf16, m![Tf, Df]>::zero()
        .to_hbm(&mut d.pdma)
        .await?;
    let mut fv = HostTensor::<bf16, m![Tf, Df]>::zero()
        .to_hbm(&mut d.pdma)
        .await?;
    let mut checked = 0;
    for (layer, cache) in ws.kv_cache.iter().enumerate() {
        match cache {
            KvCache::SlidingCanonical { k, v } => {
                launch(
                    crate::ops_commit_wide_pool::copy_sliding_pool_to_cache_rc64,
                    (d, k, v, &ws.cache_ring_offsets[a.sliding], &mut sk, &mut sv),
                )
                .await?;
                let ak = raw(d, &sk).await?;
                let av = raw(d, &sv).await?;
                launch(
                    crate::ops_commit_wide_pool::copy_sliding_pool_to_cache_rc64,
                    (d, k, v, &ws.cache_ring_offsets[b.sliding], &mut sk, &mut sv),
                )
                .await?;
                checked += equal(&format!("sliding layer{layer} K"), &ak, &raw(d, &sk).await?)?;
                checked += equal(&format!("sliding layer{layer} V"), &av, &raw(d, &sv).await?)?;
            }
            KvCache::FullCanonical { k, v } => {
                for (logical, (&pa, &pb)) in a.full.iter().zip(&b.full).enumerate() {
                    launch(
                        crate::ops_request_attention::copy_full_pool_to_page,
                        (d, k, v, &ws.cache_page_offsets[pa], &mut fk, &mut fv),
                    )
                    .await?;
                    let ak = raw(d, &fk).await?;
                    let av = raw(d, &fv).await?;
                    launch(
                        crate::ops_request_attention::copy_full_pool_to_page,
                        (d, k, v, &ws.cache_page_offsets[pb], &mut fk, &mut fv),
                    )
                    .await?;
                    checked += equal(
                        &format!("full layer{layer} page{logical} K"),
                        &ak,
                        &raw(d, &fk).await?,
                    )?;
                    checked += equal(
                        &format!("full layer{layer} page{logical} V"),
                        &av,
                        &raw(d, &fv).await?,
                    )?;
                }
            }
            _ => return Err("canonical diagnostic requires canonical BF16 KV".into()),
        }
    }
    println!(
        "KV_ALL_BYTES_EXACT ringA={} ringB={} reserved_pages={} compared_bytes={checked}",
        a.sliding,
        b.sliding,
        a.full.len()
    );
    Ok(checked)
}
