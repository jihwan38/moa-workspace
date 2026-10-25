//! Request-owned contiguous KV pools, with explicit bridges for legacy kernels.
//! A ring's owner changes only after all of its layer caches migrate successfully.
use super::{
    load::{Layer, Weights},
    model::Sequence,
    runtime::{Blocks, KvCache, Workspace},
};
use crate::ops_pool_views::Rc64;
use crate::{Chip, axes::*, ops_request_attention as op};
use furiosa_opt_std::prelude::*;
use op::{Fc, Rc};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub(crate) fn independent_sequences(sequences: &[Sequence<'_>]) -> bool {
    let mut rings = std::collections::BTreeSet::new();
    let mut pages = std::collections::BTreeSet::new();
    sequences.iter().all(|sequence| {
        rings.insert(sequence.blocks.sliding)
            && sequence.blocks.full.iter().all(|&page| pages.insert(page))
    })
}

pub(crate) enum PoolCache {
    Sliding {
        k: HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
        v: HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
    },
    Full {
        k: HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
        v: HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    },
}
pub(crate) struct RequestPool {
    pub layers: Vec<PoolCache>,
    pooled: Vec<bool>,
    ring_offsets: Vec<HbmTensor<i32, Chip, m![1]>>,
    page_offsets: Vec<HbmTensor<i32, Chip, m![1]>>,
}
impl RequestPool {
    pub async fn new(
        device: &mut Device,
        weights: &Weights,
        capacity: usize,
        requests: usize,
    ) -> Result<Option<Self>> {
        let pages = requests * Workspace::blocks_for(capacity);
        if requests < 2 || requests > Rc::SIZE || pages > Fc::SIZE {
            return Ok(None);
        }
        let mut layers = Vec::new();
        for layer in &weights.layers {
            layers.push(match layer {
                Layer::Sliding(_) => PoolCache::Sliding {
                    k: HostTensor::<bf16, m![Rc, Ts, Ns, Ds]>::zero()
                        .to_hbm(&mut device.pdma)
                        .await?,
                    v: HostTensor::<bf16, m![Rc, Ts, Ns, Ds]>::zero()
                        .to_hbm(&mut device.pdma)
                        .await?,
                },
                Layer::Full(_) => PoolCache::Full {
                    k: HostTensor::<bf16, m![Fc, Tf, Df]>::zero()
                        .to_hbm(&mut device.pdma)
                        .await?,
                    v: HostTensor::<bf16, m![Fc, Tf, Df]>::zero()
                        .to_hbm(&mut device.pdma)
                        .await?,
                },
            });
        }
        let mut ring_offsets = Vec::new();
        let mut page_offsets = Vec::new();
        for ring in 0..requests {
            ring_offsets.push(
                HostTensor::<i32, m![1]>::from_vec([
                    (ring * Ts::SIZE * Ns::SIZE * Ds::SIZE * 2) as i32
                ])
                .to_hbm(&mut device.pdma)
                .await?,
            );
        }
        for page in 0..pages {
            page_offsets.push(
                HostTensor::<i32, m![1]>::from_vec([(page * Tf::SIZE * Df::SIZE * 2) as i32])
                    .to_hbm(&mut device.pdma)
                    .await?,
            );
        }
        Ok(Some(Self {
            layers,
            pooled: vec![false; requests],
            ring_offsets,
            page_offsets,
        }))
    }
    pub async fn ensure_pool(
        &mut self,
        device: &mut Device,
        legacy: &mut [KvCache],
        blocks: &Blocks,
        position: usize,
    ) -> Result<()> {
        let ring = blocks.sliding;
        if position == 0 {
            self.pooled[ring] = false;
        }
        if self.pooled[ring] {
            return Ok(());
        }
        for (pool, legacy) in self.layers.iter_mut().zip(legacy) {
            match (pool, legacy) {
                (
                    PoolCache::Sliding {
                        k: pool_k,
                        v: pool_v,
                    },
                    KvCache::Sliding { k, v },
                ) => {
                    launch(
                        op::copy_sliding_cache_to_pool,
                        (
                            device,
                            &k[ring],
                            &v[ring],
                            &self.ring_offsets[ring],
                            pool_k,
                            pool_v,
                        ),
                    )
                    .await?
                }
                (
                    PoolCache::Full {
                        k: pool_k,
                        v: pool_v,
                    },
                    KvCache::Full { k, v },
                ) => {
                    for &page in &blocks.full {
                        launch(
                            op::copy_full_page_to_pool,
                            (
                                device,
                                &k[page],
                                &v[page],
                                &self.page_offsets[page],
                                &mut *pool_k,
                                &mut *pool_v,
                            ),
                        )
                        .await?;
                    }
                }
                _ => return Err("pool/legacy layer kind mismatch".into()),
            }
        }
        self.pooled[ring] = true;
        Ok(())
    }
    pub async fn ensure_legacy(
        &mut self,
        device: &mut Device,
        legacy: &mut [KvCache],
        blocks: &Blocks,
        position: usize,
    ) -> Result<()> {
        let ring = blocks.sliding;
        if position == 0 {
            self.pooled[ring] = false;
            return Ok(());
        }
        if !self.pooled[ring] {
            return Ok(());
        }
        for (pool, legacy) in self.layers.iter().zip(legacy) {
            match (pool, legacy) {
                (
                    PoolCache::Sliding {
                        k: pool_k,
                        v: pool_v,
                    },
                    KvCache::Sliding { k, v },
                ) => {
                    launch(
                        op::copy_sliding_pool_to_cache,
                        (
                            device,
                            pool_k,
                            pool_v,
                            &self.ring_offsets[ring],
                            &mut k[ring],
                            &mut v[ring],
                        ),
                    )
                    .await?
                }
                (
                    PoolCache::Full {
                        k: pool_k,
                        v: pool_v,
                    },
                    KvCache::Full { k, v },
                ) => {
                    for &page in &blocks.full {
                        launch(
                            op::copy_full_pool_to_page,
                            (
                                device,
                                pool_k,
                                pool_v,
                                &self.page_offsets[page],
                                &mut k[page],
                                &mut v[page],
                            ),
                        )
                        .await?;
                    }
                }
                _ => return Err("pool/legacy layer kind mismatch".into()),
            }
        }
        self.pooled[ring] = false;
        Ok(())
    }
}

pub(crate) struct FullPage {
    pub offsets: HbmTensor<i32, Chip, m![B]>,
    pub mask: HbmTensor<f32, Chip, m![B, Tf]>,
    pub active: HbmTensor<f32, Chip, m![B,1 # 8]>,
}
pub(crate) struct PooledStep {
    pub ring_offsets: HbmTensor<i32, Chip, m![B]>,
    pub sliding_mask: HbmTensor<f32, Chip, m![B, Ts]>,
    pub pages: Vec<FullPage>,
    pub commits: PoolOffsets,
}
impl PooledStep {
    pub async fn new(device: &mut Device, sequences: &[Sequence<'_>], sliding_element_bytes: usize) -> Result<Self> {
        let count = sequences.len();
        if count == 0 || count > B::SIZE {
            return Err("invalid pooled batch width".into());
        }
        let (ring_offsets, sliding_mask) = sliding_vectors_rows_bytes(B::SIZE, sequences, sliding_element_bytes);
        let ring_offsets = HostTensor::<i32, m![B]>::from_vec(ring_offsets)
            .to_hbm(&mut device.pdma)
            .await?;
        let sliding_mask = HostTensor::<f32, m![B, Ts]>::from_vec(sliding_mask)
            .to_hbm(&mut device.pdma)
            .await?;
        let mut pages = Vec::new();
        let max_page = sequences
            .iter()
            .map(|s| s.start_position / Tf::SIZE)
            .max()
            .unwrap();
        for page in 0..=max_page {
            let (offsets, mask, active) = full_page_vectors(sequences, page);
            pages.push(FullPage {
                offsets: HostTensor::<i32, m![B]>::from_vec(offsets)
                    .to_hbm(&mut device.pdma)
                    .await?,
                mask: HostTensor::<f32, m![B, Tf]>::from_vec(mask)
                    .to_hbm(&mut device.pdma)
                    .await?,
                active: HostTensor::<f32, m![B,1 # 8]>::from_vec(active)
                    .to_hbm(&mut device.pdma)
                    .await?,
            });
        }
        Ok(Self {
            ring_offsets,
            sliding_mask,
            pages,
            commits: PoolOffsets::new_bytes(device, sequences, sliding_element_bytes).await?,
        })
    }
}

pub(crate) enum PoolOffsets {
    V1 {
        sliding: HbmTensor<i32, Chip, m![B = 1]>,
        full: HbmTensor<i32, Chip, m![B = 1]>,
    },
    V2 {
        sliding: HbmTensor<i32, Chip, m![B = 2]>,
        full: HbmTensor<i32, Chip, m![B = 2]>,
    },
    V3 {
        sliding: HbmTensor<i32, Chip, m![B = 3]>,
        full: HbmTensor<i32, Chip, m![B = 3]>,
    },
    V4 {
        sliding: HbmTensor<i32, Chip, m![B = 4]>,
        full: HbmTensor<i32, Chip, m![B = 4]>,
    },
    V5 {
        sliding: HbmTensor<i32, Chip, m![B = 5]>,
        full: HbmTensor<i32, Chip, m![B = 5]>,
    },
    V6 {
        sliding: HbmTensor<i32, Chip, m![B = 6]>,
        full: HbmTensor<i32, Chip, m![B = 6]>,
    },
    V7 {
        sliding: HbmTensor<i32, Chip, m![B = 7]>,
        full: HbmTensor<i32, Chip, m![B = 7]>,
    },
    V8 {
        sliding: HbmTensor<i32, Chip, m![B = 8]>,
        full: HbmTensor<i32, Chip, m![B = 8]>,
    },
    V9 {
        sliding: HbmTensor<i32, Chip, m![B = 9]>,
        full: HbmTensor<i32, Chip, m![B = 9]>,
    },
    V10 {
        sliding: HbmTensor<i32, Chip, m![B = 10]>,
        full: HbmTensor<i32, Chip, m![B = 10]>,
    },
    V11 {
        sliding: HbmTensor<i32, Chip, m![B = 11]>,
        full: HbmTensor<i32, Chip, m![B = 11]>,
    },
    V12 {
        sliding: HbmTensor<i32, Chip, m![B = 12]>,
        full: HbmTensor<i32, Chip, m![B = 12]>,
    },
    V13 {
        sliding: HbmTensor<i32, Chip, m![B = 13]>,
        full: HbmTensor<i32, Chip, m![B = 13]>,
    },
    V14 {
        sliding: HbmTensor<i32, Chip, m![B = 14]>,
        full: HbmTensor<i32, Chip, m![B = 14]>,
    },
    V15 {
        sliding: HbmTensor<i32, Chip, m![B = 15]>,
        full: HbmTensor<i32, Chip, m![B = 15]>,
    },
    V16 {
        sliding: HbmTensor<i32, Chip, m![B = 16]>,
        full: HbmTensor<i32, Chip, m![B = 16]>,
    },
}
impl PoolOffsets {
    pub(crate) async fn new(device: &mut Device, sequences: &[Sequence<'_>]) -> Result<Self> {
        Self::new_bytes(device, sequences, 2).await
    }
    pub(crate) async fn new_bytes(device: &mut Device, sequences: &[Sequence<'_>], sliding_element_bytes: usize) -> Result<Self> {
        let sliding: Vec<_> = sequences
            .iter()
            .map(|s| {
                ((s.blocks.sliding * Ts::SIZE + s.start_position % Ts::SIZE)
                    * Ns::SIZE
                    * Ds::SIZE
                    * sliding_element_bytes) as i32
            })
            .collect();
        let full: Vec<_> = sequences
            .iter()
            .map(|s| {
                ((s.blocks.full[s.start_position / Tf::SIZE] * Tf::SIZE
                    + s.start_position % Tf::SIZE)
                    * Df::SIZE
                    * 2) as i32
            })
            .collect();
        Ok(match sequences.len() {
            1 => Self::V1 {
                sliding: HostTensor::<i32, m![B = 1]>::from_vec(sliding)
                    .to_hbm(&mut device.pdma)
                    .await?,
                full: HostTensor::<i32, m![B = 1]>::from_vec(full)
                    .to_hbm(&mut device.pdma)
                    .await?,
            },
            2 => Self::V2 {
                sliding: HostTensor::<i32, m![B = 2]>::from_vec(sliding)
                    .to_hbm(&mut device.pdma)
                    .await?,
                full: HostTensor::<i32, m![B = 2]>::from_vec(full)
                    .to_hbm(&mut device.pdma)
                    .await?,
            },
            3 => Self::V3 {
                sliding: HostTensor::<i32, m![B = 3]>::from_vec(sliding)
                    .to_hbm(&mut device.pdma)
                    .await?,
                full: HostTensor::<i32, m![B = 3]>::from_vec(full)
                    .to_hbm(&mut device.pdma)
                    .await?,
            },
            4 => Self::V4 {
                sliding: HostTensor::<i32, m![B = 4]>::from_vec(sliding)
                    .to_hbm(&mut device.pdma)
                    .await?,
                full: HostTensor::<i32, m![B = 4]>::from_vec(full)
                    .to_hbm(&mut device.pdma)
                    .await?,
            },
            5 => Self::V5 {
                sliding: HostTensor::<i32, m![B = 5]>::from_vec(sliding)
                    .to_hbm(&mut device.pdma)
                    .await?,
                full: HostTensor::<i32, m![B = 5]>::from_vec(full)
                    .to_hbm(&mut device.pdma)
                    .await?,
            },
            6 => Self::V6 {
                sliding: HostTensor::<i32, m![B = 6]>::from_vec(sliding)
                    .to_hbm(&mut device.pdma)
                    .await?,
                full: HostTensor::<i32, m![B = 6]>::from_vec(full)
                    .to_hbm(&mut device.pdma)
                    .await?,
            },
            7 => Self::V7 {
                sliding: HostTensor::<i32, m![B = 7]>::from_vec(sliding)
                    .to_hbm(&mut device.pdma)
                    .await?,
                full: HostTensor::<i32, m![B = 7]>::from_vec(full)
                    .to_hbm(&mut device.pdma)
                    .await?,
            },
            8 => Self::V8 {
                sliding: HostTensor::<i32, m![B = 8]>::from_vec(sliding)
                    .to_hbm(&mut device.pdma)
                    .await?,
                full: HostTensor::<i32, m![B = 8]>::from_vec(full)
                    .to_hbm(&mut device.pdma)
                    .await?,
            },
            9 => Self::V9 {
                sliding: HostTensor::<i32, m![B = 9]>::from_vec(sliding)
                    .to_hbm(&mut device.pdma)
                    .await?,
                full: HostTensor::<i32, m![B = 9]>::from_vec(full)
                    .to_hbm(&mut device.pdma)
                    .await?,
            },
            10 => Self::V10 {
                sliding: HostTensor::<i32, m![B = 10]>::from_vec(sliding)
                    .to_hbm(&mut device.pdma)
                    .await?,
                full: HostTensor::<i32, m![B = 10]>::from_vec(full)
                    .to_hbm(&mut device.pdma)
                    .await?,
            },
            11 => Self::V11 {
                sliding: HostTensor::<i32, m![B = 11]>::from_vec(sliding)
                    .to_hbm(&mut device.pdma)
                    .await?,
                full: HostTensor::<i32, m![B = 11]>::from_vec(full)
                    .to_hbm(&mut device.pdma)
                    .await?,
            },
            12 => Self::V12 {
                sliding: HostTensor::<i32, m![B = 12]>::from_vec(sliding)
                    .to_hbm(&mut device.pdma)
                    .await?,
                full: HostTensor::<i32, m![B = 12]>::from_vec(full)
                    .to_hbm(&mut device.pdma)
                    .await?,
            },
            13 => Self::V13 {
                sliding: HostTensor::<i32, m![B = 13]>::from_vec(sliding)
                    .to_hbm(&mut device.pdma)
                    .await?,
                full: HostTensor::<i32, m![B = 13]>::from_vec(full)
                    .to_hbm(&mut device.pdma)
                    .await?,
            },
            14 => Self::V14 {
                sliding: HostTensor::<i32, m![B = 14]>::from_vec(sliding)
                    .to_hbm(&mut device.pdma)
                    .await?,
                full: HostTensor::<i32, m![B = 14]>::from_vec(full)
                    .to_hbm(&mut device.pdma)
                    .await?,
            },
            15 => Self::V15 {
                sliding: HostTensor::<i32, m![B = 15]>::from_vec(sliding)
                    .to_hbm(&mut device.pdma)
                    .await?,
                full: HostTensor::<i32, m![B = 15]>::from_vec(full)
                    .to_hbm(&mut device.pdma)
                    .await?,
            },
            16 => Self::V16 {
                sliding: HostTensor::<i32, m![B = 16]>::from_vec(sliding)
                    .to_hbm(&mut device.pdma)
                    .await?,
                full: HostTensor::<i32, m![B = 16]>::from_vec(full)
                    .to_hbm(&mut device.pdma)
                    .await?,
            },
            _ => return Err("invalid pool commit width".into()),
        })
    }
    pub async fn sliding(
        &self,
        device: &mut Device,
        k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
        v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
        pool_k: &mut HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
        pool_v: &mut HbmTensor<bf16, Chip, m![Rc, Ts, Ns, Ds]>,
    ) -> Result<()> {
        match self {
            Self::V1 { sliding, .. } => {
                launch(
                    op::commit_sliding_pool_1,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V2 { sliding, .. } => {
                launch(
                    op::commit_sliding_pool_2,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V3 { sliding, .. } => {
                launch(
                    op::commit_sliding_pool_3,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V4 { sliding, .. } => {
                launch(
                    op::commit_sliding_pool_4,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V5 { sliding, .. } => {
                launch(
                    op::commit_sliding_pool_5,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V6 { sliding, .. } => {
                launch(
                    op::commit_sliding_pool_6,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V7 { sliding, .. } => {
                launch(
                    op::commit_sliding_pool_7,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V8 { sliding, .. } => {
                launch(
                    op::commit_sliding_pool_8,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V9 { sliding, .. } => {
                launch(
                    op::commit_sliding_pool_9,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V10 { sliding, .. } => {
                launch(
                    op::commit_sliding_pool_10,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V11 { sliding, .. } => {
                launch(
                    op::commit_sliding_pool_11,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V12 { sliding, .. } => {
                launch(
                    op::commit_sliding_pool_12,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V13 { sliding, .. } => {
                launch(
                    op::commit_sliding_pool_13,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V14 { sliding, .. } => {
                launch(
                    op::commit_sliding_pool_14,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V15 { sliding, .. } => {
                launch(
                    op::commit_sliding_pool_15,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V16 { sliding, .. } => {
                launch(
                    op::commit_sliding_pool_16,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
        }
        Ok(())
    }
    pub async fn sliding_canonical(
        &self,
        device: &mut Device,
        k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
        v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
        pool_k: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
        pool_v: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    ) -> Result<()> {
        match self {
            Self::V1 { sliding, .. } => {
                launch(
                    crate::ops_commit_r64::commit_sliding_pool_1,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V2 { sliding, .. } => {
                launch(
                    crate::ops_commit_r64::commit_sliding_pool_2,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V3 { sliding, .. } => {
                launch(
                    crate::ops_commit_r64::commit_sliding_pool_3,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V4 { sliding, .. } => {
                launch(
                    crate::ops_commit_r64::commit_sliding_pool_4,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V5 { sliding, .. } => {
                launch(
                    crate::ops_commit_r64::commit_sliding_pool_5,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V6 { sliding, .. } => {
                launch(
                    crate::ops_commit_r64::commit_sliding_pool_6,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V7 { sliding, .. } => {
                launch(
                    crate::ops_commit_r64::commit_sliding_pool_7,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V8 { sliding, .. } => {
                launch(
                    crate::ops_commit_r64::commit_sliding_pool_8,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V9 { sliding, .. } => {
                launch(
                    crate::ops_commit_r64::commit_sliding_pool_9,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V10 { sliding, .. } => {
                launch(
                    crate::ops_commit_r64::commit_sliding_pool_10,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V11 { sliding, .. } => {
                launch(
                    crate::ops_commit_r64::commit_sliding_pool_11,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V12 { sliding, .. } => {
                launch(
                    crate::ops_commit_r64::commit_sliding_pool_12,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V13 { sliding, .. } => {
                launch(
                    crate::ops_commit_r64::commit_sliding_pool_13,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V14 { sliding, .. } => {
                launch(
                    crate::ops_commit_r64::commit_sliding_pool_14,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V15 { sliding, .. } => {
                launch(
                    crate::ops_commit_r64::commit_sliding_pool_15,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V16 { sliding, .. } => {
                launch(
                    crate::ops_commit_r64::commit_sliding_pool_16,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
        }
        Ok(())
    }
    pub async fn sliding_fixed_e4(
        &self,
        device: &mut Device,
        k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
        v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
        pool_k: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
        pool_v: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    ) -> Result<()> {
        match self {
            Self::V1 { sliding, .. } => {
                launch(
                    crate::ops_fixed_e4_commit::commit_sliding_pool_1,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V2 { sliding, .. } => {
                launch(
                    crate::ops_fixed_e4_commit::commit_sliding_pool_2,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V3 { sliding, .. } => {
                launch(
                    crate::ops_fixed_e4_commit::commit_sliding_pool_3,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V4 { sliding, .. } => {
                launch(
                    crate::ops_fixed_e4_commit::commit_sliding_pool_4,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V5 { sliding, .. } => {
                launch(
                    crate::ops_fixed_e4_commit::commit_sliding_pool_5,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V6 { sliding, .. } => {
                launch(
                    crate::ops_fixed_e4_commit::commit_sliding_pool_6,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V7 { sliding, .. } => {
                launch(
                    crate::ops_fixed_e4_commit::commit_sliding_pool_7,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V8 { sliding, .. } => {
                launch(
                    crate::ops_fixed_e4_commit::commit_sliding_pool_8,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V9 { sliding, .. } => {
                launch(
                    crate::ops_fixed_e4_commit::commit_sliding_pool_9,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V10 { sliding, .. } => {
                launch(
                    crate::ops_fixed_e4_commit::commit_sliding_pool_10,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V11 { sliding, .. } => {
                launch(
                    crate::ops_fixed_e4_commit::commit_sliding_pool_11,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V12 { sliding, .. } => {
                launch(
                    crate::ops_fixed_e4_commit::commit_sliding_pool_12,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V13 { sliding, .. } => {
                launch(
                    crate::ops_fixed_e4_commit::commit_sliding_pool_13,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V14 { sliding, .. } => {
                launch(
                    crate::ops_fixed_e4_commit::commit_sliding_pool_14,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V15 { sliding, .. } => {
                launch(
                    crate::ops_fixed_e4_commit::commit_sliding_pool_15,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
            Self::V16 { sliding, .. } => {
                launch(
                    crate::ops_fixed_e4_commit::commit_sliding_pool_16,
                    (device, k, v, sliding, pool_k, pool_v),
                )
                .await?
            }
        }
        Ok(())
    }
    pub async fn full(
        &self,
        device: &mut Device,
        k: &HbmTensor<bf16, Chip, m![B, Df]>,
        v: &HbmTensor<bf16, Chip, m![B, Df]>,
        pool_k: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
        pool_v: &mut HbmTensor<bf16, Chip, m![Fc, Tf, Df]>,
    ) -> Result<()> {
        match self {
            Self::V1 { full, .. } => {
                launch(op::commit_full_pool_1, (device, k, v, full, pool_k, pool_v)).await?
            }
            Self::V2 { full, .. } => {
                launch(op::commit_full_pool_2, (device, k, v, full, pool_k, pool_v)).await?
            }
            Self::V3 { full, .. } => {
                launch(op::commit_full_pool_3, (device, k, v, full, pool_k, pool_v)).await?
            }
            Self::V4 { full, .. } => {
                launch(op::commit_full_pool_4, (device, k, v, full, pool_k, pool_v)).await?
            }
            Self::V5 { full, .. } => {
                launch(op::commit_full_pool_5, (device, k, v, full, pool_k, pool_v)).await?
            }
            Self::V6 { full, .. } => {
                launch(op::commit_full_pool_6, (device, k, v, full, pool_k, pool_v)).await?
            }
            Self::V7 { full, .. } => {
                launch(op::commit_full_pool_7, (device, k, v, full, pool_k, pool_v)).await?
            }
            Self::V8 { full, .. } => {
                launch(op::commit_full_pool_8, (device, k, v, full, pool_k, pool_v)).await?
            }
            Self::V9 { full, .. } => {
                launch(op::commit_full_pool_9, (device, k, v, full, pool_k, pool_v)).await?
            }
            Self::V10 { full, .. } => {
                launch(
                    op::commit_full_pool_10,
                    (device, k, v, full, pool_k, pool_v),
                )
                .await?
            }
            Self::V11 { full, .. } => {
                launch(
                    op::commit_full_pool_11,
                    (device, k, v, full, pool_k, pool_v),
                )
                .await?
            }
            Self::V12 { full, .. } => {
                launch(
                    op::commit_full_pool_12,
                    (device, k, v, full, pool_k, pool_v),
                )
                .await?
            }
            Self::V13 { full, .. } => {
                launch(
                    op::commit_full_pool_13,
                    (device, k, v, full, pool_k, pool_v),
                )
                .await?
            }
            Self::V14 { full, .. } => {
                launch(
                    op::commit_full_pool_14,
                    (device, k, v, full, pool_k, pool_v),
                )
                .await?
            }
            Self::V15 { full, .. } => {
                launch(
                    op::commit_full_pool_15,
                    (device, k, v, full, pool_k, pool_v),
                )
                .await?
            }
            Self::V16 { full, .. } => {
                launch(
                    op::commit_full_pool_16,
                    (device, k, v, full, pool_k, pool_v),
                )
                .await?
            }
        }
        Ok(())
    }
}

fn sliding_vectors(sequences: &[Sequence<'_>]) -> (Vec<i32>, Vec<f32>) {
    sliding_vectors_rows(B::SIZE, sequences)
}
fn full_page_vectors(sequences: &[Sequence<'_>], page: usize) -> (Vec<i32>, Vec<f32>, Vec<f32>) {
    full_page_vectors_rows(B::SIZE, sequences, page)
}
pub(crate) fn sliding_vectors_rows(
    rows: usize,
    sequences: &[Sequence<'_>],
) -> (Vec<i32>, Vec<f32>) {
    sliding_vectors_rows_bytes(rows, sequences, 2)
}
pub(crate) fn sliding_vectors_rows_bytes(
    rows: usize, sequences: &[Sequence<'_>], sliding_element_bytes: usize,
) -> (Vec<i32>, Vec<f32>) {
    let count = sequences.len();
    let mut offsets = Vec::with_capacity(rows);
    let mut mask = Vec::with_capacity(rows * Ts::SIZE);
    for row in 0..rows {
        let sequence = &sequences[row.min(count - 1)];
        offsets.push((sequence.blocks.sliding * Ts::SIZE * Ns::SIZE * Ds::SIZE * sliding_element_bytes) as i32);
        for slot in 0..Ts::SIZE {
            mask.push(
                if row < count && slot <= sequence.start_position.min(Ts::SIZE - 1) {
                    1.0
                } else {
                    0.0
                },
            );
        }
    }
    (offsets, mask)
}

pub(crate) fn full_page_vectors_rows(
    rows: usize,
    sequences: &[Sequence<'_>],
    page: usize,
) -> (Vec<i32>, Vec<f32>, Vec<f32>) {
    let count = sequences.len();
    let mut offsets = Vec::with_capacity(rows);
    let mut mask = Vec::with_capacity(rows * Tf::SIZE);
    let mut active = Vec::with_capacity(rows * 8);
    for row in 0..rows {
        let sequence = &sequences[row.min(count - 1)];
        let last = sequence.start_position / Tf::SIZE;
        let live = row < count && page <= last;
        let physical = sequence.blocks.full[if page <= last { page } else { 0 }];
        offsets.push((physical * Tf::SIZE * Df::SIZE * 2) as i32);
        for col in 0..Tf::SIZE {
            mask.push(
                if live && (page < last || col <= sequence.start_position % Tf::SIZE) {
                    1.0
                } else {
                    0.0
                },
            );
        }
        active.extend(std::iter::repeat_n(if live { 1.0 } else { 0.0 }, 8));
    }
    (offsets, mask, active)
}

#[cfg(test)]
mod tests {
    use super::super::model::Input;
    use super::*;
    #[test]
    fn request_metadata_keeps_ragged_pages_and_padding_inactive() {
        let inputs = [Input {
            token: 2,
            soft: None,
            logits: true,
            image: None,
            audio: None,
        }];
        let blocks = [
            Blocks {
                full: vec![17],
                sliding: 7,
            },
            Blocks {
                full: vec![9],
                sliding: 3,
            },
            Blocks {
                full: vec![21, 6],
                sliding: 15,
            },
            Blocks {
                full: vec![31, 25, 8],
                sliding: 1,
            },
        ];
        let positions = [0, 511, 512, 1024];
        let sequences: Vec<_> = (0..4)
            .map(|r| Sequence {
                inputs: &inputs,
                start_position: positions[r],
                blocks: &blocks[r],
            })
            .collect();
        let (offsets, visibility) = sliding_vectors(&sequences);
        for row in 0..B::SIZE {
            for slot in 0..Ts::SIZE {
                assert_eq!(
                    offsets[row],
                    (blocks[row.min(3)].sliding * Ts::SIZE * Ns::SIZE * Ds::SIZE * 2) as i32
                );
                assert_eq!(
                    visibility[row * Ts::SIZE + slot],
                    if row < 4 && slot <= positions[row].min(Ts::SIZE - 1) {
                        1.0
                    } else {
                        0.0
                    }
                );
            }
        }
        let (offsets, mask, active) = full_page_vectors(&sequences, 1);
        assert_eq!(
            offsets[..4],
            [
                (17 * Tf::SIZE * Df::SIZE * 2) as i32,
                (9 * Tf::SIZE * Df::SIZE * 2) as i32,
                (6 * Tf::SIZE * Df::SIZE * 2) as i32,
                (25 * Tf::SIZE * Df::SIZE * 2) as i32
            ]
        );
        for row in 0..B::SIZE {
            assert!(
                active[row * 8..row * 8 + 8]
                    .iter()
                    .all(|&value| value == if row == 2 || row == 3 { 1.0 } else { 0.0 })
            );
            for col in 0..Tf::SIZE {
                assert_eq!(
                    mask[row * Tf::SIZE + col],
                    if row == 3 || (row == 2 && col == 0) {
                        1.0
                    } else {
                        0.0
                    }
                );
            }
        }
        let (offsets, mask, active) = full_page_vectors(&sequences, 2);
        assert_eq!(offsets[3], (8 * Tf::SIZE * Df::SIZE * 2) as i32);
        assert_eq!(mask.iter().filter(|&&v| v == 1.0).count(), 1);
        assert_eq!(active.iter().filter(|&&v| v == 1.0).count(), 8);
        assert_eq!(mask[3 * Tf::SIZE], 1.0);
    }
    #[test]
    fn overlapping_full_pages_disable_independent_pooling_even_with_distinct_rings() {
        let input = [Input {
            token: 2,
            soft: None,
            logits: true,
            image: None,
            audio: None,
        }];
        let a = Blocks {
            full: vec![3, 9],
            sliding: 0,
        };
        let b = Blocks {
            full: vec![4, 9],
            sliding: 1,
        };
        let sequences = [
            Sequence {
                inputs: &input,
                start_position: 0,
                blocks: &a,
            },
            Sequence {
                inputs: &input,
                start_position: 0,
                blocks: &b,
            },
        ];
        assert!(!independent_sequences(&sequences));
        let b = Blocks {
            full: vec![4, 10],
            sliding: 1,
        };
        let sequences = [
            Sequence {
                inputs: &input,
                start_position: 0,
                blocks: &a,
            },
            Sequence {
                inputs: &input,
                start_position: 0,
                blocks: &b,
            },
        ];
        assert!(independent_sequences(&sequences));
    }
}

#[cfg(test)]
mod fixed_e4_metadata_tests {
    use super::*;
    #[test]
    fn storage_dtype_changes_only_byte_addresses_not_visibility_or_owner_order() {
        let blocks=[Blocks{sliding:31,full:vec![255]},Blocks{sliding:0,full:vec![0]}];
        let tokens=[super::super::model::Input{token:2,soft:None,logits:false,image:None,audio:None}];
        for position in [0,511,512,767,1023,1024,32767] {
            let sequences=[Sequence{inputs:&tokens,start_position:position,blocks:&blocks[0]},Sequence{inputs:&tokens,start_position:position,blocks:&blocks[1]}];
            let (old,mask)=sliding_vectors_rows_bytes(16,&sequences,2);
            let (new,mask_e4)=sliding_vectors_rows_bytes(16,&sequences,1);
            assert_eq!(mask,mask_e4);assert_eq!(new[0],31*2_097_152);assert_eq!(new[1],0);
            assert!(old.iter().zip(&new).all(|(a,b)|*a==2*b));
            assert!(mask[2*Ts::SIZE..].iter().all(|&x|x==0.0));
        }
    }
}
