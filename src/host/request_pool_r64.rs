//! Owned per-step metadata for R64 independent requests on the canonical pools.
use super::{
    model::Sequence,
    request_pool::{PoolOffsets, full_page_vectors_rows, sliding_vectors_rows},
};
use crate::{Chip, axes::*};
use furiosa_opt_std::prelude::*;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub(crate) struct FullPage {
    pub offsets: HbmTensor<i32, Chip, m![R64]>,
    pub mask: HbmTensor<f32, Chip, m![R64, Tf]>,
    pub active: HbmTensor<f32, Chip, m![R64,1#8]>,
}
pub(crate) struct Step64 {
    pub ring_offsets: HbmTensor<i32, Chip, m![R64]>,
    pub sliding_mask: HbmTensor<f32, Chip, m![R64, Ts]>,
    pub pages: Vec<FullPage>,
    pub full_offsets: HbmTensor<i32, Chip, m![R64]>,
    pub sliding_offsets: HbmTensor<i32, Chip, m![R64]>,
    pub live: usize,
    pub sliding_prefix512: bool,
    pub segments: Vec<PoolOffsets>,
}
impl Step64 {
    pub async fn new(device: &mut Device, sequences: &[Sequence<'_>]) -> Result<Self> {
        let live = sequences.len();
        if live == 0 || live > R64::SIZE {
            return Err("invalid R64 live width".into());
        }
        let (rings, mask) = sliding_vectors_rows(R64::SIZE, sequences);
        let ring_offsets = HostTensor::<i32, m![R64]>::from_vec(rings)
            .to_hbm(&mut device.pdma)
            .await?;
        let sliding_mask = HostTensor::<f32, m![R64, Ts]>::from_vec(mask)
            .to_hbm(&mut device.pdma)
            .await?;
        let full_offsets = HostTensor::<i32, m![R64]>::from_vec(
            (0..R64::SIZE)
                .map(|r| {
                    let s = &sequences[r.min(live - 1)];
                    ((s.blocks.full[s.start_position / Tf::SIZE] * Tf::SIZE
                        + s.start_position % Tf::SIZE)
                        * Df::SIZE
                        * 2) as i32
                })
                .collect::<Vec<_>>(),
        )
        .to_hbm(&mut device.pdma)
        .await?;
        let sliding_offsets = HostTensor::<i32, m![R64]>::from_vec(
            (0..R64::SIZE)
                .map(|r| {
                    let s = &sequences[r.min(live - 1)];
                    ((s.blocks.sliding * Ts::SIZE + s.start_position % Ts::SIZE)
                        * Ns::SIZE
                        * Ds::SIZE
                        * 2) as i32
                })
                .collect::<Vec<_>>(),
        )
        .to_hbm(&mut device.pdma)
        .await?;
        let mut pages = Vec::new();
        let last = sequences
            .iter()
            .map(|s| s.start_position / Tf::SIZE)
            .max()
            .unwrap();
        for p in 0..=last {
            let (offsets, mask, active) = full_page_vectors_rows(R64::SIZE, sequences, p);
            pages.push(FullPage {
                offsets: HostTensor::<i32, m![R64]>::from_vec(offsets)
                    .to_hbm(&mut device.pdma)
                    .await?,
                mask: HostTensor::<f32, m![R64, Tf]>::from_vec(mask)
                    .to_hbm(&mut device.pdma)
                    .await?,
                active: HostTensor::<f32, m![R64,1#8]>::from_vec(active)
                    .to_hbm(&mut device.pdma)
                    .await?,
            });
        }
        let mut segments = Vec::new();
        if live < R64::SIZE {
            for chunk in sequences.chunks(B::SIZE) {
                segments.push(PoolOffsets::new(device, chunk).await?);
            }
        }
        Ok(Self {
            ring_offsets,
            sliding_mask,
            pages,
            full_offsets,
            sliding_offsets,
            live,
            sliding_prefix512: sequences.iter().all(|s| s.start_position < 512),
            segments,
        })
    }
}
