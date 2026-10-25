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
    pub sliding_short_groups: u8,
    pub sliding_prefix768: bool,
    pub sliding_group_policy: Option<HbmScalar<i32>>,
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
        let sliding_short_groups = sliding_short_group_mask(sequences.iter().map(|s| s.start_position));
        let sliding_prefix768 = prefix768_eligible(sequences.iter().map(|s| s.start_position));
        let sliding_group_policy = if sliding_short_groups != 0 && sliding_short_groups != 255 {
            Some(HbmScalar::from_host(i32::from(sliding_short_groups), &mut device.pdma).await?)
        } else {
            None
        };
        Ok(Self {
            ring_offsets,
            sliding_mask,
            pages,
            full_offsets,
            sliding_offsets,
            live,
            sliding_short_groups,
            sliding_prefix768,
            sliding_group_policy,
            segments,
        })
    }
}

/// A group can omit the upper half only when every live row is before position512.
/// Padded rows are fully masked, so groups without a live row may use the prefix.
fn sliding_short_group_mask(positions: impl IntoIterator<Item = usize>) -> u8 {
    positions.into_iter().enumerate().fold(255, |mask, (row, position)| {
        if position >= 512 { mask & !(1 << (row / 8)) } else { mask }
    })
}

/// Keep every existing512-eligible group, including inactive groups, on its
/// faster current route. Actual C7 history gate showed a4.77ms attention regression
/// when global768 replaced the seven cheaper inactive512 groups.
/// Only unwrapped histories below768 can omit the final quarter of the ring.
fn prefix768_eligible(positions: impl IntoIterator<Item = usize>) -> bool {
    let positions: Vec<_> = positions.into_iter().collect();
    positions.len() > R64::SIZE - 8 && positions.len() <= R64::SIZE
        && positions.iter().all(|&p| p < 768)
        && positions.chunks(8).all(|group| group.iter().any(|&p| p >= 512))
}

#[cfg(test)]
mod tests {
    use super::{sliding_short_group_mask, prefix768_eligible};

    #[test]
    fn prefix768_preserves512_priority_and_rejects_future_or_wrapped_history() {
        assert!(prefix768_eligible([512;64]));
        assert!(prefix768_eligible([767;64]));
        assert!(!prefix768_eligible([512;7]));
        assert!(!prefix768_eligible([639;17]));
        assert!(!prefix768_eligible([512;56]));
        assert!(prefix768_eligible([512;57]));
        assert!(!prefix768_eligible([511;64]));
        assert!(!prefix768_eligible([768;64]));
        assert!(!prefix768_eligible([1024;64]));
        assert!(!prefix768_eligible([]));
        let mut p=[512;64]; p[8..16].fill(511);
        assert!(!prefix768_eligible(p));
        p[15]=512; assert!(prefix768_eligible(p));
        p[63]=768; assert!(!prefix768_eligible(p));
    }

    #[test]
    fn prefix_policy_tracks_live_group_boundaries_and_ring_wrap() {
        assert_eq!(sliding_short_group_mask([511; 64]), 255);
        assert_eq!(sliding_short_group_mask([512; 64]), 0);
        assert_eq!(sliding_short_group_mask([1024; 64]), 0);
        assert_eq!(sliding_short_group_mask([128; 17]), 255);
        assert_eq!(sliding_short_group_mask([512; 17]), 248);
        let mut p = [0; 64];
        for row in [7, 8, 31, 63] { p[row] = 512; }
        assert_eq!(sliding_short_group_mask(p), 0b01110100);
    }
}
