use crate::axes::{B, Tf, Ts};
use furiosa_opt_std::prelude::*;

const MASKED: f32 = -1e30;
pub(super) fn full_mask(start: usize, count: usize) -> Vec<f32> {
    (0..B::SIZE)
        .flat_map(|row| {
            (0..Tf::SIZE).map(move |col| {
                if col <= (start + row.min(count - 1)) % Tf::SIZE {
                    0.0
                } else {
                    MASKED
                }
            })
        })
        .collect()
}

pub(super) fn sliding_masks(start: usize, count: usize) -> (Vec<f32>, Vec<f32>) {
    let mut old = Vec::with_capacity(B::SIZE * Ts::SIZE);
    let mut chunk = Vec::with_capacity(B::SIZE * B::SIZE);
    for row in 0..B::SIZE {
        let row = row.min(count - 1);
        let position = start + row;
        let lower = position.saturating_sub(Ts::SIZE - 1);
        for slot in 0..Ts::SIZE {
            let previous = if start == 0 {
                None
            } else {
                let candidate = ((start - 1) / Ts::SIZE) * Ts::SIZE + slot;
                if candidate < start {
                    Some(candidate)
                } else {
                    candidate.checked_sub(Ts::SIZE)
                }
            };
            old.push(if previous.is_some_and(|p| p >= lower && p < start) {
                0.0
            } else {
                MASKED
            });
        }
        for key in 0..B::SIZE {
            chunk.push(if key <= row && key < count {
                0.0
            } else {
                MASKED
            });
        }
    }
    (old, chunk)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sliding_masks_match_absolute_causal_window_at_wraps() {
        for start in [0, 1, 15, 511, 1008, 1023, 1024, 1025, 2048, 8190] {
            for count in [1, 3, 16] {
                let (old, chunk) = sliding_masks(start, count);
                for row in 0..B::SIZE {
                    let effective = row.min(count - 1);
                    let position = start + effective;
                    let mut expected = vec![false; Ts::SIZE];
                    for prior in start.saturating_sub(Ts::SIZE)..start {
                        if position - prior < Ts::SIZE {
                            expected[prior % Ts::SIZE] = true;
                        }
                    }
                    for slot in 0..Ts::SIZE {
                        assert_eq!(
                            old[row * Ts::SIZE + slot] == 0.0,
                            expected[slot],
                            "start={start} count={count} row={row} slot={slot}"
                        );
                    }
                    for key in 0..B::SIZE {
                        assert_eq!(
                            chunk[row * B::SIZE + key] == 0.0,
                            key <= effective && key < count
                        );
                    }
                    assert_eq!(
                        chunk[row * B::SIZE],
                        0.0,
                        "every query includes a finite current key"
                    );
                }
            }
        }
    }
    #[test]
    fn full_last_page_excludes_future_rows_and_masks_padding_like_last_live_row() {
        for start in [0, 496, 511, 512, 1008, 8191] {
            let count = B::SIZE.min(Tf::SIZE - start % Tf::SIZE);
            let mask = full_mask(start, count);
            for row in 0..B::SIZE {
                for col in 0..Tf::SIZE {
                    assert_eq!(
                        mask[row * Tf::SIZE + col] == 0.0,
                        (start / Tf::SIZE) * Tf::SIZE + col <= start + row.min(count - 1)
                    );
                }
            }
        }
    }
}
