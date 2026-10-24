use crate::axes::{B, Tf, Ts};
use furiosa_opt_std::prelude::*;

const MASKED: f32 = -1e30;
pub(super) fn full_mask(start: usize, count: usize) -> Vec<f32> {
    full_mask_rows(start, count, B::SIZE)
}

/// Per-query additive masks for a page-local chunk. Padding repeats the last
/// live query, so an unused row still has a finite softmax denominator.
pub(super) fn full_mask_rows(start: usize, count: usize, rows: usize) -> Vec<f32> {
    assert!(count > 0 && count <= rows);
    assert!(start % Tf::SIZE + count <= Tf::SIZE);
    (0..rows)
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
    sliding_masks_rows(start, count, B::SIZE)
}

/// The ring still contains only history before `start`; current chunk K/V stays
/// separate until every query has attended. `rows` may be B16 or P128.
pub(super) fn sliding_masks_rows(start: usize, count: usize, rows: usize) -> (Vec<f32>, Vec<f32>) {
    assert!(count > 0 && count <= rows && rows <= Ts::SIZE);
    let mut old = Vec::with_capacity(rows * Ts::SIZE);
    let mut chunk = Vec::with_capacity(rows * rows);
    for row in 0..rows {
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
        for key in 0..rows {
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
    #[test]
    fn p128_masks_preserve_absolute_history_and_ignore_padding() {
        let rows = 128;
        for start in [0usize, 384, 896, 1024, 2048, 8064] {
            for count in [1usize, 17, 127, 128] {
                let (old, chunk) = sliding_masks_rows(start, count, rows);
                let full = full_mask_rows(start, count, rows);
                for row in 0..rows {
                    let position = start + row.min(count - 1);
                    let mut expected = vec![false; Ts::SIZE];
                    for prior in start.saturating_sub(Ts::SIZE)..start {
                        if position - prior < Ts::SIZE {
                            expected[prior % Ts::SIZE] = true;
                        }
                    }
                    for slot in 0..Ts::SIZE {
                        assert_eq!(old[row * Ts::SIZE + slot] == 0.0, expected[slot]);
                    }
                    for key in 0..rows {
                        assert_eq!(
                            chunk[row * rows + key] == 0.0,
                            key < count && start + key <= position
                        );
                    }
                    for column in 0..Tf::SIZE {
                        let key_position = start / Tf::SIZE * Tf::SIZE + column;
                        assert_eq!(
                            full[row * Tf::SIZE + column] == 0.0,
                            key_position <= position
                        );
                    }
                }
            }
        }
    }
}
