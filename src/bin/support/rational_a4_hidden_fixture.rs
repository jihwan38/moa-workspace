//! Full512×15360 fixtures; no candidate reciprocal or quantization formula.
pub const ROWS: usize = 512;
pub const WIDTH: usize = 15360;
pub const CASES: usize = 4;
const FLOOR: u16 = 15 << 7;

pub fn source_coordinate(case: usize, row: usize, col: usize) -> (usize, usize) {
    match case {
        2 => ((73 * row + 127) % ROWS, (127 * col + 73) % WIDTH),
        3 => ((301 * row + 19) % ROWS, (253 * col + 521) % WIDTH),
        _ => panic!("only permutation cases have a source coordinate"),
    }
}

fn signed(values: &mut Vec<u16>, x: u16) {
    values.extend([x, x | 0x8000]);
}

pub fn fixture(case: usize) -> Vec<u16> {
    assert!(case < CASES);
    if case >= 2 {
        let source = fixture(case - 2);
        return (0..ROWS * WIDTH).map(|i| {
            let (r, c) = source_coordinate(case, i / WIDTH, i % WIDTH);
            source[r * WIDTH + c]
        }).collect();
    }
    let mut all = Vec::with_capacity(ROWS * WIDTH);
    for row in 0..ROWS {
        let mut values = Vec::new();
        if case == 0 {
            // Each of all512 rows has a distinct maximum and scale.
            let ea = [15u16, 127, 200, 254][row / 128];
            let ma = 128 + (row % 128) as u16;
            // Maximum lives only in the final two columns, beyond H3840.
            for delta in 0..5 {
                for mx in 128..256 {
                    if delta == 0 && mx >= ma { continue; }
                    signed(&mut values, ((ea - delta) << 7) | (mx - 128));
                }
            }
            values.extend([0, 0x8000, 1, 0x807f]);
        } else {
            match row % 8 {
                0 => values.extend([0, 0x8000]),
                1 => values.extend([1, 0x8001, 0x007f, 0x807f, 0, 0x8000]),
                2 => values.extend([0x0080, 0x8080, FLOOR - 1, (FLOOR - 1) | 0x8000]),
                3 => {
                    signed(&mut values, FLOOR + (row % 128) as u16);
                    values.extend([FLOOR, FLOOR | 0x8000, FLOOR - 1, 1, 0x807f]);
                }
                4 => {
                    // A=14*2^(ea-130), x=(1,3,...,13)*2^(ea-130).
                    // Include both neighboring BF16 values around every exact half.
                    let ea = [15i32, 16, 31, 63, 100, 127, 128, 200, 250, 254][(row / 8) % 10];
                    let shift = (ea - 130) * 128;
                    let scaled = |x: u32| (((x as f32).to_bits() >> 16) as i32 + shift) as u16;
                    signed(&mut values, scaled(14));
                    for x in (1..14).step_by(2) {
                        let b = scaled(x);
                        signed(&mut values, b - 1);
                        signed(&mut values, b);
                        signed(&mut values, b + 1);
                    }
                }
                5 => values.extend([0x7f7f, 0xff7f, 0x7f7e, 0xff7e, FLOOR, 0, 0x8000, 1]),
                6 => values.extend([FLOOR, FLOOR | 0x8000, 0x0080, 0x8080, 0x0081, 0x8081, 1, 0x807f]),
                7 => {
                    let ea = [1u16, 14, 15, 16, 31, 63, 100, 127, 128, 200, 253, 254][(row / 8) % 12];
                    let ma = 128 + ((row * 29 + 17) % 128) as u16;
                    signed(&mut values, (ea << 7) | (ma - 128));
                    for delta in [0u16, 1, 2, 3, 4, 5, 6, 7, 15, 31, 63, 119, 120, 200, 253] {
                        if delta >= ea { continue; }
                        for mx in [128u16, 129, 159, 192, 223, 254, 255] {
                            if delta == 0 && mx > ma { continue; }
                            signed(&mut values, ((ea - delta) << 7) | (mx - 128));
                        }
                    }
                    values.extend([0, 0x8000, 1, 0x807f]);
                }
                _ => unreachable!(),
            }
        }
        let mut full_row: Vec<u16> = (0..WIDTH).map(|i| values[i % values.len()]).collect();
        if case == 0 {
            let a = ([15u16, 127, 200, 254][row / 128] << 7) | (row % 128) as u16;
            full_row[WIDTH - 2] = a;
            full_row[WIDTH - 1] = a | 0x8000;
        }
        all.extend(full_row);
    }
    all
}

#[cfg(test)]
#[path = "rational_a4_oracle.rs"]
mod oracle;

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn distinct_rows_and_finite_tie_neighbors() {
        let bits = fixture(0);
        let mut scales = BTreeSet::new();
        for (r, row) in bits.chunks_exact(WIDTH).enumerate() {
            let a = row.iter().map(|x| x & 0x7fff).max().unwrap();
            assert_eq!(a, ([15u16, 127, 200, 254][r / 128] << 7) | (r % 128) as u16);
            assert!(row[..WIDTH - 2].iter().all(|b| (b & 0x7fff) < a));
            scales.insert(oracle::reference(row).1.to_bits());
        }
        assert_eq!(scales.len(), ROWS);
        let bits = fixture(1);
        for (r, row) in bits.chunks_exact(WIDTH).enumerate() {
            assert!(row.iter().all(|b| ((b & 0x7fff) >> 7) != 255));
            if r % 8 == 4 {
                let q = oracle::reference(row).0;
                for k in 0..7 {
                    let i = 2 + 6 * k;
                    assert_eq!(&q[i..i + 6], &[k as i32, -(k as i32), k as i32 + 1, -(k as i32 + 1), k as i32 + 1, -(k as i32 + 1)]);
                }
            }
        }
    }

    #[test]
    fn quotient_oracle_reads_the_entire_arbitrary_width() {
        let mut row = vec![(1f32.to_bits() >> 16) as u16; WIDTH];
        row[WIDTH - 1] = (14f32.to_bits() >> 16) as u16;
        let (codes, scale) = oracle::reference(&row);
        assert_eq!(codes.len(), WIDTH);
        assert!(codes[..WIDTH - 1].iter().all(|&q| q == 1));
        assert_eq!(codes[WIDTH - 1], 7);
        assert_eq!(scale.to_bits(), 2f32.to_bits());
        row[WIDTH - 1] = 0;
        let (codes, scale) = oracle::reference(&row);
        assert!(codes[..WIDTH - 1].iter().all(|&q| q == 7));
        assert_eq!(codes[WIDTH - 1], 0);
        assert_eq!(scale.to_bits(), 0x3e124925);
    }

    #[test]
    fn both_full_bijections_cover_rows_and_channels() {
        for case in [2, 3] {
            let rows: BTreeSet<_> = (0..ROWS).map(|r| source_coordinate(case, r, 0).0).collect();
            let cols: BTreeSet<_> = (0..WIDTH).map(|c| source_coordinate(case, 0, c).1).collect();
            assert_eq!(rows.len(), ROWS);
            assert_eq!(cols.len(), WIDTH);
            let original = fixture(case - 2);
            let actual = fixture(case);
            assert_ne!(actual, original);
            for (i, &value) in actual.iter().enumerate() {
                let (r, c) = source_coordinate(case, i / WIDTH, i % WIDTH);
                assert_eq!(value, original[r * WIDTH + c]);
            }
        }
    }
}
