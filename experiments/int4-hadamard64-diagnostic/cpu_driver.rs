//! Pure-std fixed-vector entry; no SDK or device dependency.
#[path = "../../src/bin/support/int4_hadamard64.rs"]
mod rotation;
use std::io::{self, Read};
fn main() {
    let mut text = String::new();
    io::stdin().read_to_string(&mut text).unwrap();
    let mut v = text.split_whitespace();
    let count: usize = v.next().unwrap().parse().unwrap();
    for i in 0..count {
        let base: usize = v.next().unwrap().parse().unwrap();
        let input: [f32; 64] =
            std::array::from_fn(|_| f32::from_bits(v.next().unwrap().parse().unwrap()));
        let result = rotation::forward32(&input, base).unwrap();
        println!(
            "{} {} {} {}",
            i,
            (0..64)
                .map(|k| (rotation::sign(base + k) as i32).to_string())
                .collect::<Vec<_>>()
                .join(","),
            result
                .iter()
                .map(|x| x.to_bits().to_string())
                .collect::<Vec<_>>()
                .join(","),
            result
                .iter()
                .map(|&x| rotation::bf16_rne(x).to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
    }
    assert!(v.next().is_none());
}
