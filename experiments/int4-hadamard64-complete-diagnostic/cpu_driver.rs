#[path="../../src/bin/support/int4_hadamard64_transform.rs"] mod transform;
use std::io::{self,Read};
fn main(){
    let mut s=String::new();io::stdin().read_to_string(&mut s).unwrap();let mut it=s.split_whitespace();
    let n:usize=it.next().unwrap().parse().unwrap();
    for case in 0..n{
        let base:usize=it.next().unwrap().parse().unwrap();let width=(base+64).div_ceil(960)*960;
        let mut row=vec![0u16;width];
        for k in 0..64{let b:u32=it.next().unwrap().parse().unwrap();assert_eq!(b&65535,0);row[base+k]=(b>>16)as u16;}
        let a=transform::rotate_row(&row).unwrap();let b=transform::reference_row(&row).unwrap();assert_eq!(a,b);
        println!("{} {}",case,a[base..base+64].iter().map(u16::to_string).collect::<Vec<_>>().join(","));
    }
    assert!(it.next().is_none());
}
