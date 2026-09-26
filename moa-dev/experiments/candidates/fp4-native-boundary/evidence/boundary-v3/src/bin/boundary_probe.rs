use furiosa_opt_std::prelude::*;
use furiosa_opt_gemma4::{Chip, axes::{H, L}, probe::{self, Term}};

const FP4: [f64; 16] = [0.,0.5,1.,1.5,2.,3.,4.,6.,-0.,-0.5,-1.,-1.5,-2.,-3.,-4.,-6.];
fn code(r: usize, k: usize) -> u8 {
    let mut v = ((r*H::SIZE+k) as u32).wrapping_mul(0x9e3779b9).wrapping_add(0x1234567);
    v ^= v >> 16; v = v.wrapping_mul(0x85ebca6b); v ^= v >> 13;
    (v & 15) as u8
}
fn activation(case: usize, term: usize, k: usize) -> f64 {
    if case == 0 { [1.,0.5,0.25][term] }
    else if case == 1 { ((k*7+term*3+k/16)%9) as f64 / 8. - 0.5 }
    else if case == 2 { if term == 0 { ((k*7%17) as f64-8.)*2048. } else { 0. } }
    else { (((k*7+term*3+k/16)%9) as f64-4.) * [4096.,16.,0.25][term] }
}
fn packed() -> Vec<u8> {
    let mut w = vec![0u8;L::SIZE*H::SIZE/2];
    for r in 0..L::SIZE {for k in 0..H::SIZE {
        w[(r*H::SIZE+k)/2] |= code(r,k) << (4*(k%2));
    }}
    w
}
fn check_dot(case: usize, mode: &str, actual: &[f32]) -> usize {
    let mut failures=0; let mut nonfinite=0; let mut max_abs=0f64;
    let mut blocks=std::collections::BTreeMap::<usize,usize>::new();
    for r in 0..L::SIZE {for b in 0..H::SIZE/16 {
        let mut expected=0f64;
        for k in b*16..b*16+16 {for term in 0..3 {
            expected+=FP4[code(r,k) as usize]*activation(case,term,k);
        }}
        expected *= if case < 2 {1.25} else {1./65536.};
        assert_eq!(expected,(expected as f32) as f64);
        let a=actual[r*(H::SIZE/16)+b];
        if !a.is_finite() {nonfinite+=1;}
        if a as f64 != expected {
            if failures<12 {println!("BOUNDARY_BAD mode={mode} case={case} row={r} block={b} got={a} expected={expected}");}
            failures+=1; *blocks.entry(b).or_default()+=1;
            if a.is_finite() {max_abs=max_abs.max((a as f64-expected).abs());}
        }
    }}
    println!("BOUNDARY_RESULT mode={mode} case={case} checked={} mismatches={failures} nonfinite={nonfinite} max_abs={max_abs}",actual.len());
    if failures>0 {println!("BOUNDARY_BAD_BLOCKS {blocks:?}");}
    failures
}

#[tokio::main]
async fn main() {
    let mode=std::env::args().nth(1).expect("mode");
    assert!(["decode","materialized","fused"].contains(&mode.as_str()));
    let mut d=Device::new(probe::fused.topology()).unwrap();
    let w: HbmTensor<f4e2m1,Chip,m![L,H]> = HostTensor::<f4e2m1,m![L,H]>::from_buf(packed())
        .to_hbm(&mut d.pdma).await.unwrap();
    let mut failures=0;
    if mode=="decode" {
        let mut y: HbmTensor<f8e5m2,Chip,m![L,H]> = HostTensor::<f8e5m2,m![L,H]>::from_vec(vec![f8e5m2::from_f32(f32::NAN);L::SIZE*H::SIZE]).to_hbm(&mut d.pdma).await.unwrap();
        launch(probe::decode,(&mut d,&w,&mut y)).await.unwrap();
        let y: HostTensor<f8e5m2,m![L,H]> = y.to_host(&mut d.pdma).await.unwrap();
        let a=y.into_vec();
        for r in 0..L::SIZE {for k in 0..H::SIZE {
            let expected=FP4[code(r,k) as usize];
            let got=a[r*H::SIZE+k].to_f32();
            if got as f64 != expected {
                if failures<12 {println!("BOUNDARY_BAD mode=decode row={r} k={k} got={got} expected={expected}");}
                failures+=1;
            }
        }}
        println!("BOUNDARY_RESULT mode=decode case=0 checked={} mismatches={failures}",a.len());
    } else {
        for case in 0..4 {
            let scale: HbmTensor<f32,Chip,m![1]> = HostTensor::<f32,m![1]>::from_vec(vec![if case<2 {1.25} else {1./65536.}])
                .to_hbm(&mut d.pdma).await.unwrap();
            let mut terms=Vec::with_capacity(3*H::SIZE);
            for term in 0..3 {for k in 0..H::SIZE {
                let v=activation(case,term,k);
                let f=f8e5m2::from_f32(v as f32);
                assert_eq!(f.to_f32() as f64,v); terms.push(f);
            }}
            let x: HbmTensor<f8e5m2,Chip,m![Term,H]> = HostTensor::<f8e5m2,m![Term,H]>::from_vec(terms)
                .to_hbm(&mut d.pdma).await.unwrap();
            let mut y: HbmTensor<f32,Chip,m![L,H/16]> = HostTensor::<f32,m![L,H/16]>::from_vec(vec![f32::NAN;L::SIZE*(H::SIZE/16)])
                .to_hbm(&mut d.pdma).await.unwrap();
            if mode=="materialized" {launch(probe::materialized,(&mut d,&x,&scale,&w,&mut y)).await.unwrap();}
            else {launch(probe::fused,(&mut d,&x,&scale,&w,&mut y)).await.unwrap();}
            let y: HostTensor<f32,m![L,H/16]> = y.to_host(&mut d.pdma).await.unwrap();
            failures+=check_dot(case,&mode,&y.into_vec());
        }
    }
    println!("BOUNDARY_COMPLETE mode={mode} mismatches={failures}");
    if failures>0 {std::process::exit(1);}
}
