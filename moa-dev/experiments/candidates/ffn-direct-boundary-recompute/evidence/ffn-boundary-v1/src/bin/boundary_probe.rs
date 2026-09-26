use furiosa_opt_std::prelude::*;
use furiosa_opt_gemma4::{Chip, axes::{H, L}, probe::{self, Term}};

const FP4: [f64; 16] = [0.,0.5,1.,1.5,2.,3.,4.,6.,-0.,-0.5,-1.,-1.5,-2.,-3.,-4.,-6.];
fn code(r: usize, k: usize, seed: usize) -> u8 {
    let mut v = (((r+seed*L::SIZE)*H::SIZE+k) as u32).wrapping_mul(0x9e3779b9).wrapping_add(0x1234567);
    v ^= v >> 16; v = v.wrapping_mul(0x85ebca6b); v ^= v >> 13;
    (v & 15) as u8
}
fn activation(case: usize, _term: usize, k: usize) -> f64 {
    match case {
        0 => ((k*7%17) as f64-8.)/32.,
        1 => match (k/16%2,k%16) {(0,0)=>1.,(0,1)=>15./256.,(1,0)=>-1.,(1,1)=>-14./256.,_=>0.},
        2 => 0.,
        _ => ((k*11%17) as f64-8.)*2048.,
    }
}
fn packed(seed: usize) -> Vec<u8> {
    let mut w = vec![0u8;L::SIZE*H::SIZE/2];
    for r in 0..L::SIZE {for k in 0..H::SIZE {
        w[(r*H::SIZE+k)/2] |= code(r,k,seed) << (4*(k%2));
    }}
    w
}
fn blockscale(r: usize,b: usize)->f64 { [0.25,0.5,1.,2.][(r*3+b)%4] }
fn check_dot(case: usize, mode: &str, seed: usize, actual: &[f32]) -> usize {
    let mut failures=0;let mut nonfinite=0;let mut max_abs=0f64;
    for r in 0..L::SIZE {
        let mut expected=0f64;
        for b in 0..H::SIZE/16 {
            let mut partial=0f64;
            for k in b*16..b*16+16 {partial+=FP4[code(r,k,seed) as usize]*activation(case,0,k);}
            assert_eq!(partial,(partial as f32) as f64);
            expected+=partial*blockscale(r,b);
        }
        assert_eq!(expected,(expected as f32) as f64);
        let a=actual[r];
        if !a.is_finite() {nonfinite+=1;}
        if a as f64!=expected {
            if failures<12 {println!("BOUNDARY_BAD mode={mode} case={case} projection={seed} row={r} cluster={} slice={} local_row={} got={a} expected={expected}",r/7680,(r%7680)/30,r%30);}
            failures+=1;if a.is_finite() {max_abs=max_abs.max((a as f64-expected).abs());}
        }
    }
    println!("BOUNDARY_RESULT mode={mode} case={case} projection={seed} checked={} mismatches={failures} nonfinite={nonfinite} max_abs={max_abs}",actual.len());failures
}

#[tokio::main]
async fn main() {
    let mode=std::env::args().nth(1).expect("mode");
    assert!(["decode","materialized","fused","hybrid"].contains(&mode.as_str()));
    let mut d=Device::new(probe::fused.topology()).unwrap();
    let w: HbmTensor<f4e2m1,Chip,m![L,H]> = HostTensor::<f4e2m1,m![L,H]>::from_buf(packed(0))
        .to_hbm(&mut d.pdma).await.unwrap();
    let mut failures=0;
    if mode=="decode" {
        let mut y: HbmTensor<f8e5m2,Chip,m![L,H]> = HostTensor::<f8e5m2,m![L,H]>::from_vec(vec![f8e5m2::from_f32(f32::NAN);L::SIZE*H::SIZE]).to_hbm(&mut d.pdma).await.unwrap();
        launch(probe::decode,(&mut d,&w,&mut y)).await.unwrap();
        let y: HostTensor<f8e5m2,m![L,H]> = y.to_host(&mut d.pdma).await.unwrap();
        let a=y.into_vec();
        for r in 0..L::SIZE {for k in 0..H::SIZE {
            let expected=FP4[code(r,k,0) as usize];
            let got=a[r*H::SIZE+k].to_f32();
            if got as f64 != expected {
                if failures<12 {println!("BOUNDARY_BAD mode=decode row={r} k={k} got={got} expected={expected}",r/7680,(r%7680)/30,r%30);}
                failures+=1;
            }
        }}
        println!("BOUNDARY_RESULT mode=decode case=0 checked={} mismatches={failures}",a.len());
    } else {
        let g: HbmTensor<f4e2m1,Chip,m![L,H]> = HostTensor::<f4e2m1,m![L,H]>::from_buf(packed(1)).to_hbm(&mut d.pdma).await.unwrap();
        for case in 0..4 {
            let values: Vec<bf16> = (0..H::SIZE).map(|k| {
                let v=activation(case,0,k); let f=bf16::from_f32(v as f32);
                assert_eq!(f.to_f32() as f64,v);f
            }).collect();
            let x: HbmTensor<bf16,Chip,m![H]> = HostTensor::<bf16,m![H]>::from_vec(values)
                .to_hbm(&mut d.pdma).await.unwrap();
            let bs: HbmTensor<f8e4m3,Chip,m![L,H/16]> = HostTensor::<f8e4m3,m![L,H/16]>::from_vec((0..L::SIZE*(H::SIZE/16)).map(|i|f8e4m3::from_f32(blockscale(i/(H::SIZE/16),i%(H::SIZE/16)) as f32))).to_hbm(&mut d.pdma).await.unwrap();
            let mut y: HbmTensor<f32,Chip,m![L]> = HostTensor::<f32,m![L]>::from_vec(vec![f32::NAN;L::SIZE])
                .to_hbm(&mut d.pdma).await.unwrap();
            let mut z: HbmTensor<f32,Chip,m![L]> = HostTensor::<f32,m![L]>::from_vec(vec![f32::NAN;L::SIZE]).to_hbm(&mut d.pdma).await.unwrap();
            if mode=="materialized" {launch(probe::materialized,(&mut d,&x,&w,&g,&bs,&bs,&mut y,&mut z)).await.unwrap();}
            else if mode=="hybrid" {launch(probe::hybrid,(&mut d,&x,&w,&g,&bs,&bs,&mut y,&mut z)).await.unwrap();}
            else {launch(probe::fused,(&mut d,&x,&w,&g,&bs,&bs,&mut y,&mut z)).await.unwrap();}
            let y: HostTensor<f32,m![L]> = y.to_host(&mut d.pdma).await.unwrap();
            failures+=check_dot(case,&mode,0,&y.into_vec());
            let z: HostTensor<f32,m![L]> = z.to_host(&mut d.pdma).await.unwrap();
            failures+=check_dot(case,&mode,1,&z.into_vec());
        }
    }
    println!("BOUNDARY_COMPLETE mode={mode} mismatches={failures}");
    if failures>0 {std::process::exit(1);}
}
