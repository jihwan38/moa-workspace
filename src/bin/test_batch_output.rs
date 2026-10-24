//! Independent dyadic oracle for batched attention output projection and BF16 sum trees.
use furiosa_opt_std::prelude::*;
use furiosa_opt_gemma4::{axes::*,Chip,ops_batch};
fn x(b:usize,k:usize)->f32 { ((b*17+k*13)%31) as f32/16.0-15.0/16.0 }
fn w(n:usize,k:usize)->f32 { ((n*5+k*7)%17) as f32/8.0-1.0 }
fn scale(n:usize)->f32 { 1.0+(n%7) as f32/8.0 }
#[tokio::main]
async fn main() {
    let mut device=Device::new(ops_batch::project_sliding_output_batch.topology()).unwrap();
    sliding(&mut device).await;
    full(&mut device).await;
}

async fn sliding(device:&mut Device) {
    let input:HbmTensor<bf16,Chip,m![B,Ns,Gs,Ds]>=HostTensor::<bf16,m![B,Ns,Gs,Ds]>::from_vec((0..B::SIZE).flat_map(|b|(0..Qs::SIZE).map(move|k|bf16::from_f32(x(b,k))))).to_hbm(&mut device.pdma).await.unwrap();
    let weight:HbmTensor<f8e4m3,Chip,m![H,Qs]>=HostTensor::<f8e4m3,m![H,Qs]>::from_vec((0..H::SIZE).flat_map(|n|(0..Qs::SIZE).map(move|k|f8e4m3::from_f32(w(n,k))))).to_hbm(&mut device.pdma).await.unwrap();
    let scales:HbmTensor<bf16,Chip,m![H]>=HostTensor::<bf16,m![H]>::from_vec((0..H::SIZE).map(|n|bf16::from_f32(scale(n)))).to_hbm(&mut device.pdma).await.unwrap();
    let mut output:HbmTensor<bf16,Chip,m![B,H]>=HostTensor::<bf16,m![B,H]>::from_vec((0..B::SIZE*H::SIZE).map(|_|bf16::from_f32(f32::NAN))).to_hbm(&mut device.pdma).await.unwrap();
    let start=std::time::Instant::now();
    launch(ops_batch::project_sliding_output_batch,(device,&input,&weight,&scales,&mut output)).await.unwrap();
    let actual:HostTensor<bf16,m![B,H]>=output.to_host(&mut device.pdma).await.unwrap();
    let cold_ms=start.elapsed().as_secs_f64()*1000.0;
    let actual=actual.into_vec();
    let mut dots=vec![0f32;B::SIZE*17];
    for b in 0..B::SIZE { for n in 0..17 {
        let mut partials:Vec<f32>=(0..4).map(|chunk|bf16::from_f32((chunk*1024..(chunk+1)*1024).map(|k|x(b,k)*w(n,k)).sum()).to_f32()).collect();
        while partials.len()>1 { partials=partials.chunks_exact(2).map(|p|bf16::from_f32(p[0]+p[1]).to_f32()).collect(); }
        dots[b*17+n]=partials[0];
    }}
    let mut failures=0;
    for b in 0..B::SIZE { for n in 0..H::SIZE {
        let expected=bf16::from_f32(dots[b*17+n%17]*scale(n));
        let got=actual[b*H::SIZE+n];
        if got.to_bits()!=expected.to_bits() {
            if failures<10 {eprintln!("sliding row={b} col={n} got={} expected={}",got.to_f32(),expected.to_f32());}
            failures+=1;
        }
    }}
    let start=std::time::Instant::now();
    for _ in 0..5 { launch(ops_batch::project_sliding_output_batch,(device,&input,&weight,&scales,&mut output)).await.unwrap(); }
    let _:HostTensor<bf16,m![B,H]>=output.to_host(&mut device.pdma).await.unwrap();
    println!("sliding output B={} K={} H={} checked={} mismatches={failures} cold_ms={cold_ms:.3} warm5launches1read_ms={:.3}",B::SIZE,Qs::SIZE,H::SIZE,actual.len(),start.elapsed().as_secs_f64()*1000.0/5.0);
    assert_eq!(failures,0);
}

async fn full(device:&mut Device) {
    let input:HbmTensor<bf16,Chip,m![B,Gf,Df]>=HostTensor::<bf16,m![B,Gf,Df]>::from_vec((0..B::SIZE).flat_map(|b|(0..Qf::SIZE).map(move|k|bf16::from_f32(x(b,k))))).to_hbm(&mut device.pdma).await.unwrap();
    let weight:HbmTensor<f8e4m3,Chip,m![H,Qf]>=HostTensor::<f8e4m3,m![H,Qf]>::from_vec((0..H::SIZE).flat_map(|n|(0..Qf::SIZE).map(move|k|f8e4m3::from_f32(w(n,k))))).to_hbm(&mut device.pdma).await.unwrap();
    let scales:HbmTensor<bf16,Chip,m![H]>=HostTensor::<bf16,m![H]>::from_vec((0..H::SIZE).map(|n|bf16::from_f32(scale(n)))).to_hbm(&mut device.pdma).await.unwrap();
    let mut output:HbmTensor<bf16,Chip,m![B,H]>=HostTensor::<bf16,m![B,H]>::from_vec((0..B::SIZE*H::SIZE).map(|_|bf16::from_f32(f32::NAN))).to_hbm(&mut device.pdma).await.unwrap();
    let start=std::time::Instant::now();
    launch(ops_batch::project_full_output_batch,(device,&input,&weight,&scales,&mut output)).await.unwrap();
    let actual:HostTensor<bf16,m![B,H]>=output.to_host(&mut device.pdma).await.unwrap();
    let cold_ms=start.elapsed().as_secs_f64()*1000.0;
    let actual=actual.into_vec();
    let mut dots=vec![0f32;B::SIZE*17];
    for b in 0..B::SIZE { for n in 0..17 {
        let mut partials:Vec<f32>=(0..8).map(|chunk|bf16::from_f32((chunk*1024..(chunk+1)*1024).map(|k|x(b,k)*w(n,k)).sum()).to_f32()).collect();
        while partials.len()>1 { partials=partials.chunks_exact(2).map(|p|bf16::from_f32(p[0]+p[1]).to_f32()).collect(); }
        dots[b*17+n]=partials[0];
    }}
    let mut failures=0;
    for b in 0..B::SIZE { for n in 0..H::SIZE {
        let expected=bf16::from_f32(dots[b*17+n%17]*scale(n));
        let got=actual[b*H::SIZE+n];
        if got.to_bits()!=expected.to_bits() {
            if failures<10 {eprintln!("full row={b} col={n} got={} expected={}",got.to_f32(),expected.to_f32());}
            failures+=1;
        }
    }}
    let start=std::time::Instant::now();
    for _ in 0..5 { launch(ops_batch::project_full_output_batch,(device,&input,&weight,&scales,&mut output)).await.unwrap(); }
    let _:HostTensor<bf16,m![B,H]>=output.to_host(&mut device.pdma).await.unwrap();
    println!("full output B={} K={} H={} checked={} mismatches={failures} cold_ms={cold_ms:.3} warm5launches1read_ms={:.3}",B::SIZE,Qf::SIZE,H::SIZE,actual.len(),start.elapsed().as_secs_f64()*1000.0/5.0);
    assert_eq!(failures,0);
}
