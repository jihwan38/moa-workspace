//! Independent dyadic oracle for batched attention output projection and BF16 sum trees.
use furiosa_opt_std::prelude::*;
use furiosa_opt_gemma4::{axes::*,Chip,ops_batch_r64,ops_output_r64_direct,ops_output_r64_direct16};
fn x(b:usize,k:usize)->f32 { ((b*17+k*13)%31) as f32/16.0-15.0/16.0 }
fn varied(b:usize,k:usize)->f32 { let mut v=(b*8192+k)as u64+31;v=(v^(v>>30)).wrapping_mul(0xbf58476d1ce4e5b9);v=(v^(v>>27)).wrapping_mul(0x94d049bb133111eb);v^=v>>31; (((v>>40)as f32/16777216.0)*2.0-1.0)*2f32.powi(-((v%8)as i32)) }
fn w(n:usize,k:usize)->f32 { ((n*5+k*7)%17) as f32/8.0-1.0 }
fn scale(n:usize)->f32 { 1.0+(n%7) as f32/8.0 }
#[tokio::main]
async fn main() {
    let mut device=Device::new(ops_batch_r64::project_sliding_output_batch.topology()).unwrap();
    sliding(&mut device,false).await;
    sliding(&mut device,true).await;
    full(&mut device,false).await;
    full(&mut device,true).await;
}

async fn sliding(device:&mut Device,random:bool) {
    let input:HbmTensor<bf16,Chip,m![R64,Ns,Gs,Ds]>=HostTensor::<bf16,m![R64,Ns,Gs,Ds]>::from_vec((0..R64::SIZE).flat_map(|b|(0..Qs::SIZE).map(move|k|bf16::from_f32(if random {varied(b,k)} else {x(b,k)})))).to_hbm(&mut device.pdma).await.unwrap();
    let weight:HbmTensor<f8e4m3,Chip,m![H,Qs]>=HostTensor::<f8e4m3,m![H,Qs]>::from_vec((0..H::SIZE).flat_map(|n|(0..Qs::SIZE).map(move|k|f8e4m3::from_f32(w(n,k))))).to_hbm(&mut device.pdma).await.unwrap();
    let scales:HbmTensor<bf16,Chip,m![H]>=HostTensor::<bf16,m![H]>::from_vec((0..H::SIZE).map(|n|bf16::from_f32(scale(n)))).to_hbm(&mut device.pdma).await.unwrap();
    let mut output:HbmTensor<bf16,Chip,m![R64,H]>=HostTensor::<bf16,m![R64,H]>::from_vec((0..R64::SIZE*H::SIZE).map(|_|bf16::from_f32(f32::NAN))).to_hbm(&mut device.pdma).await.unwrap();
    let start=std::time::Instant::now();
    launch(ops_batch_r64::project_sliding_output_batch,(device,&input,&weight,&scales,&mut output)).await.unwrap();
    let actual:HostTensor<bf16,m![R64,H]>=output.to_host(&mut device.pdma).await.unwrap();
    let cold_ms=start.elapsed().as_secs_f64()*1000.0;
    let actual=actual.into_vec();
    launch(ops_output_r64_direct16::project_sliding_output_batch,(device,&input,&weight,&scales,&mut output)).await.unwrap();
    let direct:HostTensor<bf16,m![R64,H]>=output.to_host(&mut device.pdma).await.unwrap();
    let direct=direct.into_vec();
    let differences=actual.iter().zip(&direct).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();
    println!("sliding direct_vs_control random={random} checked={} bit_differences={differences}",direct.len());
    assert_eq!(differences,0);
    let mut dots=vec![0f32;R64::SIZE*17];
    for b in 0..R64::SIZE { for n in 0..17 {
        let mut partials:Vec<f32>=(0..4).map(|chunk|bf16::from_f32((chunk*1024..(chunk+1)*1024).map(|k|x(b,k)*w(n,k)).sum()).to_f32()).collect();
        while partials.len()>1 { partials=partials.chunks_exact(2).map(|p|bf16::from_f32(p[0]+p[1]).to_f32()).collect(); }
        dots[b*17+n]=partials[0];
    }}
    let mut failures=0;
    for b in 0..R64::SIZE { for n in 0..H::SIZE {
        let expected=bf16::from_f32(dots[b*17+n%17]*scale(n));
        let got=actual[b*H::SIZE+n];
        if !random && got.to_bits()!=expected.to_bits() {
            if failures<10 {eprintln!("sliding row={b} col={n} got={} expected={}",got.to_f32(),expected.to_f32());}
            failures+=1;
        }
    }}
    if !random {
        let mut times=[Vec::<f64>::new(),Vec::<f64>::new(),Vec::<f64>::new()];
        for variant in [0,1,2,2,1,0] {
            let start=std::time::Instant::now();
            for _ in 0..5 {
                if variant==0 {launch(ops_batch_r64::project_sliding_output_batch,(device,&input,&weight,&scales,&mut output)).await.unwrap();}
                else if variant==1 {launch(ops_output_r64_direct::project_sliding_output_batch,(device,&input,&weight,&scales,&mut output)).await.unwrap();}
                else {launch(ops_output_r64_direct16::project_sliding_output_batch,(device,&input,&weight,&scales,&mut output)).await.unwrap();}
            }
            let _:HostTensor<bf16,m![R64,H]>=output.to_host(&mut device.pdma).await.unwrap();
            times[variant].push(start.elapsed().as_secs_f64()*1000.0/5.0);
        }
        println!("sliding R64 independent_cpu_checked={} mismatches={failures} cold_ms={cold_ms:.3} control_ms={:?} direct8_ms={:?} direct16_ms={:?} scope=5launches+1matchedread_divided5",actual.len(),times[0],times[1],times[2]);
    }
    assert_eq!(failures,0);
}

async fn full(device:&mut Device,random:bool) {
    let input:HbmTensor<bf16,Chip,m![R64,Gf,Df]>=HostTensor::<bf16,m![R64,Gf,Df]>::from_vec((0..R64::SIZE).flat_map(|b|(0..Qf::SIZE).map(move|k|bf16::from_f32(if random {varied(b,k)} else {x(b,k)})))).to_hbm(&mut device.pdma).await.unwrap();
    let weight:HbmTensor<f8e4m3,Chip,m![H,Qf]>=HostTensor::<f8e4m3,m![H,Qf]>::from_vec((0..H::SIZE).flat_map(|n|(0..Qf::SIZE).map(move|k|f8e4m3::from_f32(w(n,k))))).to_hbm(&mut device.pdma).await.unwrap();
    let scales:HbmTensor<bf16,Chip,m![H]>=HostTensor::<bf16,m![H]>::from_vec((0..H::SIZE).map(|n|bf16::from_f32(scale(n)))).to_hbm(&mut device.pdma).await.unwrap();
    let mut output:HbmTensor<bf16,Chip,m![R64,H]>=HostTensor::<bf16,m![R64,H]>::from_vec((0..R64::SIZE*H::SIZE).map(|_|bf16::from_f32(f32::NAN))).to_hbm(&mut device.pdma).await.unwrap();
    let start=std::time::Instant::now();
    launch(ops_batch_r64::project_full_output_batch,(device,&input,&weight,&scales,&mut output)).await.unwrap();
    let actual:HostTensor<bf16,m![R64,H]>=output.to_host(&mut device.pdma).await.unwrap();
    let cold_ms=start.elapsed().as_secs_f64()*1000.0;
    let actual=actual.into_vec();
    launch(ops_output_r64_direct16::project_full_output_batch,(device,&input,&weight,&scales,&mut output)).await.unwrap();
    let direct:HostTensor<bf16,m![R64,H]>=output.to_host(&mut device.pdma).await.unwrap();
    let direct=direct.into_vec();
    let differences=actual.iter().zip(&direct).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();
    println!("full direct_vs_control random={random} checked={} bit_differences={differences}",direct.len());
    assert_eq!(differences,0);
    let mut dots=vec![0f32;R64::SIZE*17];
    for b in 0..R64::SIZE { for n in 0..17 {
        let mut partials:Vec<f32>=(0..8).map(|chunk|bf16::from_f32((chunk*1024..(chunk+1)*1024).map(|k|x(b,k)*w(n,k)).sum()).to_f32()).collect();
        while partials.len()>1 { partials=partials.chunks_exact(2).map(|p|bf16::from_f32(p[0]+p[1]).to_f32()).collect(); }
        dots[b*17+n]=partials[0];
    }}
    let mut failures=0;
    for b in 0..R64::SIZE { for n in 0..H::SIZE {
        let expected=bf16::from_f32(dots[b*17+n%17]*scale(n));
        let got=actual[b*H::SIZE+n];
        if !random && got.to_bits()!=expected.to_bits() {
            if failures<10 {eprintln!("full row={b} col={n} got={} expected={}",got.to_f32(),expected.to_f32());}
            failures+=1;
        }
    }}
    if !random {
        let mut times=[Vec::<f64>::new(),Vec::<f64>::new(),Vec::<f64>::new()];
        for variant in [0,1,2,2,1,0] {
            let start=std::time::Instant::now();
            for _ in 0..5 {
                if variant==0 {launch(ops_batch_r64::project_full_output_batch,(device,&input,&weight,&scales,&mut output)).await.unwrap();}
                else if variant==1 {launch(ops_output_r64_direct::project_full_output_batch,(device,&input,&weight,&scales,&mut output)).await.unwrap();}
                else {launch(ops_output_r64_direct16::project_full_output_batch,(device,&input,&weight,&scales,&mut output)).await.unwrap();}
            }
            let _:HostTensor<bf16,m![R64,H]>=output.to_host(&mut device.pdma).await.unwrap();
            times[variant].push(start.elapsed().as_secs_f64()*1000.0/5.0);
        }
        println!("full R64 independent_cpu_checked={} mismatches={failures} cold_ms={cold_ms:.3} control_ms={:?} direct8_ms={:?} direct16_ms={:?} scope=5launches+1matchedread_divided5",actual.len(),times[0],times[1],times[2]);
    }
    assert_eq!(failures,0);
}
