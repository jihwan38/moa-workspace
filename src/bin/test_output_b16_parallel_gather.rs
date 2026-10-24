//! Independent dyadic/CPU partial-tree oracle and original B16 differential.
use furiosa_opt_gemma4::{
    Chip, axes::*, ops_batch as control, ops_output_b16_parallel_gather as op,
};
use furiosa_opt_std::prelude::*;
fn hash(i: usize, seed: usize) -> usize {
    let mut v = (i as u64).wrapping_add(seed as u64);
    v = (v ^ (v >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    v = (v ^ (v >> 27)).wrapping_mul(0x94d049bb133111eb);
    (v ^ (v >> 31)) as usize
}
fn x(b: usize, k: usize, seed: usize) -> f32 {
    if seed == 0 {
        return ((b * 17 + k * 13) % 31) as f32 / 16.0 - 15.0 / 16.0;
    }
    if b == 0 {
        return 0.0;
    }
    if b == 1 {
        return if k % 2 == 0 { 0.25 } else { -0.25 };
    }
    let v = hash(b * 8192 + k, seed);
    ((v % 255) as f32 - 127.0) / 128.0 * (1 + (v / 256) % 2) as f32
}
fn w(n: usize, k: usize, seed: usize) -> f32 {
    if seed == 0 {
        return ((n * 5 + k * 7) % 17) as f32 / 8.0 - 1.0;
    }
    // Equal adjacent weights force exact cancellation for input row1.
    let v = hash(n * 8192 + k / 2, seed + 7);
    ((v % 16) as f32 - 8.0) / 16.0 * (1 + (v / 16) % 2) as f32
}
fn scale(n: usize) -> f32 {
    [0.0, 0.5, -0.5, 1.0, -1.0, 1.5, 2.0][n % 7]
}
fn partial(b: usize, n: usize, chunk: usize, seed: usize) -> bf16 {
    // These domains keep every K1024 partial exact in F32 as well as F64.
    let dot = (chunk * 1024..(chunk + 1) * 1024)
        .map(|k| x(b, k, seed) as f64 * w(n, k, seed) as f64)
        .sum::<f64>();
    bf16::from_f32(dot as f32)
}
fn expected(b: usize, n: usize, chunks: usize, seed: usize) -> bf16 {
    let mut p: Vec<_> = (0..chunks)
        .map(|c| partial(b, n, c, seed).to_f32())
        .collect();
    while p.len() > 1 {
        p = p
            .chunks_exact(2)
            .map(|v| bf16::from_f32(v[0] + v[1]).to_f32())
            .collect();
    }
    bf16::from_f32(p[0] * scale(n))
}
async fn read<E: M>(d: &mut Device, x: &HbmTensor<bf16, Chip, E>) -> Vec<bf16> {
    let h: HostTensor<bf16, E> = x.to_host(&mut d.pdma).await.unwrap();
    h.into_vec()
}
macro_rules! probe {($name:ident,$k:ident,$shape:ty,$project:ident,$partials:ident)=>{
async fn $name(d:&mut Device) {
 for seed in [0,31,79] {
    let input=HostTensor::<bf16,$shape>::from_vec((0..B::SIZE).flat_map(|b|(0..$k::SIZE)
        .map(move|k|bf16::from_f32(x(b,k,seed))))).to_hbm(&mut d.pdma).await.unwrap();
    let weight=HostTensor::<f8e4m3,m![H,$k]>::from_vec((0..H::SIZE).flat_map(|n|(0..$k::SIZE)
        .map(move|k|f8e4m3::from_f32(w(n,k,seed))))).to_hbm(&mut d.pdma).await.unwrap();
    let scales=HostTensor::<bf16,m![H]>::from_vec((0..H::SIZE).map(|n|bf16::from_f32(scale(n))))
        .to_hbm(&mut d.pdma).await.unwrap();
    let mut out=HostTensor::<bf16,m![B,H]>::from_vec(vec![bf16::from_f32(f32::NAN);B::SIZE*H::SIZE])
        .to_hbm(&mut d.pdma).await.unwrap();
    launch(control::$project,(&mut *d,&input,&weight,&scales,&mut out)).await.unwrap();
    let reference=read(d,&out).await;
    launch(op::$project,(&mut *d,&input,&weight,&scales,&mut out)).await.unwrap();
    let actual=read(d,&out).await;
    let diff=reference.iter().zip(&actual).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();
    println!("{} seed={seed} original_B16 checked={} raw_differences={diff}",stringify!($name),actual.len());
    assert_eq!(diff,0);
    let mut p=HostTensor::<bf16,m![B,$k/1024,H]>::zero().to_hbm(&mut d.pdma).await.unwrap();
    launch(op::$partials,(&mut *d,&input,&weight,&mut p)).await.unwrap();
    let got=read(d,&p).await;
    let columns:Vec<_>=if seed==0 {(0..H::SIZE).collect()}else{vec![0,1,119,120,3838,3839]};
    let mut pdiff=0;let mut odiff=0;
    for b in 0..B::SIZE {
        // Dyadic columns repeat every17 before the separately verified channel scale.
        let cpu:Vec<Vec<bf16>>=(0..if seed==0 {17}else{columns.len()}).map(|j| {
            let n=if seed==0{j}else{columns[j]};
            (0..$k::SIZE/1024).map(|c|partial(b,n,c,seed)).collect()
        }).collect();
        for (j,&n) in columns.iter().enumerate() {
            for c in 0..$k::SIZE/1024 {
                pdiff+=usize::from(got[(b*($k::SIZE/1024)+c)*H::SIZE+n].to_bits()!=cpu[if seed==0{n%17}else{j}][c].to_bits());
            }
            let mut parts:Vec<_>=cpu[if seed==0{n%17}else{j}].iter().map(|x|x.to_f32()).collect();
            while parts.len()>1 {parts=parts.chunks_exact(2).map(|v|bf16::from_f32(v[0]+v[1]).to_f32()).collect();}
            let expected=bf16::from_f32(parts[0]*scale(n));
            odiff+=usize::from(actual[b*H::SIZE+n].to_bits()!=expected.to_bits());
        }
    }
    println!("{} seed={seed} cpu_columns={} partial_differences={pdiff} final_differences={odiff}",stringify!($name),columns.len());
    assert_eq!(pdiff,0);assert_eq!(odiff,0);
    if seed==0 {
        for variant in [0,1,1,0] {
            let now=std::time::Instant::now();
            for _ in 0..20 {
                if variant==0 {launch(control::$project,(&mut *d,&input,&weight,&scales,&mut out)).await.unwrap();}
                else {launch(op::$project,(&mut *d,&input,&weight,&scales,&mut out)).await.unwrap();}
            }
            println!("{} ABBA variant={variant} warm_us={:.3}",stringify!($name),now.elapsed().as_secs_f64()*1e6/20.0);
        }
    }
 }
}}}
probe!(
    sliding,
    Qs,
    m![B, Ns, Gs, Ds],
    project_sliding_output_batch,
    project_sliding_partial_dots
);
probe!(
    full,
    Qf,
    m![B, Gf, Df],
    project_full_output_batch,
    project_full_partial_dots
);
#[tokio::main]
async fn main() {
    let mut d = Device::new(op::project_sliding_output_batch.topology()).unwrap();
    sliding(&mut d).await;
    full(&mut d).await;
}
