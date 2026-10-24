//! Independent and differential checks for P128 H960-partitioned query.
//! Dyadic inputs make the F32 dot exact; the check includes both BF16 boundaries.
use furiosa_opt_gemma4::{
    Chip,
    axes::{H, P128, Qs},
    ops_batch_wide, ops_batch_kpart,
};
use furiosa_opt_std::prelude::*;

fn input(b: usize, k: usize) -> f32 {
    ((b * 17 + k * 13) % 31) as f32 / 16.0 - 15.0 / 16.0
}
fn weight(n: usize, k: usize) -> f32 {
    ((n * 5 + k * 7) % 17) as f32 / 8.0 - 1.0
}
fn scale(n: usize) -> f32 {
    2f32.powi((n % 7) as i32 - 3)
}

#[tokio::main]
async fn main() {
    let mut device = Device::new(ops_batch_kpart::project_query_wide.topology()).unwrap();
    let x: HbmTensor<bf16, Chip, m![P128, H]> = HostTensor::<bf16, m![P128, H]>::from_vec(
        (0..P128::SIZE).flat_map(|b| (0..H::SIZE).map(move |k| bf16::from_f32(input(b, k)))),
    )
    .to_hbm(&mut device.pdma)
    .await
    .unwrap();
    let w: HbmTensor<f8e4m3, Chip, m![Qs, H]> = HostTensor::<f8e4m3, m![Qs, H]>::from_vec(
        (0..Qs::SIZE).flat_map(|n| (0..H::SIZE).map(move |k| f8e4m3::from_f32(weight(n, k)))),
    )
    .to_hbm(&mut device.pdma)
    .await
    .unwrap();
    let s: HbmTensor<bf16, Chip, m![Qs]> =
        HostTensor::<bf16, m![Qs]>::from_vec((0..Qs::SIZE).map(|n| bf16::from_f32(scale(n))))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
    let mut out: HbmTensor<bf16, Chip, m![P128, Qs]> = HostTensor::<bf16, m![P128, Qs]>::from_vec(
        (0..Qs::SIZE * P128::SIZE).map(|_| bf16::from_f32(f32::NAN)),
    )
    .to_hbm(&mut device.pdma)
    .await
    .unwrap();
    let start = std::time::Instant::now();
    launch(
        ops_batch_wide::project_query_batch,
        (&mut device, &x, &w, &s, &mut out),
    )
    .await
    .unwrap();
    let host: HostTensor<bf16, m![P128, Qs]> = out.to_host(&mut device.pdma).await.unwrap();
    let elapsed = start.elapsed();
    let actual = host.into_vec();
    let mut dot = vec![0.0f32; P128::SIZE * 17];
    for b in 0..P128::SIZE {
        for n in 0..17 {
            dot[b * 17 + n] =
                bf16::from_f32((0..H::SIZE).map(|k| input(b, k) * weight(n, k)).sum()).to_f32();
        }
    }
    let mut failures = 0;
    for b in 0..P128::SIZE {
        for n in 0..Qs::SIZE {
            let expected = bf16::from_f32(dot[b * 17 + n % 17] * scale(n));
            let got = actual[b * Qs::SIZE + n];
            if got.to_bits() != expected.to_bits() {
                if failures < 10 {
                    eprintln!(
                        "row={b} col={n} got={} expected={}",
                        got.to_f32(),
                        expected.to_f32()
                    );
                }
                failures += 1;
            }
        }
    }
    println!(
        "projection P128={} H={} Qs={} checked={} mismatches={} launch_and_download_ms={:.3}",
        P128::SIZE,
        H::SIZE,
        Qs::SIZE,
        actual.len(),
        failures,
        elapsed.as_secs_f64() * 1000.0
    );
    let mut times=[Vec::<f64>::new(),Vec::<f64>::new()];
    for variant in [0,1,1,0] {
        let start=std::time::Instant::now();
        for _ in 0..5 {
            if variant==0 {launch(ops_batch_wide::project_query_batch,(&mut device,&x,&w,&s,&mut out)).await.unwrap();}
            else {launch(ops_batch_kpart::project_query_wide,(&mut device,&x,&w,&s,&mut out)).await.unwrap();}
        }
        let _:HostTensor<bf16,m![P128,Qs]>=out.to_host(&mut device.pdma).await.unwrap();
        times[variant].push(start.elapsed().as_secs_f64()*1000.0/5.0);
    }
    println!("P128 H960 query exact_dyadic_failures={failures} fullH_ms={:?} H960_ms={:?} scope=5launches+1read_divided_by5",times[0],times[1]);
    random_probe(&mut device).await;
    assert_eq!(failures,0);
}

fn mix(i:usize,seed:u64)->u64 {
    let mut x=(i as u64).wrapping_add(seed);
    x=(x^(x>>30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x=(x^(x>>27)).wrapping_mul(0x94d049bb133111eb);
    x^(x>>31)
}
fn varied(i:usize,seed:u64,top:f32)->f32 {
    let bits=mix(i,seed);
    let signed=((bits>>40)as f32/16777216.0)*2.0-1.0;
    signed*top*2f32.powi(-((bits%9)as i32))
}
async fn random_probe(device:&mut Device) {
    let x:HbmTensor<bf16,Chip,m![P128,H]>=HostTensor::<bf16,m![P128,H]>::from_vec((0..P128::SIZE*H::SIZE).map(|i|bf16::from_f32(varied(i,53,4.0)))).to_hbm(&mut device.pdma).await.unwrap();
    let w:HbmTensor<f8e4m3,Chip,m![Qs,H]>=HostTensor::<f8e4m3,m![Qs,H]>::from_vec((0..Qs::SIZE*H::SIZE).map(|i|f8e4m3::from_f32(varied(i,59,1.0)))).to_hbm(&mut device.pdma).await.unwrap();
    let s:HbmTensor<bf16,Chip,m![Qs]>=HostTensor::<bf16,m![Qs]>::from_vec((0..Qs::SIZE).map(|i|bf16::from_f32(0.25+(mix(i,61)%31)as f32/32.0))).to_hbm(&mut device.pdma).await.unwrap();
    let mut out:HbmTensor<bf16,Chip,m![P128,Qs]>=HostTensor::<bf16,m![P128,Qs]>::from_vec((0..P128::SIZE*Qs::SIZE).map(|_|bf16::from_f32(0.0))).to_hbm(&mut device.pdma).await.unwrap();
    launch(ops_batch_wide::project_query_batch,(device,&x,&w,&s,&mut out)).await.unwrap();
    let original:HostTensor<bf16,m![P128,Qs]>=out.to_host(&mut device.pdma).await.unwrap();
    launch(ops_batch_kpart::project_query_wide,(device,&x,&w,&s,&mut out)).await.unwrap();
    let changed:HostTensor<bf16,m![P128,Qs]>=out.to_host(&mut device.pdma).await.unwrap();
    let mut bitdiff=0;let mut failures=0;let mut max_abs=0f32;let mut max_rel=0f32;
    for (i,(expected,got)) in original.into_vec().iter().zip(changed.into_vec()).enumerate() {
        if expected.to_bits()!=got.to_bits(){bitdiff+=1;}
        let e=expected.to_f32();let a=got.to_f32();let err=(e-a).abs();
        max_abs=max_abs.max(err);max_rel=max_rel.max(err/e.abs().max(0.001));
        if !a.is_finite() || err>0.001+0.01*e.abs(){if failures<8{eprintln!("random i={i} got={a} expected={e}");}failures+=1;}
    }
    println!("P128H960 random checked={} bit_differences={bitdiff} tolerance_failures={failures} max_abs={max_abs} max_rel={max_rel}",P128::SIZE*Qs::SIZE);
    assert_eq!(failures,0);
}
