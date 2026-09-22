use furiosa_opt_std::prelude::*;
use furiosa_opt_gemma4::{Chip, packet_probe::{self, H, R}};

fn inputs(case: usize) -> (Vec<bf16>, Vec<bf16>, Vec<f8e4m3>, Vec<f32>) {
    let x: Vec<bf16> = (0..H::SIZE).map(|h| bf16::from_f32(match case {
        2 => match (h / 16 % 2, h % 16) {
            (0, 0) => 1.0, (0, 1) => 15.0 / 256.0,
            (1, 0) => -1.0, (1, 1) => -14.0 / 256.0, _ => 0.0,
        },
        3 => 0.0,
        _ => ((h * 3 % 17) as f32 - 8.0) / 8.0,
    })).collect();
    let fp4 = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let w: Vec<bf16> = (0..R::SIZE).flat_map(|r| (0..H::SIZE).map(move |h| bf16::from_f32(match case {
        1 => if h / 16 == [0, 1, 238, 239][r] {1.0} else {0.0},
        2 => [0.5, 1.0, 2.0, 4.0][r],
        _ => fp4[(r * 7 + h * 3 + h / 16) % fp4.len()],
    }))).collect();
    let s: Vec<f8e4m3> = (0..R::SIZE).flat_map(|r| (0..H::SIZE / 16).map(move |b|
        f8e4m3::from_f32(if case == 2 {1.0} else {[-2.0, -0.5, 0.25, 0.5, 1.0, 2.0][(r * 3 + b) % 6]}))).collect();
    let mut expected = vec![0.0; R::SIZE];
    for r in 0..R::SIZE {
        let mut exact = 0.0f64;
        for b in 0..H::SIZE / 16 {
            let mut dot = 0.0f32;
            for j in 0..16 { dot += x[b * 16 + j].to_f32() * w[r * H::SIZE + b * 16 + j].to_f32(); }
            let product = dot * s[r * (H::SIZE / 16) + b].to_f32();
            expected[r] += product;
            exact += product as f64;
        }
        assert_eq!(expected[r] as f64, exact);
    }
    if case == 2 { assert_eq!(expected, vec![0.234375, 0.46875, 0.9375, 1.875]); }
    (x, w, s, expected)
}

#[tokio::main]
async fn main() {
    let oracle_only = std::env::args().any(|a| a == "--oracle-only");
    let mut device = if oracle_only {None} else {Some(Device::new(packet_probe::packet_probe.topology()).unwrap())};
    let mut errors = 0;
    for case in 0..4 {
        let (x, w, s, expected) = inputs(case);
        println!("C02_ORACLE case={case} expected={expected:?}");
        if oracle_only {continue;}
        let d = device.as_mut().unwrap();
        let expected_scales: Vec<f32> = s.iter().map(|v| v.to_f32()).collect();
        let x: HbmTensor<bf16, Chip, m![H]> = HostTensor::<bf16,m![H]>::from_vec(x).to_hbm(&mut d.pdma).await.unwrap();
        let w: HbmTensor<bf16, Chip, m![R,H]> = HostTensor::<bf16,m![R,H]>::from_vec(w).to_hbm(&mut d.pdma).await.unwrap();
        let s: HbmTensor<f8e4m3, Chip, m![R,H/16]> = HostTensor::<f8e4m3,m![R,H/16]>::from_vec(s).to_hbm(&mut d.pdma).await.unwrap();
        let mut trace: HbmTensor<f32, Chip, m![R,H/16]> = HostTensor::<f32,m![R,H/16]>::from_vec(vec![f32::NAN;R::SIZE*H::SIZE/16]).to_hbm(&mut d.pdma).await.unwrap();
        let mut y: HbmTensor<f32, Chip, m![R]> = HostTensor::<f32,m![R]>::from_vec(vec![f32::NAN;R::SIZE]).to_hbm(&mut d.pdma).await.unwrap();
        launch(packet_probe::packet_probe, (&mut *d,&x,&w,&s,&mut trace,&mut y)).await.unwrap();
        let scales: HostTensor<f32,m![R,H/16]> = trace.to_host(&mut d.pdma).await.unwrap();
        let actual: HostTensor<f32,m![R]> = y.to_host(&mut d.pdma).await.unwrap();
        let scales = scales.into_vec(); let actual = actual.into_vec();
        let scale_errors = scales.iter().zip(&expected_scales).filter(|(a,b)|a!=b).count();
        let output_errors = actual.iter().zip(&expected).filter(|(a,b)|a!=b).count();
        errors += scale_errors + output_errors;
        println!("C02_RESULT case={case} scale_checked={} scale_errors={scale_errors} output_checked={} output_errors={output_errors} actual={actual:?}",scales.len(),actual.len());
    }
    println!("C02_COMPLETE oracle_only={oracle_only} errors={errors}");
    if errors != 0 {std::process::exit(1);}
}
