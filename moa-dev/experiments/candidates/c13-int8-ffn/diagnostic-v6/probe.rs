use furiosa_opt_std::prelude::*;
use furiosa_opt_gemma4::{Chip, axes::*, integer_probe};

const FP4: [f32; 16] = [0., 0.5, 1., 1.5, 2., 3., 4., 6., -0., -0.5, -1., -1.5, -2., -3., -4., -6.];

fn input(case: usize, k: usize) -> bf16 {
    let value = match case {
        0 => (((k * 13 + 7) % 31) as f32 - 15.) * 2f32.powi((k / 16 % 9) as i32 - 8),
        1 => match k % 16 { 0 => 1., 1 => -1., 2 => 15. / 256., 3 => -14. / 256., _ => 2f32.powi(-22) },
        _ => 0.,
    };
    bf16::from_f32(value)
}

fn check_quant(name: &str, case: usize, x: &[bf16], actual: &[i8], scales: &[f32]) -> usize {
    let n = x.len();
    let mut failures = 0;
    for b in 0..n / 16 {
        let maximum = x[b*16..b*16+16].iter().map(|v| v.to_f32().abs()).fold(2f32.powi(-120), f32::max);
        let s = f32::from_bits(maximum.to_bits() & 0x7f80_0000) / 64.;
        if scales[b].to_bits() != (s / 256.).to_bits() {
            if failures < 8 { println!("C13_DIAG_SCALE {name} case={case} block={b} actual={} bits={:08x} expected={} bits={:08x}", scales[b], scales[b].to_bits(), s/256., (s/256.).to_bits()); }
            failures += 1;
        }
        for j in 0..16 {
            let k = b*16+j;
            let y = x[k].to_f32() / s;
            let hi = (y.round() as i32).clamp(-127,127);
            let lo = ((y-hi as f32)*128.).round() as i32;
            for (term, expected) in [(0,hi), (1,lo)] {
                if actual[term*n+k] as i32 != expected {
                    if failures < 8 { println!("C13_DIAG_DIGIT {name} case={case} term={term} k={k} x={} actual={} expected={expected}", x[k].to_f32(), actual[term*n+k]); }
                    failures += 1;
                }
            }
        }
    }
    println!("C13_DIAG_QUANT name={name} case={case} checked={} mismatches={failures}", 2*n+n/16);
    failures
}

#[tokio::main]
async fn main() {
    let mut d = Device::new(integer_probe::integer_blocks.topology()).unwrap();
    let mut failures = 0;
    let mut packed = vec![0u8; Rows::SIZE * K::SIZE / 2];
    let mut expected_decoded = vec![0i8; Rows::SIZE * K::SIZE];
    for r in 0..Rows::SIZE { for k in 0..K::SIZE {
        let c = (r*3+k) % 16;
        packed[(r*K::SIZE+k)/2] |= (c as u8) << (4*(k%2));
        expected_decoded[r*K::SIZE+k] = (2.*FP4[c]) as i8;
    }}
    let digits: Vec<i8> = (0..2*K::SIZE).map(|k| if k < K::SIZE { ((k*7%255) as i32-127) as i8 } else { ((k*11%129) as i32-64) as i8 }).collect();
    let w: HbmTensor<f4e2m1,Chip,m![Rows,K]> = HostTensor::<f4e2m1,m![Rows,K]>::from_buf(packed).to_hbm(&mut d.pdma).await.unwrap();
    let x: HbmTensor<i8,Chip,m![Term,K]> = HostTensor::<i8,m![Term,K]>::from_vec(digits.clone()).to_hbm(&mut d.pdma).await.unwrap();
    let mut decoded: HbmTensor<i8,Chip,m![Rows,K]> = HostTensor::<i8,m![Rows,K]>::from_vec(vec![99;Rows::SIZE*K::SIZE]).to_hbm(&mut d.pdma).await.unwrap();
    let mut sums: HbmTensor<f32,Chip,m![Rows,K/16]> = HostTensor::<f32,m![Rows,K/16]>::from_vec(vec![f32::NAN;Rows::SIZE*K::SIZE/16]).to_hbm(&mut d.pdma).await.unwrap();
    launch(integer_probe::integer_blocks, (&mut d,&w,&x,&mut decoded,&mut sums)).await.unwrap();
    let actual = decoded.to_host(&mut d.pdma).await.unwrap().into_vec();
    let mut decoded_bad=0;
    for (i,(a,e)) in actual.iter().zip(&expected_decoded).enumerate() { if a!=e {
        if decoded_bad<8 {println!("C13_DIAG_WEIGHT i={i} actual={a} expected={e}");} decoded_bad+=1;
    }}
    let actual = sums.to_host(&mut d.pdma).await.unwrap().into_vec();
    let mut dot_bad=0;
    for r in 0..Rows::SIZE { for b in 0..K::SIZE/16 {
        let expected: i32 = (b*16..b*16+16).map(|k| expected_decoded[r*K::SIZE+k] as i32 * (128*digits[k] as i32 + digits[K::SIZE+k] as i32)).sum();
        let a=actual[r*K::SIZE/16+b];
        if a != expected as f32 { if dot_bad<8 {println!("C13_DIAG_DOT r={r} b={b} actual={a} expected={expected}");} dot_bad+=1; }
    }}
    println!("C13_DIAG_INTEGER weight_mismatches={decoded_bad} dot_mismatches={dot_bad}");
    failures+=decoded_bad+dot_bad;
    for case in 0..3 {
        let values: Vec<bf16> = (0..H::SIZE).map(|k| input(case,k)).collect();
        let x: HbmTensor<bf16,Chip,m![H]> = HostTensor::<bf16,m![H]>::from_vec(values.clone()).to_hbm(&mut d.pdma).await.unwrap();
        let mut q: HbmTensor<i8,Chip,m![IntTerm,H]> = HostTensor::<i8,m![IntTerm,H]>::from_vec(vec![99;2*H::SIZE]).to_hbm(&mut d.pdma).await.unwrap();
        let mut s: HbmTensor<f32,Chip,m![H/16]> = HostTensor::<f32,m![H/16]>::from_vec(vec![f32::NAN;H::SIZE/16]).to_hbm(&mut d.pdma).await.unwrap();
        launch(integer_probe::quant_h, (&mut d,&x,&mut q,&mut s)).await.unwrap();
        let q=q.to_host(&mut d.pdma).await.unwrap().into_vec();
        let s=s.to_host(&mut d.pdma).await.unwrap().into_vec();
        failures+=check_quant("H",case,&values,&q,&s);

        let values: Vec<bf16> = (0..L::SIZE).map(|k| input(case,k)).collect();
        let x: HbmTensor<bf16,Chip,m![L]> = HostTensor::<bf16,m![L]>::from_vec(values.clone()).to_hbm(&mut d.pdma).await.unwrap();
        let mut q: HbmTensor<i8,Chip,m![IntTerm,L]> = HostTensor::<i8,m![IntTerm,L]>::from_vec(vec![99;2*L::SIZE]).to_hbm(&mut d.pdma).await.unwrap();
        let mut s: HbmTensor<f32,Chip,m![L/16]> = HostTensor::<f32,m![L/16]>::from_vec(vec![f32::NAN;L::SIZE/16]).to_hbm(&mut d.pdma).await.unwrap();
        launch(integer_probe::quant_l, (&mut d,&x,&mut q,&mut s)).await.unwrap();
        let q=q.to_host(&mut d.pdma).await.unwrap().into_vec();
        let s=s.to_host(&mut d.pdma).await.unwrap().into_vec();
        failures+=check_quant("L",case,&values,&q,&s);
    }
    println!("C13_DIAG_COMPLETE mismatches={failures}");
    if failures!=0 {std::process::exit(1);}
}
