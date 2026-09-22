use furiosa_opt_std::prelude::*;
use furiosa_opt_gemma4::{Chip, scale_probe::{self, H, L, Trace}};
use std::io::Write;

fn partial(case: usize, row: usize, block: usize) -> f32 {
    match case {
        0 => ((17 * row + 3 * block) % 127) as f32 - 63.0,
        1 => {
            let edge = [0, 15, 16, 119, 120, 239, 240, 479, 480, 839, 840, 959][row % 12];
            if block == edge { 1.0 + (row % 16) as f32 } else { 0.0 }
        }
        2 => if block % 2 == 0 { 1.0 + 15.0 / 256.0 } else { -1.0 - 14.0 / 256.0 },
        _ => unreachable!(),
    }
}
fn scale(case: usize, row: usize, block: usize) -> f32 {
    if case == 2 { 1.0 } else { [0.25, 0.5, 1.0, 2.0][(row * 5 + block * 3) % 4] }
}
fn rounded(p: f32) -> bf16 {
    let bits = p.to_bits();
    bf16::from_bits(((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16)
}
fn check(name: &str, case: usize, actual: &[f32], expected: &[f32]) -> usize {
    assert_eq!(actual.len(), expected.len());
    let mut bad = 0; let mut max = 0.0f32; let mut prefix_bad=0; let mut cleared_bad=0;
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        if a == e { continue; }
        if bad < 8 { println!("MISMATCH stage={name} case={case} index={i} actual={a} expected={e}"); }
        bad += 1; max = max.max((a-e).abs()); if name.ends_with("128") {if i%128<112 {prefix_bad+=1;} else {cleared_bad+=1;}}
    }
    println!("D01 stage={name} case={case} checked={} mismatches={bad} max_abs={max} prefix_mismatches={prefix_bad} cleared_mismatches={cleared_bad} status={}",actual.len(),if bad==0 {"PASS"} else {"FAIL"});
    bad
}
#[tokio::main]
async fn main() {
    let oracle_only = std::env::args().any(|a| a == "--oracle-only");
    let mut device = if oracle_only { None } else { Some(Device::new(scale_probe::scale_probe.topology()).expect("device")) };
    let mut failures = 0;
    for case in 0..3 {
        let mut p = Vec::with_capacity(H::SIZE * L::SIZE / 16);
        let mut s = Vec::with_capacity(p.capacity());
        let mut local = vec![0.0f32; H::SIZE * 8];
        let mut sum = vec![0.0f32; H::SIZE];
        let mut unrounded = vec![0.0f32; H::SIZE];
        for row in 0..H::SIZE { for block in 0..L::SIZE / 16 {
            let v=partial(case,row,block); let sv=scale(case,row,block);
            assert_eq!(rounded(v).to_bits(),bf16::from_f32(v).to_bits());
            let sf=f8e4m3::from_f32(sv); assert_eq!(sf.to_f32(),sv);
            p.push(v); s.push(sf);
            if block%120<112 { local[row*8+block/120]+=rounded(v).to_f32()*sv; }
            if block%120<112 { unrounded[row]+=v*sv; }
        }}
        for row in 0..H::SIZE { sum[row]=local[row*8..row*8+8].iter().sum(); }
        let mut ep = Vec::with_capacity(Trace::SIZE); let mut es = Vec::with_capacity(Trace::SIZE);
        for c in 0..2 { for g in 0..32 { for col in 0..8 { for r in 0..60 { for b in 0..128 {
            let row=c*1920+g*60+r; let block=col*120+b;
            ep.push(if b<112 {rounded(partial(case,row,block)).to_f32()} else {0.0});
            es.push(if b<112 {scale(case,row,block)} else {0.0});
        }}}}}
        let differences=sum.iter().zip(&unrounded).filter(|(a,b)|a!=b).count();
        let max_error=sum.iter().zip(&unrounded).map(|(a,b)|(a-b).abs()).fold(0.0f32,f32::max);
        assert_eq!(differences, if case==2 {H::SIZE} else {0});
        println!("ROUNDING_MODEL case={case} changed_rows={differences} max_abs={max_error} rounded_first={} original_first={}",sum[0],unrounded[0]);
        if oracle_only {continue;}
        let d=device.as_mut().unwrap();
        let ph:HbmTensor<f32,Chip,m![H,L/16]>=HostTensor::<f32,m![H,L/16]>::from_vec(p).to_hbm(&mut d.pdma).await.unwrap();
        let sh:HbmTensor<f8e4m3,Chip,m![H,L/16]>=HostTensor::<f8e4m3,m![H,L/16]>::from_vec(s).to_hbm(&mut d.pdma).await.unwrap();
        let mut pt:HbmTensor<bf16,Chip,Trace>=HostTensor::<bf16,Trace>::from_vec(vec![bf16::from_bits(0x7fc1);Trace::SIZE]).to_hbm(&mut d.pdma).await.unwrap();
        let mut st:HbmTensor<bf16,Chip,Trace>=HostTensor::<bf16,Trace>::from_vec(vec![bf16::from_bits(0x7fc1);Trace::SIZE]).to_hbm(&mut d.pdma).await.unwrap();
        let mut lt:HbmTensor<f32,Chip,m![H,L/1920]>=HostTensor::<f32,m![H,L/1920]>::from_vec(vec![f32::NAN;H::SIZE*8]).to_hbm(&mut d.pdma).await.unwrap();
        let mut gt:HbmTensor<f32,Chip,m![H]>=HostTensor::<f32,m![H]>::from_vec(vec![f32::NAN;H::SIZE]).to_hbm(&mut d.pdma).await.unwrap();
        println!("D01 case={case} LAUNCH_BEGIN"); std::io::stdout().flush().unwrap();
        launch(scale_probe::scale_probe,(&mut *d,&ph,&sh,&mut pt,&mut st,&mut lt,&mut gt)).await.expect("scale probe");
        let ap:HostTensor<bf16,Trace>=pt.to_host(&mut d.pdma).await.unwrap();
        let asc:HostTensor<bf16,Trace>=st.to_host(&mut d.pdma).await.unwrap();
        let al:HostTensor<f32,m![H,L/1920]>=lt.to_host(&mut d.pdma).await.unwrap();
        let ag:HostTensor<f32,m![H]>=gt.to_host(&mut d.pdma).await.unwrap();
        failures+=check("partial128",case,&ap.into_vec().into_iter().map(bf16::to_f32).collect::<Vec<_>>(),&ep);
        failures+=check("scale128",case,&asc.into_vec().into_iter().map(bf16::to_f32).collect::<Vec<_>>(),&es);
        failures+=check("local_dot",case,&al.into_vec(),&local);
        failures+=check("column_sum",case,&ag.into_vec(),&sum);
    }
    println!("D01_COMPLETE oracle_only={oracle_only} mismatches={failures} status={}",if failures==0 {"PASS"} else {"FAIL"});
    if failures!=0 { std::process::exit(1); }
}
