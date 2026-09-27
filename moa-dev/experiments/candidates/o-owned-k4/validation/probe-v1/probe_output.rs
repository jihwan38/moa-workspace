//! Calls the frozen E132 project_output; checks its pre-RMS BF16 boundary.
use furiosa_opt_std::prelude::*;
use furiosa_opt_gemma4::{axes::*, Chip, probe};
use std::io::Write;
const NAMES: [&str;8] = ["signed_zero","isolated_k1023","isolated_k1024","isolated_k4095","multi_k_boundaries","stage_row_boundaries","tiny_correlated_cancellation","scale_before_bf16"];
fn checksum(bytes: &[u8]) -> u32 {
    assert_eq!(bytes.len()%4,0);
    bytes.chunks_exact(4).enumerate().fold(0u32,|acc,(i,p)|acc.wrapping_add(u32::from_le_bytes(p.try_into().unwrap()).wrapping_mul((2*i+1) as u32)))
}
fn weights(case: usize) -> Vec<u8> {
    let mut w=Vec::with_capacity(H::SIZE*Qs::SIZE);
    for r in 0..H::SIZE { for k in 0..Qs::SIZE {
        let mut code=(0x38+((r*3+r/16+r/30+r/1920+k+k/1024)%16)) | (((r+k/1024)%2)<<7);
        if case==5 && ![0,14,15,16,17,28,29].contains(&(r%30)) {code=0;}
        if case==6 {code=if k%1024<2 {0x38} else {0x77};}
        if case==7 {code=0x38+8*(r%2);}
        w.push(code as u8);
    }} w
}
fn take<'a>(data: &'a [u8], cursor: &mut usize, n: usize) -> &'a [u8] {let result=&data[*cursor..*cursor+n]; *cursor+=n; result}
#[tokio::main]
async fn main() {
    let args:Vec<_>=std::env::args().collect();
    let host_only=args.iter().any(|s|s=="--host-check-only");
    let input=args.windows(2).find(|a|a[0]=="--fixture").map(|a|a[1].as_str()).unwrap_or("probe-fixture.bin");
    let data=std::fs::read(input).unwrap(); assert_eq!(&data[..8],b"E132PR01"); let mut cursor=8;
    let mut device=if host_only {None} else {Some(Device::new(probe::output_before_rms.topology()).unwrap())};
    let mut total=0;let mut completed=0;
    println!("PROBE_BEGIN cases=8 values_per_case=3840 pre_rms=true bitwise_bf16=true signed_zeros_equivalent=true host_only={host_only} performance=false");
    for case in 0..8 {
        let checks:Vec<u32>=take(&data,&mut cursor,12).chunks_exact(4).map(|p|u32::from_le_bytes(p.try_into().unwrap())).collect();
        let x=take(&data,&mut cursor,2*Qs::SIZE).to_vec();let scale=take(&data,&mut cursor,2*H::SIZE).to_vec();
        let want:Vec<u16>=take(&data,&mut cursor,2*H::SIZE).chunks_exact(2).map(|p|u16::from_le_bytes(p.try_into().unwrap())).collect();
        let w=weights(case);
        for (i,(name,bytes)) in [("x",&x),("scale",&scale),("weight",&w)].into_iter().enumerate() {
            let got=checksum(bytes);assert_eq!(got,checks[i],"case {case} {name} input checksum");
            println!("PROBE_INPUT case={case} name={name} bytes={} checksum={got:08x} status=PASS",bytes.len());
        }
        if host_only {completed+=1;continue;}
        let device=device.as_mut().unwrap();
        let x:HbmTensor<bf16,Chip,m![Ns,Gs,Ds]>=HostTensor::<bf16,m![Ns,Gs,Ds]>::from_buf(x).to_hbm(&mut device.pdma).await.unwrap();
        let scale:HbmTensor<bf16,Chip,m![H]>=HostTensor::<bf16,m![H]>::from_buf(scale).to_hbm(&mut device.pdma).await.unwrap();
        let w:HbmTensor<f8e4m3,Chip,m![H,Qs]>=HostTensor::<f8e4m3,m![H,Qs]>::from_buf(w).to_hbm(&mut device.pdma).await.unwrap();
        // NaN sentinel also catches an unwritten output boundary.
        let mut output:HbmTensor<bf16,Chip,m![H]>=HostTensor::<bf16,m![H]>::from_buf([0xc1u8,0x7f].repeat(H::SIZE)).to_hbm(&mut device.pdma).await.unwrap();
        println!("PROBE_LAUNCH case={case} name={}",NAMES[case]);std::io::stdout().flush().unwrap();
        launch(probe::output_before_rms,(&mut *device,&x,&w,&scale,&mut output)).await.expect("pre RMS projection launch");
        let got:HostTensor<bf16,m![H]>=output.to_host(&mut device.pdma).await.unwrap();let got=got.into_vec();
        let mut failed=0;let mut nonfinite=0;let mut groups=[0usize;128];
        for r in 0..H::SIZE {let bits=got[r].to_bits();
            let equal=bits==want[r] || ((bits&0x7fff)==0 && (want[r]&0x7fff)==0);
            if !got[r].to_f32().is_finite() {nonfinite+=1;}
            if !equal {if failed<24 {println!("PROBE_MISMATCH case={case} row={r} stage_row={} cluster={} got={bits:04x} want={:04x}",r%30,r/1920,want[r]);}failed+=1;groups[r/30]+=1;}
        }
        for r in [0,14,15,16,17,28,29,30,1918,1919,1920,1921,3839] {println!("PROBE_SAMPLE case={case} row={r} got={:04x} want={:04x}",got[r].to_bits(),want[r]);}
        println!("PROBE_GROUP_FAILURES case={case} counts={groups:?}");
        println!("PROBE_RESULT case={case} name={} checked={} failures={failed} nonfinite={nonfinite} status={}",NAMES[case],H::SIZE,if failed==0 && nonfinite==0 {"PASS"} else {"FAIL"});
        total+=failed+nonfinite;completed+=1;
    }
    assert_eq!(cursor,data.len());
    println!("PROBE_COMPLETE completed={completed} planned=8 failed_values={total} host_only={host_only} status={}",if total==0 {"PASS"} else {"FAIL"});
    if total!=0 {std::process::exit(1);}
}
