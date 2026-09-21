use furiosa_opt_std::prelude::*;
use furiosa_opt_gemma4::{Chip,packet_probe};
use packet_probe::{ProbeK,ProbeR,ProbeS};
#[path = "probe_common/mod.rs"] mod common;

#[tokio::main]
async fn main() {
    let cases=common::cases();
    if common::oracle_only(&cases) { return; }
    let mut device=Device::new(packet_probe::fp8_scalar.topology()).expect("create NPU Device");
    let mut all_ok=true;
    for case in cases {
        let x: HbmTensor<f8e4m3,Chip,m![ProbeK]> = HostTensor::<f8e4m3,m![ProbeK]>::from_vec(case.x)
            .to_hbm(&mut device.pdma).await.unwrap();
        let w: HbmTensor<f8e4m3,Chip,m![ProbeR,ProbeK]> = HostTensor::<f8e4m3,m![ProbeR,ProbeK]>::from_vec(case.weights)
            .to_hbm(&mut device.pdma).await.unwrap();
        let mut out: HbmTensor<f32,Chip,m![ProbeS,ProbeR]> = HostTensor::<f32,m![ProbeS,ProbeR]>::from_vec(vec![f32::NAN; <m![ProbeS,ProbeR]>::SIZE])
            .to_hbm(&mut device.pdma).await.unwrap();
        common::mark("scalar",case.id,"LAUNCH_BEGIN");
        launch(packet_probe::fp8_scalar,(&mut device,&x,&w,&mut out)).await.expect("probe launch");
        common::mark("scalar",case.id,"LAUNCH_RETURNED");
        let host: HostTensor<f32,m![ProbeS,ProbeR]>=out.to_host(&mut device.pdma).await.unwrap();
        all_ok &= common::compare("scalar",case.id,&host.into_vec(),&case.scalar);
    }
    if !all_ok { std::process::exit(1); }
}
