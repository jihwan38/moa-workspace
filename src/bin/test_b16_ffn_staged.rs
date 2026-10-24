//! H960 FFN differential test; F32 reassociation may change BF16 rounding.
use furiosa_opt_gemma4::{
    Chip,
    axes::{H, L, B},
    ops_batch_prepack8, ops_native_fp8_blocked, ops_native_fp8_b16_blocked, ops_native_fp8_b16_staged,
};
use furiosa_opt_std::prelude::*;
#[path="support/npu_profile.rs"] mod npu_profile;

fn word(i: usize, seed: u64) -> u64 {
    let mut x = (i as u64).wrapping_add(seed);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
async fn upload<D: MaterializableScalar, E: M>(
    device: &mut Device,
    bytes: Vec<u8>,
) -> HbmTensor<D, Chip, E> {
    HostTensor::<D, E>::from_buf(bytes)
        .to_hbm(&mut device.pdma)
        .await
        .unwrap()
}
async fn packed<E: M>(device: &mut Device, seed: u64) -> HbmTensor<f4e2m1, Chip, E> {
    upload(
        device,
        (0..E::SIZE / 2).map(|i| word(i, seed) as u8).collect(),
    )
    .await
}
async fn scales<E: M>(device: &mut Device, seed: u64) -> HbmTensor<f8e4m3, Chip, E> {
    HostTensor::<f8e4m3, E>::from_vec(
        (0..E::SIZE).map(|i| f8e4m3::from_f32(((word(i, seed) % 8) + 1) as f32 / 128.0)),
    )
    .to_hbm(&mut device.pdma)
    .await
    .unwrap()
}
async fn global(device: &mut Device, v: f32) -> HbmTensor<f32, Chip, m![1]> {
    HostTensor::<f32, m![1]>::from_vec([v])
        .to_hbm(&mut device.pdma)
        .await
        .unwrap()
}
#[tokio::main]
async fn main() {
    let collector=npu_profile::Collector::default();tracing::subscriber::set_global_default(collector.clone()).unwrap();
    let mut device=Device::new(ops_native_fp8_b16_staged::quantize_hidden.topology()).unwrap();
    let values:Vec<bf16>=(0..B::SIZE*H::SIZE).map(|i| {
        let row=i/H::SIZE;
        bf16::from_f32(if row==0 {0.0} else if row==1 {if i%2==0 {2f32.powi(-127)} else {-2f32.powi(-127)}} else {
            (((word(i,11)>>40) as f32 /16777216.0)*2.0-1.0)*4.0*2f32.powi(-((word(i,13)%9)as i32))})
    }).collect();
    let input:HbmTensor<bf16,Chip,m![B,H]>=HostTensor::<bf16,m![B,H]>::from_vec(values.iter().copied()).to_hbm(&mut device.pdma).await.unwrap();
    let second:Vec<bf16>=values.iter().enumerate().map(|(i,v)|if i/H::SIZE<2{*v}else{bf16::from_f32(v.to_f32()*-0.75+((i%11)as f32-5.)/32.)}).collect();
    let input2:HbmTensor<bf16,Chip,m![B,H]>=HostTensor::<bf16,m![B,H]>::from_vec(second).to_hbm(&mut device.pdma).await.unwrap();
    let inputs=[input,input2];
    let mut qinput:HbmTensor<f8e4m3,Chip,m![B,H]>=upload(&mut device,vec![0;B::SIZE*H::SIZE]).await;
    let mut qhidden:HbmTensor<f8e4m3,Chip,m![B,L]>=upload(&mut device,vec![0;B::SIZE*L::SIZE]).await;
    let mut xscale:HbmTensor<f32,Chip,m![B]>=upload(&mut device,vec![0;B::SIZE*4]).await;
    let mut hscale:HbmTensor<f32,Chip,m![B]>=upload(&mut device,vec![0;B::SIZE*4]).await;
    let mut output:HbmTensor<bf16,Chip,m![B,H]>=upload(&mut device,vec![0;B::SIZE*H::SIZE*2]).await;
    let mut hidden:HbmTensor<bf16,Chip,m![B,L]>=upload(&mut device,vec![0;B::SIZE*L::SIZE*2]).await;
    let up: HbmTensor<f4e2m1, Chip, m![L, H]> = packed(&mut device, 13).await;
    let gate: HbmTensor<f4e2m1, Chip, m![L, H]> = packed(&mut device, 17).await;
    let down: HbmTensor<f4e2m1, Chip, m![H, L]> = packed(&mut device, 19).await;
    let us: HbmTensor<f8e4m3, Chip, m![L, H / 16]> = scales(&mut device, 23).await;
    let gs: HbmTensor<f8e4m3, Chip, m![L, H / 16]> = scales(&mut device, 29).await;
    let ds: HbmTensor<f8e4m3, Chip, m![H, L / 16]> = scales(&mut device, 31).await;
    let ug = global(&mut device, 0.125).await;
    let gg = global(&mut device, 0.125).await;
    let dg = global(&mut device, 0.125).await;
    let mut qu:HbmTensor<f8e4m3,Chip,m![L,H]>=upload(&mut device,vec![0;L::SIZE*H::SIZE]).await;
    let mut qg:HbmTensor<f8e4m3,Chip,m![L,H]>=upload(&mut device,vec![0;L::SIZE*H::SIZE]).await;
    let mut qd:HbmTensor<f8e4m3,Chip,m![H,L]>=upload(&mut device,vec![0;H::SIZE*L::SIZE]).await;
    let mut su:HbmTensor<f32,Chip,m![L]>=upload(&mut device,vec![0;L::SIZE*4]).await;
    let mut sg:HbmTensor<f32,Chip,m![L]>=upload(&mut device,vec![0;L::SIZE*4]).await;
    let mut sd:HbmTensor<f32,Chip,m![H]>=upload(&mut device,vec![0;H::SIZE*4]).await;
    launch(ops_batch_prepack8::prepack_up,(&mut device,&up,&us,&mut qu,&mut su)).await.unwrap();
    launch(ops_batch_prepack8::prepack_up,(&mut device,&gate,&gs,&mut qg,&mut sg)).await.unwrap();
    launch(ops_batch_prepack8::prepack_down,(&mut device,&down,&ds,&mut qd,&mut sd)).await.unwrap();
    let mut bu:HbmTensor<f8e4m3,Chip,ops_native_fp8_blocked::UpWeight>=upload(&mut device,vec![0;L::SIZE*H::SIZE]).await;
    let mut bg:HbmTensor<f8e4m3,Chip,ops_native_fp8_blocked::UpWeight>=upload(&mut device,vec![0;L::SIZE*H::SIZE]).await;
    let mut bd:HbmTensor<f8e4m3,Chip,ops_native_fp8_blocked::DownWeight>=upload(&mut device,vec![0;L::SIZE*H::SIZE]).await;
    launch(ops_native_fp8_blocked::reblock_up,(&mut device,&qu,&mut bu)).await.unwrap();
    launch(ops_native_fp8_blocked::reblock_up,(&mut device,&qg,&mut bg)).await.unwrap();
    launch(ops_native_fp8_blocked::reblock_down,(&mut device,&qd,&mut bd)).await.unwrap();



    macro_rules! control {($input:expr) => {launch(ops_native_fp8_b16_blocked::batch_ffn,(&mut device,$input,&bu,&bg,&bd,&su,&sg,&sd,&ug,&gg,&dg,&mut output)).await.unwrap()};}
    macro_rules! candidate {($input:expr) => {{
        launch(ops_native_fp8_b16_staged::quantize_hidden,(&mut device,$input,&mut qinput,&mut xscale)).await.unwrap();
        launch(ops_native_fp8_b16_staged::upgate_geglu_prequantized,(&mut device,&qinput,&xscale,&bu,&bg,&su,&sg,&ug,&gg,&mut hidden)).await.unwrap();
        launch(ops_native_fp8_b16_staged::quantize_ffn,(&mut device,&hidden,&mut qhidden,&mut hscale)).await.unwrap();
        launch(ops_native_fp8_b16_staged::down_prequantized,(&mut device,&qhidden,&hscale,&bd,&sd,&dg,&mut output)).await.unwrap();
    }};}
    for case in 0..2 {
        control!(&inputs[case]);
        let reference:HostTensor<bf16,m![B,H]>=output.to_host(&mut device.pdma).await.unwrap();let reference=reference.into_vec();
        candidate!(&inputs[case]);
        let result:HostTensor<bf16,m![B,H]>=output.to_host(&mut device.pdma).await.unwrap();let result=result.into_vec();
        let differences=reference.iter().zip(&result).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();
        println!("B_HBM_four_stage_FFN case={case} checked={} bit_differences={differences}",result.len());
        for (i,(a,b)) in reference.iter().zip(&result).enumerate().filter(|(_, (a,b))|a.to_bits()!=b.to_bits()).take(20){eprintln!("difference index={i} reference={} candidate={}",a.to_f32(),b.to_f32());}
        assert_eq!(differences,0);
    }
    let mut times=[Vec::<f64>::new(),Vec::<f64>::new()];
    for variant in [0,1,1,0] {
        let start=std::time::Instant::now();
        for _ in 0..10 {if variant==0 {control!(&inputs[1]);} else {candidate!(&inputs[1]);}}
        times[variant].push(start.elapsed().as_secs_f64()*1000.0/10.0);
    }
    println!("B_FFN localquant_TRF_control_ms={:?} four_stage_HBM_ms={:?} scope=10completedlaunches_nohostread",times[0],times[1]);
    if std::env::var_os("FURIOSA_OPT_PROFILE").is_some() {for variant in [0,1] {
        println!("PROFILE_VARIANT {variant}");collector.enable(true);
        if variant==0 {control!(&inputs[1]);} else {candidate!(&inputs[1]);}
        let _:HostTensor<bf16,m![B,H]>=output.to_host(&mut device.pdma).await.unwrap();collector.enable(false);
    }}
}
