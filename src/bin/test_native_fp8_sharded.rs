//! H960 FFN differential test; F32 reassociation may change BF16 rounding.
use furiosa_opt_gemma4::{
    Chip,
    axes::{H, L, P128},
    ops_batch_prepack8, ops_native_fp8, ops_native_fp8_blocked, ops_native_fp8_sharded,
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
    let collector=npu_profile::Collector::default();
    tracing::subscriber::set_global_default(collector.clone()).unwrap();
    let mut device = Device::new(ops_native_fp8::batch_ffn.topology()).unwrap();
    let mut values: Vec<bf16> = (0..P128::SIZE * H::SIZE)
        .map(|i| bf16::from_f32((((word(i,11)>>40) as f32 / 16777216.0)*2.0-1.0)*4.0*2f32.powi(-((word(i,13)%9)as i32))))
        .collect();
    // Exercise zero and tiny/subnormal rows in addition to mixed-sign random data.
    for i in 0..H::SIZE { values[i]=bf16::from_f32(0.0); values[H::SIZE+i]=bf16::from_bits(if i%2==0 {1} else {0x8001}); }
    let input: HbmTensor<bf16, Chip, m![P128, H]> =
        HostTensor::<bf16, m![P128, H]>::from_vec(values.iter().copied())
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
    let up: HbmTensor<f4e2m1, Chip, m![L, H]> = packed(&mut device, 13).await;
    let gate: HbmTensor<f4e2m1, Chip, m![L, H]> = packed(&mut device, 17).await;
    let down: HbmTensor<f4e2m1, Chip, m![H, L]> = packed(&mut device, 19).await;
    let us: HbmTensor<f8e4m3, Chip, m![L, H / 16]> = scales(&mut device, 23).await;
    let gs: HbmTensor<f8e4m3, Chip, m![L, H / 16]> = scales(&mut device, 29).await;
    let ds: HbmTensor<f8e4m3, Chip, m![H, L / 16]> = scales(&mut device, 31).await;
    let ug = global(&mut device, 0.125).await;
    let gg = global(&mut device, 0.125).await;
    let dg = global(&mut device, 0.125).await;
    let mut output: HbmTensor<bf16, Chip, m![P128, H]> =
        upload(&mut device, vec![0; P128::SIZE * H::SIZE * 2]).await;
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
    launch(ops_native_fp8_blocked::batch_ffn_direct_geglu,(&mut device,&input,&bu,&bg,&bd,&su,&sg,&sd,&ug,&gg,&dg,&mut output)).await.unwrap();
    let original:HostTensor<bf16,m![P128,H]>=output.to_host(&mut device.pdma).await.unwrap();let original=original.into_vec();
    launch(ops_native_fp8_sharded::batch_ffn,(&mut device,&input,&bu,&bg,&bd,&su,&sg,&sd,&ug,&gg,&dg,&mut output)).await.unwrap();
    let blocked:HostTensor<bf16,m![P128,H]>=output.to_host(&mut device.pdma).await.unwrap();let blocked=blocked.into_vec();
    let differences=original.iter().zip(&blocked).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();
    println!("L_sharded_native_FP8_FFN checked={} bit_differences={differences}",original.len());assert_eq!(differences,0);
    launch(ops_native_fp8_sharded::batch_ffn_partitioned_input,(&mut device,&input,&bu,&bg,&bd,&su,&sg,&sd,&ug,&gg,&dg,&mut output)).await.unwrap();
    let partitioned:HostTensor<bf16,m![P128,H]>=output.to_host(&mut device.pdma).await.unwrap();let partitioned=partitioned.into_vec();
    let differences=original.iter().zip(&partitioned).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();
    println!("L_sharded_Hlocal_native_FP8_FFN checked={} bit_differences={differences}",original.len());assert_eq!(differences,0);
    launch(ops_native_fp8_sharded::batch_ffn_seeded_input,(&mut device,&input,&bu,&bg,&bd,&su,&sg,&sd,&ug,&gg,&dg,&mut output)).await.unwrap();
    let seeded:HostTensor<bf16,m![P128,H]>=output.to_host(&mut device.pdma).await.unwrap();let seeded=seeded.into_vec();
    let differences=original.iter().zip(&seeded).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();
    println!("L_sharded_Hseed_native_FP8_FFN checked={} bit_differences={differences}",original.len());assert_eq!(differences,0);
    let mut times=[Vec::<f64>::new(),Vec::<f64>::new(),Vec::<f64>::new(),Vec::<f64>::new()];
    for variant in [0,1,2,3,3,2,1,0] {
        let start=std::time::Instant::now();
        for _ in 0..5 {
            if variant==0{launch(ops_native_fp8_blocked::batch_ffn_direct_geglu,(&mut device,&input,&bu,&bg,&bd,&su,&sg,&sd,&ug,&gg,&dg,&mut output)).await.unwrap();}
            else if variant==1{launch(ops_native_fp8_sharded::batch_ffn,(&mut device,&input,&bu,&bg,&bd,&su,&sg,&sd,&ug,&gg,&dg,&mut output)).await.unwrap();}
            else if variant==2{launch(ops_native_fp8_sharded::batch_ffn_partitioned_input,(&mut device,&input,&bu,&bg,&bd,&su,&sg,&sd,&ug,&gg,&dg,&mut output)).await.unwrap();}
        else{launch(ops_native_fp8_sharded::batch_ffn_seeded_input,(&mut device,&input,&bu,&bg,&bd,&su,&sg,&sd,&ug,&gg,&dg,&mut output)).await.unwrap();}
        }
        let _:HostTensor<bf16,m![P128,H]>=output.to_host(&mut device.pdma).await.unwrap();times[variant].push(start.elapsed().as_secs_f64()*1000.0/5.0);
    }
    println!("native_FP8_FFN direct_geglu_control_ms={:?} L_sharded_ms={:?} L_sharded_Hlocal_ms={:?} L_sharded_Hseed_ms={:?} scope=5launches+1same_read_divided5",times[0],times[1],times[2],times[3]);
    if std::env::args().any(|a|a=="--profile") {
      for variant in 0..4 {
        println!("PROFILE_VARIANT {variant}");collector.enable(true);
        if variant==0{launch(ops_native_fp8_blocked::batch_ffn_direct_geglu,(&mut device,&input,&bu,&bg,&bd,&su,&sg,&sd,&ug,&gg,&dg,&mut output)).await.unwrap();}
        else if variant==1{launch(ops_native_fp8_sharded::batch_ffn,(&mut device,&input,&bu,&bg,&bd,&su,&sg,&sd,&ug,&gg,&dg,&mut output)).await.unwrap();}
        else if variant==2{launch(ops_native_fp8_sharded::batch_ffn_partitioned_input,(&mut device,&input,&bu,&bg,&bd,&su,&sg,&sd,&ug,&gg,&dg,&mut output)).await.unwrap();}
        else{launch(ops_native_fp8_sharded::batch_ffn_seeded_input,(&mut device,&input,&bu,&bg,&bd,&su,&sg,&sd,&ug,&gg,&dg,&mut output)).await.unwrap();}
        collector.enable(false);
      }
    }
}
