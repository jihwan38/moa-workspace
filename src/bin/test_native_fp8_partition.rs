//! H960 FFN differential test; F32 reassociation may change BF16 rounding.
use furiosa_opt_gemma4::{
    Chip,
    axes::{H, L, P128},
    ops_batch_down960, ops_batch_prepack8, ops_native_fp8, ops_native_fp8_partition,
};
use furiosa_opt_std::prelude::*;

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
    let mut device = Device::new(ops_batch_down960::batch_ffn.topology()).unwrap();
    let values: Vec<bf16> = (0..P128::SIZE * H::SIZE)
        .map(|i| { let row=i/H::SIZE; if row%16==0 {bf16::from_bits(0)} else if row%16==1 {bf16::from_bits(if i%2==0 {1} else {0x8001})} else {bf16::from_f32((((word(i,11)>>40) as f32 / 16777216.0)*2.0-1.0)*4.0*2f32.powi(-((word(i,13)%9)as i32)))} })
        .collect();
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
    let start=std::time::Instant::now();
    launch(ops_native_fp8::batch_ffn,(&mut device,&input,&qu,&qg,&qd,&su,&sg,&sd,&ug,&gg,&dg,&mut output)).await.unwrap();
    let actual:HostTensor<bf16,m![P128,H]>=output.to_host(&mut device.pdma).await.unwrap();
    let actual=actual.into_vec();let cold_ms=start.elapsed().as_secs_f64()*1000.0;
    launch(ops_native_fp8_partition::batch_ffn,(&mut device,&input,&qu,&qg,&qd,&su,&sg,&sd,&ug,&gg,&dg,&mut output)).await.unwrap();
    let changed:HostTensor<bf16,m![P128,H]>=output.to_host(&mut device.pdma).await.unwrap();
    let mut bitdiff=0;let mut failures=0;let mut max_abs=0f32;let mut max_rel=0f32;let mut err2=0f64;let mut ref2=0f64;let mut nonfinite=0;
    for (i,(expected,got)) in actual.iter().zip(changed.into_vec()).enumerate() {
        if expected.to_bits()!=got.to_bits(){bitdiff+=1;}
        let e=expected.to_f32();let a=got.to_f32();let error=(e-a).abs();err2+=(error as f64).powi(2);ref2+=(e as f64).powi(2);nonfinite+=usize::from(!a.is_finite());
        max_abs=max_abs.max(error);max_rel=max_rel.max(error/e.abs().max(0.01));
        if !a.is_finite() || error>0.02+0.01*e.abs(){if failures<8{eprintln!("i={i} got={a} expected={e}");}failures+=1;}
    }
    let mut times=[Vec::<f64>::new(),Vec::<f64>::new()];
    for variant in [0,1,1,0] {
        let start=std::time::Instant::now();
        for _ in 0..3 {
            if variant==0{launch(ops_native_fp8::batch_ffn,(&mut device,&input,&qu,&qg,&qd,&su,&sg,&sd,&ug,&gg,&dg,&mut output)).await.unwrap();}
            else{launch(ops_native_fp8_partition::batch_ffn,(&mut device,&input,&qu,&qg,&qd,&su,&sg,&sd,&ug,&gg,&dg,&mut output)).await.unwrap();}
        }
        let _:HostTensor<bf16,m![P128,H]>=output.to_host(&mut device.pdma).await.unwrap();
        times[variant].push(start.elapsed().as_secs_f64()*1000.0/3.0);
    }
    println!("P128PartitionFP8FFN checked={} bit_differences={bitdiff} tolerance_failures={failures} max_abs={max_abs} max_rel={max_rel} cold_control_ms={cold_ms:.3} base_FP8_ms={:?} partition_FP8_ms={:?} scope=3launches+1read_divided_by3",actual.len(),times[0],times[1]);
    let nrmse=(err2/ref2.max(1e-30)).sqrt();println!("approximate_FFN relative_l2={nrmse} nonfinite={nonfinite} full_model_accuracy=UNVALIDATED");assert_eq!(nonfinite,0);assert_eq!(bitdiff,0,"partition quantization changed FP8 output bits");
}
