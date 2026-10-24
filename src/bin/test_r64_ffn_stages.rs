//! H960 FFN differential test; F32 reassociation may change BF16 rounding.
use furiosa_opt_gemma4::{
    Chip,
    axes::{H, L, R64},
    ops_batch_prepack8, ops_native_fp8_blocked, ops_native_fp8_r64_local_control, ops_native_fp8_r64_split,
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
fn quantize(values:&[bf16],width:usize)->(Vec<f8e4m3>,Vec<f32>){
 let mut q=Vec::new();let mut scales=Vec::new();
 for row in values.chunks(width){let amax=row.iter().map(|v|v.to_f32().abs()).fold(0f32,f32::max).max(2f32.powi(-112));let scale=f32::from_bits(amax.to_bits()&0x7f800000)/128.;scales.push(scale);for v in row{let x=v.to_f32();let x=if x.abs()<f32::MIN_POSITIVE{f32::from_bits(x.to_bits()&0x80000000)}else{x};q.push(f8e4m3::from_f32(x/scale));}}(q,scales)
}
fn compare(name:&str,a:&[bf16],b:&[bf16]){let differences=a.iter().zip(b).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();println!("STAGE {name} checked={} bitdiff={differences}",a.len());for(i,(a,b))in a.iter().zip(b).enumerate().filter(|(_, (a,b))|a.to_bits()!=b.to_bits()).take(12){println!("difference stage={name} index={i} expected={} actual={}",a.to_f32(),b.to_f32());}}

#[tokio::main]
async fn main() {
    let collector=npu_profile::Collector::default();tracing::subscriber::set_global_default(collector.clone()).unwrap();
    let mut device=Device::new(ops_native_fp8_r64_split::upgate_geglu.topology()).unwrap();
    let values:Vec<bf16>=(0..R64::SIZE*H::SIZE).map(|i| {
        let row=i/H::SIZE;
        bf16::from_f32(if row==0 {0.0} else if row==1 {if i%2==0 {2f32.powi(-127)} else {-2f32.powi(-127)}} else {
            (((word(i,11)>>40) as f32 /16777216.0)*2.0-1.0)*4.0*2f32.powi(-((word(i,13)%9)as i32))})
    }).collect();
    let input:HbmTensor<bf16,Chip,m![R64,H]>=HostTensor::<bf16,m![R64,H]>::from_vec(values.iter().copied()).to_hbm(&mut device.pdma).await.unwrap();
    let mut output:HbmTensor<bf16,Chip,m![R64,H]>=upload(&mut device,vec![0;R64::SIZE*H::SIZE*2]).await;
    let mut hidden:HbmTensor<bf16,Chip,m![R64,L]>=upload(&mut device,vec![0;R64::SIZE*L::SIZE*2]).await;
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


    macro_rules! control {() => {launch(ops_native_fp8_r64_local_control::batch_ffn_partitioned_input,(&mut device,&input,&bu,&bg,&bd,&su,&sg,&sd,&ug,&gg,&dg,&mut output)).await.unwrap()};}
    macro_rules! candidate {() => {{
        launch(ops_native_fp8_r64_split::upgate_geglu,(&mut device,&input,&bu,&bg,&su,&sg,&ug,&gg,&mut hidden)).await.unwrap();
        launch(ops_native_fp8_r64_split::down,(&mut device,&hidden,&bd,&sd,&dg,&mut output)).await.unwrap();
    }};}
    control!();
    let reference:HostTensor<bf16,m![R64,H]>=output.to_host(&mut device.pdma).await.unwrap();let reference=reference.into_vec();

    let (qx,xs)=quantize(&values,H::SIZE);
    let qx: HbmTensor<f8e4m3,Chip,m![R64,H]>=HostTensor::<f8e4m3,m![R64,H]>::from_vec(qx).to_hbm(&mut device.pdma).await.unwrap();
    let xs: HbmTensor<f32,Chip,m![R64]>=HostTensor::<f32,m![R64]>::from_vec(xs).to_hbm(&mut device.pdma).await.unwrap();
    let mut hidden_ref:HbmTensor<bf16,Chip,m![R64,L]>=upload(&mut device,vec![0;R64::SIZE*L::SIZE*2]).await;
    launch(ops_native_fp8_r64_split::upgate_geglu_prequantized,(&mut device,&qx,&xs,&bu,&bg,&su,&sg,&ug,&gg,&mut hidden_ref)).await.unwrap();
    let hr:HostTensor<bf16,m![R64,L]>=hidden_ref.to_host(&mut device.pdma).await.unwrap();let hr=hr.into_vec();
    launch(ops_native_fp8_r64_split::upgate_geglu,(&mut device,&input,&bu,&bg,&su,&sg,&ug,&gg,&mut hidden)).await.unwrap();
    let hc:HostTensor<bf16,m![R64,L]>=hidden.to_host(&mut device.pdma).await.unwrap();let hc=hc.into_vec();
    compare("upgate_seeded_vs_CPU_quantized_input",&hr,&hc);
    let (qh,hs)=quantize(&hr,L::SIZE);
    let qh: HbmTensor<f8e4m3,Chip,m![R64,L]>=HostTensor::<f8e4m3,m![R64,L]>::from_vec(qh).to_hbm(&mut device.pdma).await.unwrap();
    let hs: HbmTensor<f32,Chip,m![R64]>=HostTensor::<f32,m![R64]>::from_vec(hs).to_hbm(&mut device.pdma).await.unwrap();
    launch(ops_native_fp8_r64_split::down_prequantized,(&mut device,&qh,&hs,&bd,&sd,&dg,&mut output)).await.unwrap();
    let pr:HostTensor<bf16,m![R64,H]>=output.to_host(&mut device.pdma).await.unwrap();let pr=pr.into_vec();
    compare("both_prequantized_vs_fused_original_control",&reference,&pr);
    launch(ops_native_fp8_r64_split::down,(&mut device,&hidden_ref,&bd,&sd,&dg,&mut output)).await.unwrap();
    let dr:HostTensor<bf16,m![R64,H]>=output.to_host(&mut device.pdma).await.unwrap();let dr=dr.into_vec();
    compare("down_seeded_vs_CPU_quantized_hidden",&pr,&dr);
}
