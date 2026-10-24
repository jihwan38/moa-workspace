//! Matched ABBA probes for the isolated final TU-transpose change.
use furiosa_opt_gemma4::{
    Chip,
    axes::{B, H, L},
    ops_batch,
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
    let mut device = Device::new(ops_batch::batch_ffn.topology()).unwrap();
    let values: Vec<bf16> = (0..B::SIZE * H::SIZE)
        .map(|i| bf16::from_f32((word(i, 11) % 257) as f32 / 256.0 - 0.5))
        .collect();
    let input: HbmTensor<bf16, Chip, m![B, H]> =
        HostTensor::<bf16, m![B, H]>::from_vec(values.iter().copied())
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
    let mut output: HbmTensor<bf16, Chip, m![B, H]> =
        upload(&mut device, vec![0; B::SIZE * H::SIZE * 2]).await;
    let start = std::time::Instant::now();
    launch(
        ops_batch::batch_ffn,
        (
            &mut device,
            &input,
            &up,
            &gate,
            &down,
            &us,
            &gs,
            &ds,
            &ug,
            &gg,
            &dg,
            &mut output,
        ),
    )
    .await
    .unwrap();
    let actual: HostTensor<bf16, m![B, H]> = output.to_host(&mut device.pdma).await.unwrap();
    let cold_ms = start.elapsed().as_secs_f64() * 1000.0;
    let actual = actual.into_vec();
    launch(ops_batch::batch_ffn_rowmajor,(&mut device,&input,&up,&gate,&down,&us,&gs,&ds,&ug,&gg,&dg,&mut output)).await.unwrap();
    let changed:HostTensor<bf16,m![B,H]>=output.to_host(&mut device.pdma).await.unwrap();
    let mismatch=actual.iter().zip(changed.into_vec()).filter(|(a,b)| a.to_bits()!=b.to_bits()).count();
    assert_eq!(mismatch,0,"rowmajor FFN changed arithmetic");
    let mut times=[Vec::<f64>::new(),Vec::<f64>::new()];
    for variant in [0,1,1,0,0,1,1,0] {
        let start=std::time::Instant::now();
        for _ in 0..5 {
            if variant==0 { launch(ops_batch::batch_ffn,(&mut device,&input,&up,&gate,&down,&us,&gs,&ds,&ug,&gg,&dg,&mut output)).await.unwrap(); }
            else { launch(ops_batch::batch_ffn_rowmajor,(&mut device,&input,&up,&gate,&down,&us,&gs,&ds,&ug,&gg,&dg,&mut output)).await.unwrap(); }
        }
        let _:HostTensor<bf16,m![B,H]>=output.to_host(&mut device.pdma).await.unwrap();
        times[variant].push(start.elapsed().as_secs_f64()*1000.0/5.0);
    }
    println!("FFN rowmajor_transfer checked={} mismatches={mismatch} cold_original_ms={cold_ms:.3} original_ms={:?} rowmajor_ms={:?} scope=5launches+1read_per_sample_divided_by5",actual.len(),times[0],times[1]);
    query_probe(&mut device).await;
}

async fn query_probe(device:&mut Device) {
    use furiosa_opt_gemma4::axes::Qs;
    let x:HbmTensor<bf16,Chip,m![B,H]>=HostTensor::<bf16,m![B,H]>::from_vec((0..B::SIZE*H::SIZE).map(|i|bf16::from_f32((word(i,37)%257)as f32/256.0-0.5))).to_hbm(&mut device.pdma).await.unwrap();
    let w:HbmTensor<f8e4m3,Chip,m![Qs,H]>=HostTensor::<f8e4m3,m![Qs,H]>::from_vec((0..Qs::SIZE*H::SIZE).map(|i|f8e4m3::from_f32((word(i,41)%33)as f32/32.0-0.5))).to_hbm(&mut device.pdma).await.unwrap();
    let s:HbmTensor<bf16,Chip,m![Qs]>=HostTensor::<bf16,m![Qs]>::from_vec((0..Qs::SIZE).map(|i|bf16::from_f32(((word(i,43)%8)+1)as f32/16.0))).to_hbm(&mut device.pdma).await.unwrap();
    let mut output:HbmTensor<bf16,Chip,m![B,Qs]>=upload(device,vec![0;B::SIZE*Qs::SIZE*2]).await;
    launch(ops_batch::project_query_batch,(device,&x,&w,&s,&mut output)).await.unwrap();
    let original:HostTensor<bf16,m![B,Qs]>=output.to_host(&mut device.pdma).await.unwrap();
    launch(ops_batch::project_query_batch_rowmajor,(device,&x,&w,&s,&mut output)).await.unwrap();
    let changed:HostTensor<bf16,m![B,Qs]>=output.to_host(&mut device.pdma).await.unwrap();
    let mismatch=original.into_vec().iter().zip(changed.into_vec()).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();
    assert_eq!(mismatch,0,"rowmajor query changed arithmetic");
    let mut times=[Vec::<f64>::new(),Vec::<f64>::new()];
    for variant in [0,1,1,0,0,1,1,0] {
        let start=std::time::Instant::now();
        for _ in 0..5 {
            if variant==0 { launch(ops_batch::project_query_batch,(device,&x,&w,&s,&mut output)).await.unwrap(); }
            else { launch(ops_batch::project_query_batch_rowmajor,(device,&x,&w,&s,&mut output)).await.unwrap(); }
        }
        let _:HostTensor<bf16,m![B,Qs]>=output.to_host(&mut device.pdma).await.unwrap();
        times[variant].push(start.elapsed().as_secs_f64()*1000.0/5.0);
    }
    println!("query rowmajor_transfer checked={} mismatches={mismatch} original_ms={:?} rowmajor_ms={:?} scope=5launches+1read_per_sample_divided_by5",B::SIZE*Qs::SIZE,times[0],times[1]);
}
