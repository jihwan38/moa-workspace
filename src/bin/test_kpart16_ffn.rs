//! H960 FFN differential test; F32 reassociation may change BF16 rounding.
use furiosa_opt_gemma4::{
    Chip,
    axes::{H, L, B},
    ops_batch_clusters, ops_batch_kpart16,
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
    let mut device = Device::new(ops_batch_clusters::batch_ffn_clusters.topology()).unwrap();
    let values: Vec<bf16> = (0..B::SIZE * H::SIZE)
        .map(|i| bf16::from_f32((((word(i,11)>>40) as f32 / 16777216.0)*2.0-1.0)*4.0*2f32.powi(-((word(i,13)%9)as i32))))
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
        ops_batch_clusters::batch_ffn_clusters,
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
    launch(ops_batch_kpart16::batch_ffn,(&mut device,&input,&up,&gate,&down,&us,&gs,&ds,&ug,&gg,&dg,&mut output)).await.unwrap();
    let changed:HostTensor<bf16,m![B,H]>=output.to_host(&mut device.pdma).await.unwrap();
    let mut bitdiff=0;let mut failures=0;let mut max_abs=0f32;let mut max_rel=0f32;
    for (i,(expected,got)) in actual.iter().zip(changed.into_vec()).enumerate() {
        if expected.to_bits()!=got.to_bits(){bitdiff+=1;}
        let e=expected.to_f32();let a=got.to_f32();let error=(e-a).abs();
        max_abs=max_abs.max(error);max_rel=max_rel.max(error/e.abs().max(0.01));
        if !a.is_finite() || error>0.02+0.01*e.abs(){if failures<8{eprintln!("i={i} got={a} expected={e}");}failures+=1;}
    }
    let mut times=[Vec::<f64>::new(),Vec::<f64>::new()];
    for variant in [0,1,1,0] {
        let start=std::time::Instant::now();
        for _ in 0..3 {
            if variant==0{launch(ops_batch_clusters::batch_ffn_clusters,(&mut device,&input,&up,&gate,&down,&us,&gs,&ds,&ug,&gg,&dg,&mut output)).await.unwrap();}
            else{launch(ops_batch_kpart16::batch_ffn,(&mut device,&input,&up,&gate,&down,&us,&gs,&ds,&ug,&gg,&dg,&mut output)).await.unwrap();}
        }
        let _:HostTensor<bf16,m![B,H]>=output.to_host(&mut device.pdma).await.unwrap();
        times[variant].push(start.elapsed().as_secs_f64()*1000.0/3.0);
    }
    println!("BH960FFN checked={} bit_differences={bitdiff} tolerance_failures={failures} max_abs={max_abs} max_rel={max_rel} cold_control_ms={cold_ms:.3} fullH_cluster_ms={:?} H960_ms={:?} scope=3launches+1read_divided_by3",actual.len(),times[0],times[1]);
    assert_eq!(failures,0);
}
