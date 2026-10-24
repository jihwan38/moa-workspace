// Exact dequant-once BF16-weight FFN versus packed-weight FFN.
use furiosa_opt_gemma4::{
    Chip,
    axes::{B, H, L},
    ops_batch, ops_batch_predecoded,
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
    let bup:HbmTensor<bf16,Chip,m![L,H]>=predecode(&mut device,13,23).await;
    let bgate:HbmTensor<bf16,Chip,m![L,H]>=predecode(&mut device,17,29).await;
    let bdown:HbmTensor<bf16,Chip,m![H,L]>=predecode(&mut device,19,31).await;
    launch(ops_batch_predecoded::batch_ffn,(&mut device,&input,&bup,&bgate,&bdown,&ug,&gg,&dg,&mut output)).await.unwrap();
    let decoded:HostTensor<bf16,m![B,H]>=output.to_host(&mut device.pdma).await.unwrap();
    let decoded=decoded.into_vec();
    let mut mismatches=0;
    for (i,(expected,got)) in actual.iter().zip(decoded).enumerate() {
        if expected.to_bits()!=got.to_bits() {
            if mismatches<10 { eprintln!("i={i} got={} expected={}",got.to_f32(),expected.to_f32()); }
            mismatches+=1;
        }
    }
    let mut times=[Vec::<f64>::new(),Vec::<f64>::new()];
    for variant in [0,1,1,0,0,1,1,0] {
        let start=std::time::Instant::now();
        for _ in 0..5 {
            if variant==0 { launch(ops_batch::batch_ffn,(&mut device,&input,&up,&gate,&down,&us,&gs,&ds,&ug,&gg,&dg,&mut output)).await.unwrap(); }
            else { launch(ops_batch_predecoded::batch_ffn,(&mut device,&input,&bup,&bgate,&bdown,&ug,&gg,&dg,&mut output)).await.unwrap(); }
        }
        let _:HostTensor<bf16,m![B,H]>=output.to_host(&mut device.pdma).await.unwrap();
        times[variant].push(start.elapsed().as_secs_f64()*1000.0/5.0);
    }
    println!("FFN predecoded B={} checked={} mismatches={mismatches} original_cold_ms={cold_ms:.3} packed_ms={:?} predecoded_ms={:?} scope=5launches+1read_per_sample_divided_by5",B::SIZE,actual.len(),times[0],times[1]);
    assert_eq!(mismatches,0);
}

async fn predecode<E:M>(device:&mut Device,weight_seed:u64,scale_seed:u64)->HbmTensor<bf16,Chip,E> {
    let packed:Vec<u8>=(0..E::SIZE/2).map(|i|word(i,weight_seed) as u8).collect();
    let scales:Vec<f32>=(0..E::SIZE/16).map(|i|f8e4m3::from_f32(((word(i,scale_seed)%8)+1) as f32/128.0).to_f32()).collect();
    let lookup=[0.0f32,0.5,1.0,1.5,2.0,3.0,4.0,6.0,-0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0];
    let mut bytes=Vec::with_capacity(E::SIZE*2);
    for i in 0..E::SIZE {
        let nibble=(packed[i/2] >> ((i%2)*4)) & 15;
        let value=bf16::from_f32(lookup[nibble as usize]*scales[i/16]);
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    upload(device,bytes).await
}
