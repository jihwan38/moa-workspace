//! H960 FFN differential test; F32 reassociation may change BF16 rounding.
use furiosa_opt_gemma4::{
    Chip,
    axes::{H, L, P128},
    ops_batch_down960, ops_batch_prepack8, ops_native_fp8,
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
        .map(|i| bf16::from_f32((((word(i,11)>>40) as f32 / 16777216.0)*2.0-1.0)*4.0*2f32.powi(-((word(i,13)%9)as i32))))
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
    let start = std::time::Instant::now();
    launch(
        ops_batch_down960::batch_ffn,
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
    let actual: HostTensor<bf16, m![P128, H]> = output.to_host(&mut device.pdma).await.unwrap();
    let cold_ms = start.elapsed().as_secs_f64() * 1000.0;
    let actual = actual.into_vec();
    let mut qu:HbmTensor<f8e4m3,Chip,m![L,H]>=upload(&mut device,vec![0;L::SIZE*H::SIZE]).await;
    let mut qg:HbmTensor<f8e4m3,Chip,m![L,H]>=upload(&mut device,vec![0;L::SIZE*H::SIZE]).await;
    let mut qd:HbmTensor<f8e4m3,Chip,m![H,L]>=upload(&mut device,vec![0;H::SIZE*L::SIZE]).await;
    let mut su:HbmTensor<f32,Chip,m![L]>=upload(&mut device,vec![0;L::SIZE*4]).await;
    let mut sg:HbmTensor<f32,Chip,m![L]>=upload(&mut device,vec![0;L::SIZE*4]).await;
    let mut sd:HbmTensor<f32,Chip,m![H]>=upload(&mut device,vec![0;H::SIZE*4]).await;
    launch(ops_batch_prepack8::prepack_up,(&mut device,&up,&us,&mut qu,&mut su)).await.unwrap();
    launch(ops_batch_prepack8::prepack_up,(&mut device,&gate,&gs,&mut qg,&mut sg)).await.unwrap();
    launch(ops_batch_prepack8::prepack_down,(&mut device,&down,&ds,&mut qd,&mut sd)).await.unwrap();
    check_up(&mut device,&qu,&su).await;
    check_down(&mut device,&qd,&sd).await;
    launch(ops_native_fp8::batch_ffn,(&mut device,&input,&qu,&qg,&qd,&su,&sg,&sd,&ug,&gg,&dg,&mut output)).await.unwrap();
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
            if variant==0{launch(ops_batch_down960::batch_ffn,(&mut device,&input,&up,&gate,&down,&us,&gs,&ds,&ug,&gg,&dg,&mut output)).await.unwrap();}
            else{launch(ops_native_fp8::batch_ffn,(&mut device,&input,&qu,&qg,&qd,&su,&sg,&sd,&ug,&gg,&dg,&mut output)).await.unwrap();}
        }
        let _:HostTensor<bf16,m![P128,H]>=output.to_host(&mut device.pdma).await.unwrap();
        times[variant].push(start.elapsed().as_secs_f64()*1000.0/3.0);
    }
    println!("P128NativeFP8FFN checked={} bit_differences={bitdiff} tolerance_failures={failures} max_abs={max_abs} max_rel={max_rel} cold_control_ms={cold_ms:.3} packed_L960_ms={:?} native_FP8_ms={:?} scope=3launches+1read_divided_by3",actual.len(),times[0],times[1]);
    let nrmse=(err2/ref2.max(1e-30)).sqrt();println!("approximate_FFN relative_l2={nrmse} nonfinite={nonfinite} full_model_accuracy=UNVALIDATED");assert_eq!(nonfinite,0);assert!(nrmse<0.2,"gross-error diagnostic gate failed; this is not a model accuracy gate");
}

macro_rules! matrix_check {($name:ident,$out:ident,$kin:ident,$kernel:ident)=>{async fn $name(device:&mut Device,weight:&HbmTensor<f8e4m3,Chip,m![$out,$kin]>,weight_scale:&HbmTensor<f32,Chip,m![$out]>) {
 let w:HostTensor<f8e4m3,m![$out,$kin]>=weight.to_host(&mut device.pdma).await.unwrap();let w=w.into_vec();
 let ws:HostTensor<f32,m![$out]>=weight_scale.to_host(&mut device.pdma).await.unwrap();let ws=ws.into_vec();
 let x:Vec<f8e4m3>=(0..P128::SIZE*$kin::SIZE).map(|i|{let b=word(i,83);f8e4m3::from_f32((((b>>40)as f32/16777216.0)*2.0-1.0)*64.0*2f32.powi(-((b%9)as i32)))}).collect();
 let xs:Vec<f32>=(0..P128::SIZE).map(|i|2f32.powi((word(i,89)%7)as i32-3)).collect();
 let xh=HostTensor::<f8e4m3,m![P128,$kin]>::from_vec(x.iter().copied()).to_hbm(&mut device.pdma).await.unwrap();
 let xsh=HostTensor::<f32,m![P128]>::from_vec(xs.iter().copied()).to_hbm(&mut device.pdma).await.unwrap();
 let mut output=HostTensor::<bf16,m![P128,$out]>::zero().to_hbm(&mut device.pdma).await.unwrap();
 launch(ops_native_fp8::$kernel,(device,&xh,&xsh,weight,weight_scale,&mut output)).await.unwrap();let got:HostTensor<bf16,m![P128,$out]>=output.to_host(&mut device.pdma).await.unwrap();let got=got.into_vec();
 let(mut bitdiff,mut failures,mut maxabs)=(0,0,0f32);
 for j in 0..1024 {let b=(word(j,97)as usize)%P128::SIZE;let n=(word(j,101)as usize)%$out::SIZE;let dot:f64=(0..$kin::SIZE).map(|k|x[b*$kin::SIZE+k].to_f32()as f64*w[n*$kin::SIZE+k].to_f32()as f64).sum();let expected=bf16::from_f32((dot as f32*xs[b])*ws[n]);let actual=got[b*$out::SIZE+n];bitdiff+=usize::from(expected.to_bits()!=actual.to_bits());let e=(expected.to_f32()-actual.to_f32()).abs();maxabs=maxabs.max(e);if !actual.to_f32().is_finite()||e>0.003+0.01*expected.to_f32().abs(){if failures<6{eprintln!("scaled {} row={b} col={n} got={} expected={}",stringify!($out),actual.to_f32(),expected.to_f32());}failures+=1;}}
 println!("scaled_FP8_{} checked=1024 bitdiff={bitdiff} tolerance_failures={failures} max_abs={maxabs}",stringify!($out));assert_eq!(failures,0);
}}}
matrix_check!(check_up,L,H,project_up_scaled);matrix_check!(check_down,H,L,project_down_scaled);
