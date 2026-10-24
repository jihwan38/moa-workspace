//! H960 FFN differential test; F32 reassociation may change BF16 rounding.
use furiosa_opt_gemma4::{
    Chip,
    axes::{H, L, P128, P512},
    ops_batch_prepack8, ops_native_fp8_blocked, ops_native_fp8_p512, ops_native_fp8_p128_local_control, ops_native_fp8_p512_inputblocks, ops_native_fp8_p512_blocks, ops_batch_layouts::{P512H960,P512L960,P512L240},
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
fn check_blocked(name:&str,actual:&[u32],expected:&[u32],width:usize,block:usize){
    let mut diff=0;
    for c in 0..2 {for k in 0..width/block {for t in 0..256 {for x in 0..block {
        let ai=(((c*(width/block)+k)*256+t)*block)+x;let ei=(c*256+t)*width+k*block+x;
        if actual[ai]!=expected[ei]{if diff<8{eprintln!("blocked_diff name={name} c={c} k={k} t={t} x={x} actual={} expected={}",actual[ai],expected[ei]);}diff+=1;}
    }}}}
    println!("blocked_layout name={name} checked={} bitdiff={diff}",actual.len());assert_eq!(diff,0);
}

#[tokio::main]
async fn main() {
    let collector=npu_profile::Collector::default();
    tracing::subscriber::set_global_default(collector.clone()).unwrap();
    let mut device=Device::new(ops_native_fp8_p512::quantize_hidden.topology()).unwrap();
    let values:Vec<bf16>=(0..P512::SIZE*H::SIZE).map(|i| {
        let row=i/H::SIZE;
        bf16::from_f32(if row%128==0 {0.0} else if row%128==1 {if i%2==0 {2f32.powi(-127)} else {-2f32.powi(-127)}} else {
        (((word(i,11)>>40) as f32 / 16777216.0)*2.0-1.0)*4.0*2f32.powi(-((word(i,13)%9)as i32))})
    }).collect();
    let input:HbmTensor<bf16,Chip,m![P512,H]>=HostTensor::<bf16,m![P512,H]>::from_vec(values.iter().copied()).to_hbm(&mut device.pdma).await.unwrap();
    let mut controls=Vec::new();let mut control_outputs=Vec::new();
    for part in 0..4 {
        controls.push(HostTensor::<bf16,m![P128,H]>::from_vec(values[part*P128::SIZE*H::SIZE..(part+1)*P128::SIZE*H::SIZE].iter().copied()).to_hbm(&mut device.pdma).await.unwrap());
        control_outputs.push(HostTensor::<bf16,m![P128,H]>::zero().to_hbm(&mut device.pdma).await.unwrap());
    }
    let mut output:HbmTensor<bf16,Chip,m![P512,H]>=upload(&mut device,vec![0;P512::SIZE*H::SIZE*2]).await;
    let mut qinput:HbmTensor<f8e4m3,Chip,m![P512,H]>=upload(&mut device,vec![0;P512::SIZE*H::SIZE]).await;
    let mut qhidden:HbmTensor<f8e4m3,Chip,m![P512,L]>=upload(&mut device,vec![0;P512::SIZE*L::SIZE]).await;
    let mut xscale:HbmTensor<f32,Chip,m![P512]>=upload(&mut device,vec![0;P512::SIZE*4]).await;
    let mut hscale:HbmTensor<f32,Chip,m![P512]>=upload(&mut device,vec![0;P512::SIZE*4]).await;
    let mut up_result:HbmTensor<bf16,Chip,m![P512,L]>=upload(&mut device,vec![0;P512::SIZE*L::SIZE*2]).await;
    let mut gate_result:HbmTensor<bf16,Chip,m![P512,L]>=upload(&mut device,vec![0;P512::SIZE*L::SIZE*2]).await;
    let mut iq:HbmTensor<f8e4m3,Chip,P512H960>=upload(&mut device,vec![0;P512::SIZE*H::SIZE]).await;
    let mut hq:HbmTensor<f8e4m3,Chip,P512L960>=upload(&mut device,vec![0;P512::SIZE*L::SIZE]).await;
    let mut ub:HbmTensor<bf16,Chip,P512L240>=upload(&mut device,vec![0;P512::SIZE*L::SIZE*2]).await;
    let mut gb:HbmTensor<bf16,Chip,P512L240>=upload(&mut device,vec![0;P512::SIZE*L::SIZE*2]).await;
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

    macro_rules! staged {
        () => {{
            launch(ops_native_fp8_p512::quantize_hidden,(&mut device,&input,&mut qinput,&mut xscale)).await.unwrap();
            launch(ops_native_fp8_p512::project_up_scaled,(&mut device,&qinput,&xscale,&bu,&su,&mut up_result)).await.unwrap();
            launch(ops_native_fp8_p512::project_up_scaled,(&mut device,&qinput,&xscale,&bg,&sg,&mut gate_result)).await.unwrap();
            launch(ops_native_fp8_p512::geglu_quantize,(&mut device,&up_result,&gate_result,&ug,&gg,&mut qhidden,&mut hscale)).await.unwrap();
            launch(ops_native_fp8_p512::project_down_scaled,(&mut device,&qhidden,&hscale,&bd,&sd,&dg,&mut output)).await.unwrap();
        }};
    }
    macro_rules! one_stage {
        ($variant:expr,$stage:expr) => {{match ($variant,$stage) {
            (1,0)=>launch(ops_native_fp8_p512::quantize_hidden,(&mut device,&input,&mut qinput,&mut xscale)).await.unwrap(),
            (1,1)=>launch(ops_native_fp8_p512::project_up_scaled,(&mut device,&qinput,&xscale,&bu,&su,&mut up_result)).await.unwrap(),
            (1,2)=>launch(ops_native_fp8_p512::project_up_scaled,(&mut device,&qinput,&xscale,&bg,&sg,&mut gate_result)).await.unwrap(),
            (1,3)=>launch(ops_native_fp8_p512::geglu_quantize,(&mut device,&up_result,&gate_result,&ug,&gg,&mut qhidden,&mut hscale)).await.unwrap(),
            (1,4)=>launch(ops_native_fp8_p512::project_down_scaled,(&mut device,&qhidden,&hscale,&bd,&sd,&dg,&mut output)).await.unwrap(),
            (2,0)=>launch(ops_native_fp8_p512_inputblocks::quantize_hidden,(&mut device,&input,&mut iq,&mut xscale)).await.unwrap(),
            (2,1)=>launch(ops_native_fp8_p512_inputblocks::project_up_scaled,(&mut device,&iq,&xscale,&bu,&su,&mut up_result)).await.unwrap(),
            (2,2)=>launch(ops_native_fp8_p512_inputblocks::project_up_scaled,(&mut device,&iq,&xscale,&bg,&sg,&mut gate_result)).await.unwrap(),
            (2,3)=>launch(ops_native_fp8_p512_inputblocks::geglu_quantize,(&mut device,&up_result,&gate_result,&ug,&gg,&mut hq,&mut hscale)).await.unwrap(),
            (2,4)=>launch(ops_native_fp8_p512_inputblocks::project_down_scaled,(&mut device,&hq,&hscale,&bd,&sd,&dg,&mut output)).await.unwrap(),
            (3,0)=>launch(ops_native_fp8_p512_blocks::quantize_hidden,(&mut device,&input,&mut iq,&mut xscale)).await.unwrap(),
            (3,1)=>launch(ops_native_fp8_p512_blocks::project_up_scaled,(&mut device,&iq,&xscale,&bu,&su,&mut ub)).await.unwrap(),
            (3,2)=>launch(ops_native_fp8_p512_blocks::project_up_scaled,(&mut device,&iq,&xscale,&bg,&sg,&mut gb)).await.unwrap(),
            (3,3)=>launch(ops_native_fp8_p512_blocks::geglu_quantize,(&mut device,&ub,&gb,&ug,&gg,&mut hq,&mut hscale)).await.unwrap(),
            (3,4)=>launch(ops_native_fp8_p512_blocks::project_down_scaled,(&mut device,&hq,&hscale,&bd,&sd,&dg,&mut output)).await.unwrap(),
            _=>unreachable!(),
        }}};
    }
    macro_rules! full {($variant:expr)=>{{if $variant==0 {for part in 0..4 {
        launch(ops_native_fp8_p128_local_control::batch_ffn_partitioned_input,(&mut device,&controls[part],&bu,&bg,&bd,&su,&sg,&sd,&ug,&gg,&dg,&mut control_outputs[part])).await.unwrap();
    }}else{for stage in 0..5{one_stage!($variant,stage);}}}};}
    let mut reference=Vec::new();
    for part in 0..4 {
        launch(ops_native_fp8_p128_local_control::batch_ffn_partitioned_input,(&mut device,&controls[part],&bu,&bg,&bd,&su,&sg,&sd,&ug,&gg,&dg,&mut control_outputs[part])).await.unwrap();
        let host:HostTensor<bf16,m![P128,H]>=control_outputs[part].to_host(&mut device.pdma).await.unwrap();reference.extend(host.into_vec());
    }
    staged!();
    let candidate:HostTensor<bf16,m![P512,H]>=output.to_host(&mut device.pdma).await.unwrap();let candidate=candidate.into_vec();
    let differences=reference.iter().zip(&candidate).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();
    println!("P512_staged_vs_fourP128 checked={} bit_differences={differences}",candidate.len());
    for (i,(a,b)) in reference.iter().zip(&candidate).enumerate().filter(|(_, (a,b))|a.to_bits()!=b.to_bits()).take(20){eprintln!("difference index={i} reference={} candidate={}",a.to_f32(),b.to_f32());}
    assert_eq!(differences,0);

    let iq_ref:HostTensor<f8e4m3,m![P512,H]>=qinput.to_host(&mut device.pdma).await.unwrap();let iq_ref:Vec<u32>=iq_ref.into_vec().iter().map(|v|v.to_f32().to_bits()).collect();
    let hq_ref:HostTensor<f8e4m3,m![P512,L]>=qhidden.to_host(&mut device.pdma).await.unwrap();let hq_ref:Vec<u32>=hq_ref.into_vec().iter().map(|v|v.to_f32().to_bits()).collect();
    let up_ref:HostTensor<bf16,m![P512,L]>=up_result.to_host(&mut device.pdma).await.unwrap();let up_ref:Vec<u32>=up_ref.into_vec().iter().map(|v|v.to_bits()as u32).collect();
    let gate_ref:HostTensor<bf16,m![P512,L]>=gate_result.to_host(&mut device.pdma).await.unwrap();let gate_ref:Vec<u32>=gate_ref.into_vec().iter().map(|v|v.to_bits()as u32).collect();
    for variant in [2,3] {
        full!(variant);
        let result:HostTensor<bf16,m![P512,H]>=output.to_host(&mut device.pdma).await.unwrap();let result=result.into_vec();
        let differences=reference.iter().zip(&result).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();
        println!("P512_blocked_FFN variant={variant} checked={} bit_differences={differences}",result.len());assert_eq!(differences,0);
        let aq:HostTensor<f8e4m3,P512H960>=iq.to_host(&mut device.pdma).await.unwrap();let aq:Vec<u32>=aq.into_vec().iter().map(|v|v.to_f32().to_bits()).collect();check_blocked("inputH960",&aq,&iq_ref,H::SIZE,960);
        let aq:HostTensor<f8e4m3,P512L960>=hq.to_host(&mut device.pdma).await.unwrap();let aq:Vec<u32>=aq.into_vec().iter().map(|v|v.to_f32().to_bits()).collect();check_blocked("hiddenL960",&aq,&hq_ref,L::SIZE,960);
        if variant==3 {
            let aq:HostTensor<bf16,P512L240>=ub.to_host(&mut device.pdma).await.unwrap();let aq:Vec<u32>=aq.into_vec().iter().map(|v|v.to_bits()as u32).collect();check_blocked("upL240",&aq,&up_ref,L::SIZE,240);
            let aq:HostTensor<bf16,P512L240>=gb.to_host(&mut device.pdma).await.unwrap();let aq:Vec<u32>=aq.into_vec().iter().map(|v|v.to_bits()as u32).collect();check_blocked("gateL240",&aq,&gate_ref,L::SIZE,240);
        }
    }
    let mut times=[Vec::<f64>::new(),Vec::<f64>::new(),Vec::<f64>::new(),Vec::<f64>::new()];
    for variant in [0,1,2,3,3,2,1,0] {
        let start=std::time::Instant::now();for _ in 0..10 {full!(variant);}
        times[variant].push(start.elapsed().as_secs_f64()*1000.0/10.0);
    }
    println!("P512_blocked_FFN four_current_P128_ms={:?} rowmajor_ms={:?} input_blocks_ms={:?} input_output_blocks_ms={:?} scope=10batches_completed_launches_no_host_read",times[0],times[1],times[2],times[3]);
    for variant in [1,2,3] {full!(variant);for stage in [0,1,2,3,4,4,3,2,1,0] {
        let start=std::time::Instant::now();for _ in 0..10 {one_stage!(variant,stage);}
        println!("P512_blocked_stage variant={variant} stage={stage} ms={} scope=10_completed_launches_no_host_read",start.elapsed().as_secs_f64()*1000.0/10.0);
    }}
    if std::env::var_os("FURIOSA_OPT_PROFILE").is_some() {for variant in [0,1,2,3] {
        println!("PROFILE_VARIANT {variant}");collector.enable(true);full!(variant);collector.enable(false);
    }}
}
