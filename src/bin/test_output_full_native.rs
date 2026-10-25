//! Fused per-K activation quantization: independent CPU, roundtrip and completed four-group timing.
#[path = "support/npu_profile.rs"]
mod npu_profile;
thread_local! {static PROFILE:npu_profile::Collector=npu_profile::Collector::default();}
use furiosa_opt_gemma4::{
    Chip, axes::*, ops_output_full_native as op, ops_output_p128_four_keyrings as control,
};
use furiosa_opt_std::prelude::*;
fn hash(i: usize, seed: usize) -> usize {
    let mut v = (i as u64).wrapping_add(seed as u64);
    v = (v ^ (v >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    v = (v ^ (v >> 27)).wrapping_mul(0x94d049bb133111eb);
    (v ^ (v >> 31)) as usize
}
fn value(row: usize, k: usize, seed: usize, group: usize) -> bf16 {
    let x = if row == 0 {
        0.0
    } else if row == 1 {
        if k % 2 == 0 { 1e-30 } else { -1e-30 }
    } else if seed == 0 {
        ((row * 17 + k * 13 + group * 19) % 31) as f32 / 16.0 - 15.0 / 16.0
    } else {
        (hash(row * 8192 + k, seed + group * 137) % 1023) as f32 / 512.0 - 511.0 / 512.0
    };
    bf16::from_f32(x * [0.25, 1.0, 4.0, 16.0, 0.5, 2.0, 8.0, 32.0][k / 1024])
}
fn weight(n: usize, k: usize, seed: usize) -> f8e4m3 {
    f8e4m3::from_f32(if seed == 0 {
        ((n * 5 + k * 7) % 17) as f32 / 8.0 - 1.0
    } else {
        (hash(n * 8192 + k, seed + 739) % 31) as f32 / 16.0 - 15.0 / 16.0
    })
}
fn channel(n: usize) -> f32 {
    [0.0, 0.5, -0.5, 1.0, -1.0, 1.5, 2.0][n % 7]
}
fn quant_scale(row: &[bf16]) -> f32 {
    let max = row
        .iter()
        .map(|v| v.to_f32().abs())
        .fold(1.925929944387236e-34f32, f32::max);
    f32::from_bits(max.to_bits() & 0x7f80_0000) * 0.0078125
}
async fn read<E: M>(d: &mut Device, x: &HbmTensor<bf16, Chip, E>) -> Vec<bf16> {
    let h: HostTensor<bf16, E> = x.to_host(&mut d.pdma).await.unwrap();
    h.into_vec()
}
async fn read_output(d: &mut Device, x: &HbmTensor<bf16, Chip, op::Output>) -> Vec<bf16> {
    read(d, x).await
}

fn provided_weights(layer: usize) -> (Vec<f8e4m3>, Vec<bf16>) {
    assert_eq!(layer % 6, 5);
    let path = furiosa_opt_gemma4::host::load::model_dir()
        .unwrap()
        .join("model.safetensors");
    let file = std::fs::File::open(path).unwrap();
    let map = unsafe { memmap2::Mmap::map(&file).unwrap() };
    let tensors = safetensors::SafeTensors::deserialize(&map).unwrap();
    let prefix = format!("model.language_model.layers.{layer}.self_attn.o_proj");
    let w = tensors.tensor(&format!("{prefix}.weight")).unwrap();
    let weights = HostTensor::<f8e4m3, m![H, Qf]>::from_safetensors(&w)
        .unwrap()
        .into_vec();
    let scale = tensors.tensor(&format!("{prefix}.weight_scale")).unwrap();
    assert_eq!(scale.dtype(), safetensors::Dtype::BF16);
    assert_eq!(scale.shape(), [H::SIZE, 1]);
    let scales = HostTensor::<bf16, m![H]>::from_buf(scale.data().to_vec()).into_vec();
    println!(
        "PROVIDED_FULL_O layer={layer} weights={} scales={}",
        weights.len(),
        scales.len()
    );
    (weights, scales)
}
macro_rules! fixture {($d:ident,$seed:ident,$axis:ident,$shape:ty,$quant:ident,$four:ident,$partials:ident)=>{{
    let mut original=Vec::<HbmTensor<bf16,Chip,$shape>>::new();
    let mut quantized=Vec::<HbmTensor<f8e4m3,Chip,m![P128,$axis]>>::new();
    let mut scales=Vec::<HbmTensor<f32,Chip,m![P128,$axis/1024,1#8]>>::new();
    let mut roundtrip=Vec::<HbmTensor<bf16,Chip,$shape>>::new();
    let mut outputs=Vec::<HbmTensor<bf16,Chip,op::Output>>::new();
    let mut reference_outputs=Vec::<HbmTensor<bf16,Chip,op::Output>>::new();
    let mut cpu_q=Vec::<Vec<f32>>::new();let mut cpu_scale=Vec::<Vec<f32>>::new();
    for group in 0..4 {
        let x:Vec<_>=(0..P128::SIZE).flat_map(|r|(0..$axis::SIZE).map(move|k|value(r,k,$seed,group))).collect();
        let ss:Vec<_>=x.chunks_exact(1024).map(quant_scale).collect();
        let q:Vec<_>=x.iter().enumerate().map(|(i,v)|f8e4m3::from_f32(v.to_f32()/ss[i/1024])).collect();
        let rt:Vec<_>=q.iter().enumerate().map(|(i,v)|bf16::from_f32(v.to_f32()*ss[i/1024])).collect();
        let flat=HostTensor::<bf16,$shape>::from_vec(x).to_hbm(&mut $d.pdma).await.unwrap();
        let mut qdev=HostTensor::<f8e4m3,m![P128,$axis]>::from_buf(vec![0x7f;P128::SIZE*$axis::SIZE]).to_hbm(&mut $d.pdma).await.unwrap();
        let mut sdev=HostTensor::<f32,m![P128,$axis/1024,1#8]>::from_buf(vec![0x7f;P128::SIZE*($axis::SIZE/1024)*8*4]).to_hbm(&mut $d.pdma).await.unwrap();
        launch(op::$quant,(&mut $d,&flat,&mut qdev,&mut sdev)).await.unwrap();
        let gotq:HostTensor<f8e4m3,m![P128,$axis]>=qdev.to_host(&mut $d.pdma).await.unwrap();
        let gots:HostTensor<f32,m![P128,$axis/1024,1#8]>=sdev.to_host(&mut $d.pdma).await.unwrap();
        let gotq=gotq.into_vec();let gots=gots.into_vec();
        assert_eq!(gotq.len(),q.len());assert_eq!(gots.len(),P128::SIZE*($axis::SIZE/1024)*8);
        let qdiff=q.iter().zip(&gotq).filter(|(a,b)|a.to_f32().to_bits()!=b.to_f32().to_bits()).count();
        let sdiff=ss.iter().enumerate().filter(|(r,s)|s.to_bits()!=gots[r*8].to_bits()).count();
        println!("O_NATIVE_QUANT axis={} seed={} group={group} bytes={} qdiff={qdiff} sdiff={sdiff}",stringify!($axis),$seed,q.len());assert_eq!(qdiff,0);assert_eq!(sdiff,0);
        original.push(flat);quantized.push(qdev);scales.push(sdev);
        roundtrip.push(HostTensor::<bf16,$shape>::from_vec(rt).to_hbm(&mut $d.pdma).await.unwrap());
        outputs.push(HostTensor::<bf16,op::Output>::from_vec(vec![bf16::from_f32(f32::NAN);P128::SIZE*H::SIZE]).to_hbm(&mut $d.pdma).await.unwrap());
        reference_outputs.push(HostTensor::<bf16,op::Output>::from_vec(vec![bf16::from_f32(f32::NAN);P128::SIZE*H::SIZE]).to_hbm(&mut $d.pdma).await.unwrap());
        cpu_q.push(q.iter().map(|v|v.to_f32()).collect());cpu_scale.push(ss);
    }
    let (cpu_weights,cpu_channels)=if $seed>=1000 {provided_weights($seed-1000)} else {
        ((0..H::SIZE).flat_map(|n|(0..$axis::SIZE).map(move|k|weight(n,k,$seed))).collect(),(0..H::SIZE).map(|n|bf16::from_f32(channel(n))).collect())
    };
    let weights=HostTensor::<f8e4m3,m![H,$axis]>::from_vec(cpu_weights.clone()).to_hbm(&mut $d.pdma).await.unwrap();
    let ws=HostTensor::<bf16,m![H]>::from_vec(cpu_channels.clone()).to_hbm(&mut $d.pdma).await.unwrap();
    for order in [[0,1,2,3],[3,1,0,2]] {
        // Candidate executes first into independent NaN outputs.
        let [a,b,c,d]=outputs.as_mut_slice() else{unreachable!()};
        launch(op::$four,(&mut $d,&original[order[0]],&original[order[1]],&original[order[2]],&original[order[3]],&weights,&ws,a,b,c,d)).await.unwrap();
        let mut actual=Vec::new();for out in &outputs {actual.push(read_output(&mut $d,out).await);}
        let [a,b,c,d]=reference_outputs.as_mut_slice() else{unreachable!()};
        launch(control::$four,(&mut $d,&roundtrip[order[0]],&roundtrip[order[1]],&roundtrip[order[2]],&roundtrip[order[3]],&weights,&ws,a,b,c,d)).await.unwrap();
        let mut reference=Vec::new();for out in &reference_outputs {reference.push(read_output(&mut $d,out).await);}
        assert!(actual.iter().flatten().all(|x|x.to_f32().is_finite()));
        let diff:usize=actual.iter().zip(&reference).map(|(a,b)|a.iter().zip(b).filter(|(x,y)|x.to_bits()!=y.to_bits()).count()).sum();
        println!("O_NATIVE_ROUNDTRIP axis={} seed={} order={order:?} values={} differences={diff}",stringify!($axis),$seed,4*P128::SIZE*H::SIZE);assert_eq!(diff,0);
        let mut cpu_diff=0usize;let mut cpu_count=0usize;let mut partial_diff=0usize;let mut partial_count=0usize;
        let mut raw_partials=Vec::new();
        let mut raw=HostTensor::<bf16,m![P128,$axis/1024,H]>::from_vec(vec![bf16::from_f32(f32::NAN);P128::SIZE*($axis::SIZE/1024)*H::SIZE]).to_hbm(&mut $d.pdma).await.unwrap();
        for g in 0..4 {launch(op::$partials,(&mut $d,&original[order[g]],&weights,&mut raw)).await.unwrap();raw_partials.push(read(&mut $d,&raw).await);}
        for g in 0..4 {for r in 0..P128::SIZE {
            let columns:Vec<_>=if $seed==0{(0..17).collect()}else{vec![0,1,59,60,119,120,3838,3839]};
            let mut cpu=Vec::new();for &n in &columns {
                let mut partials:Vec<f32>=(0..$axis::SIZE/1024).map(|c|{
                    let dot=(c*1024..(c+1)*1024).map(|k|cpu_q[order[g]][r*$axis::SIZE+k]as f64*cpu_weights[n*$axis::SIZE+k].to_f32()as f64).sum::<f64>();
                    bf16::from_f32((dot as f32)*cpu_scale[order[g]][r*($axis::SIZE/1024)+c]).to_f32()
                }).collect();
                for (c,&expected) in partials.iter().enumerate() {
                    let check_columns:Vec<_>=if $seed==0{(n..H::SIZE).step_by(17).collect()}else{vec![n]};
                    for col in check_columns {partial_count+=1;partial_diff+=usize::from(raw_partials[g][(r*($axis::SIZE/1024)+c)*H::SIZE+col].to_bits()!=bf16::from_f32(expected).to_bits());}
                }
                while partials.len()>1 {partials=partials.chunks_exact(2).map(|p|bf16::from_f32(p[0]+p[1]).to_f32()).collect();}
                cpu.push(partials[0]);
            }
            if $seed==0 {for n in 0..H::SIZE{cpu_count+=1;cpu_diff+=usize::from(actual[g][r*H::SIZE+n].to_bits()!=bf16::from_f32(cpu[n%17]*cpu_channels[n].to_f32()).to_bits());}}
            else {for (j,&n)in columns.iter().enumerate(){cpu_count+=1;cpu_diff+=usize::from(actual[g][r*H::SIZE+n].to_bits()!=bf16::from_f32(cpu[j]*cpu_channels[n].to_f32()).to_bits());}}
        }}
        println!("O_NATIVE_CPU axis={} seed={} order={order:?} values={cpu_count} differences={cpu_diff}",stringify!($axis),$seed);assert_eq!(cpu_diff,0);println!("O_NATIVE_CPU_PARTIAL axis={} seed={} checked={partial_count} differences={partial_diff}",stringify!($axis),$seed);assert_eq!(partial_diff,0);
        if order==[0,1,2,3] {
            let [a,b,c,d]=outputs.as_mut_slice() else{unreachable!()};
            launch(control::$four,(&mut $d,&original[0],&original[1],&original[2],&original[3],&weights,&ws,a,b,c,d)).await.unwrap();
            let mut err=0.0f64;let mut energy=0.0f64;let mut max=0.0f32;let mut changed=0usize;
            for g in 0..4 {let reference=read_output(&mut $d,&outputs[g]).await;for (a,b)in actual[g].iter().zip(reference){let e=a.to_f32()-b.to_f32();err+=(e as f64).powi(2);energy+=(b.to_f32()as f64).powi(2);max=max.max(e.abs());changed+=usize::from(a.to_bits()!=b.to_bits());}}
            println!("O_NATIVE_BF16_ERROR axis={} seed={} values={} changed={changed} relative_l2={} max_abs={max}",stringify!($axis),$seed,4*P128::SIZE*H::SIZE,(err/energy.max(1e-30)).sqrt());
        }
    }
    if $seed==31 || $seed>=1000 {for variant in [1,0,0,1] {
        let start=std::time::Instant::now();for _ in 0..8 {
            if variant==0 {let[a,b,c,d]=outputs.as_mut_slice()else{unreachable!()};launch(control::$four,(&mut $d,&original[0],&original[1],&original[2],&original[3],&weights,&ws,a,b,c,d)).await.unwrap();}
            else {
                let[a,b,c,d]=outputs.as_mut_slice()else{unreachable!()};launch(op::$four,(&mut $d,&original[0],&original[1],&original[2],&original[3],&weights,&ws,a,b,c,d)).await.unwrap();
            }
        }
        println!("O_NATIVE_ABBA axis={} variant={variant} warm_us={} scope=fourgroups_fused_quantization_canonical_output",stringify!($axis),start.elapsed().as_secs_f64()*1e6/8.0);
     }
        println!("TRACE_CURRENT_FULL_CONTROL");PROFILE.with(|p|p.enable(true));
        let[a,b,c,d]=reference_outputs.as_mut_slice()else{unreachable!()};launch(control::$four,(&mut $d,&original[0],&original[1],&original[2],&original[3],&weights,&ws,a,b,c,d)).await.unwrap();PROFILE.with(|p|p.enable(false));
        println!("TRACE_FUSED_NATIVE");PROFILE.with(|p|p.enable(true));
        let[a,b,c,d]=outputs.as_mut_slice()else{unreachable!()};launch(op::$four,(&mut $d,&original[0],&original[1],&original[2],&original[3],&weights,&ws,a,b,c,d)).await.unwrap();PROFILE.with(|p|p.enable(false));
    }
    for group in 0..4 {
        let values=read(&mut $d,&original[group]).await;
        assert!(values.iter().enumerate().all(|(i,x)|x.to_bits()==value(i/$axis::SIZE,i%$axis::SIZE,$seed,group).to_bits()),"original input changed");
    }
    let readw:HostTensor<f8e4m3,m![H,$axis]>=weights.to_host(&mut $d.pdma).await.unwrap();
    assert!(readw.into_vec().iter().zip(&cpu_weights).all(|(a,b)|a.to_f32().to_bits()==b.to_f32().to_bits()),"weight changed");
    assert!(read(&mut $d,&ws).await.iter().zip(&cpu_channels).all(|(a,b)|a.to_bits()==b.to_bits()),"scale changed");
    println!("FULL_O_IMMUTABLE seed={}",$seed);
}};}
#[tokio::main]
async fn main() {
    PROFILE.with(|p| tracing::subscriber::set_global_default(p.clone()).unwrap());
    let mut d = Device::new(op::project_full_output_four.topology()).unwrap();
    for seed in [0, 31, 79, 1005, 1047] {
        fixture!(
            d,
            seed,
            Qf,
            m![P128, Gf, Df],
            quantize_full_input,
            project_full_output_four,
            project_full_partials
        );
    }
    println!(
        "PASS fullO native perK1024 quantization, independent CPU partial/tree and quantized-roundtrip oracle; original BF16 error reported; no model quality claim"
    );
}
