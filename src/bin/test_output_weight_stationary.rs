//! Complete four-P128 O oracle and wall-time comparison, with original BF16 input.
#[path = "support/npu_profile.rs"] mod npu_profile;
thread_local! { static PROFILE: npu_profile::Collector = npu_profile::Collector::default(); }
use furiosa_opt_gemma4::{Chip, axes::*, ops_output_weight_stationary as op,
    ops_output_p128_h64_blocked as control};
use furiosa_opt_std::prelude::*;

fn hash(i: usize, seed: usize) -> usize {
    let mut v=(i as u64).wrapping_add((seed as u64).wrapping_mul(0x9e3779b97f4a7c15));
    v=(v^(v>>30)).wrapping_mul(0xbf58476d1ce4e5b9);
    v=(v^(v>>27)).wrapping_mul(0x94d049bb133111eb);
    (v^(v>>31)) as usize
}
fn input_value(g: usize, row: usize, k: usize, seed: usize) -> bf16 {
    let value = if row == 0 { 0.0 } else {
        let code=if seed==0 {row*17+k*13+g*19} else {hash(row*4096+k,seed+g*137)};
        (code % 31) as f32 / 32.0 - 15.0 / 32.0
    };
    bf16::from_f32(value * [0.5, 1.0, 2.0, 4.0][g])
}
fn blocked_index(row: usize, col: usize) -> usize {
    (((row / 64) * 60 + col / 64) * 64 + row % 64) * 64 + col % 64
}
fn provided(layer: usize) -> (Vec<f8e4m3>, Vec<bf16>) {
    assert_ne!(layer % 6, 5);
    let file = std::fs::File::open(furiosa_opt_gemma4::host::load::model_dir().unwrap().join("model.safetensors")).unwrap();
    let map = unsafe { memmap2::Mmap::map(&file).unwrap() };
    let tensors = safetensors::SafeTensors::deserialize(&map).unwrap();
    let prefix = format!("model.language_model.layers.{layer}.self_attn.o_proj");
    let weights = HostTensor::<f8e4m3,m![H,Qs]>::from_safetensors(&tensors.tensor(&format!("{prefix}.weight")).unwrap()).unwrap().into_vec();
    let scales = tensors.tensor(&format!("{prefix}.weight_scale")).unwrap();
    assert_eq!(scales.shape(), [H::SIZE,1]);
    assert_eq!(scales.dtype(), safetensors::Dtype::BF16);
    (weights, HostTensor::<bf16,m![H]>::from_buf(scales.data().to_vec()).into_vec())
}
async fn read<E:M>(d: &mut Device, x: &HbmTensor<bf16,Chip,E>) -> Vec<bf16> {
    let h: HostTensor<bf16,E> = x.to_host(&mut d.pdma).await.unwrap(); h.into_vec()
}
macro_rules! call { ($d:ident,$fun:path,$inputs:ident,$weights:ident,$scales:ident,$outputs:ident,$order:ident) => {{
    let [a,b,c,e] = $outputs.as_mut_slice() else { unreachable!() };
    launch($fun,(&mut $d,&$inputs[$order[0]],&$inputs[$order[1]],&$inputs[$order[2]],&$inputs[$order[3]],&$weights,&$scales,a,b,c,e)).await.unwrap();
}}; }
macro_rules! call_views { ($d:ident,$fun:path,$inputs:ident,$weights:ident,$scales:ident,$outputs:ident,$order:ident) => {{
    let [a,b,c,e] = $outputs.as_mut_slice() else { unreachable!() };
    launch($fun,(&mut $d,&$inputs[$order[0]],&$inputs[$order[1]],&$inputs[$order[2]],&$inputs[$order[3]],&$weights,&$scales,a.view_mut(),b.view_mut(),c.view_mut(),e.view_mut())).await.unwrap();
}}; }
async fn fixture(case: usize) {
    let mut d = Device::new(op::project_sliding_output_four.topology()).unwrap();
    let (cpu_w,cpu_s) = if case < 3 {
        ((0..H::SIZE*Qs::SIZE).map(|i| f8e4m3::from_f32((if case==0 {(i/Qs::SIZE*5+i%Qs::SIZE*7)%17} else {hash(i,case+739)%17}) as f32/8.0-1.0)).collect(),
         (0..H::SIZE).map(|i|bf16::from_f32([0.0,0.5,-0.5,1.0,-1.0,1.5,2.0][i%7])).collect())
    } else { provided([0usize,6,46][case-3]) };
    let weights = HostTensor::<f8e4m3,m![H,Qs]>::from_vec(cpu_w.clone()).to_hbm(&mut d.pdma).await.unwrap();
    let scales = HostTensor::<bf16,m![H]>::from_vec(cpu_s.clone()).to_hbm(&mut d.pdma).await.unwrap();
    let mut cpu_inputs = Vec::new();
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut references = Vec::new();
    for g in 0..4 {
        let values: Vec<bf16> = (0..P128::SIZE*Qs::SIZE).map(|i|input_value(g,i/Qs::SIZE,i%Qs::SIZE,case)).collect();
        inputs.push(HostTensor::<bf16,m![P128,Ns,Gs,Ds]>::from_vec(values.clone()).to_hbm(&mut d.pdma).await.unwrap());
        cpu_inputs.push(values);
        outputs.push(HostTensor::<bf16,op::Output>::from_vec(vec![bf16::from_f32(f32::NAN);128*3840]).to_hbm(&mut d.pdma).await.unwrap());
        references.push(HostTensor::<bf16,control::Output>::from_vec(vec![bf16::from_f32(f32::NAN);128*3840]).to_hbm(&mut d.pdma).await.unwrap());
    }
    for order in [[0usize,1,2,3],[3,1,0,2]] {
        call_views!(d,op::project_sliding_output_four,inputs,weights,scales,outputs,order);
        call!(d,control::project_sliding_output_four,inputs,weights,scales,references,order);
        let mut differences=0usize;
        let mut cpu_checks=0usize;
        for g in 0..4 {
            let got=read(&mut d,&outputs[g]).await;
            let expected=read(&mut d,&references[g]).await;
            for row in 0..128 { for col in 0..3840 {
                let i=blocked_index(row,col);
                if got[i].to_bits()!=expected[blocked_index(row,col)].to_bits() {
                    if differences<8 {println!("WEIGHT_STATIONARY_DIFF case={case} group={g} row={row} col={col} got={} expected={}",got[i].to_f32(),expected[blocked_index(row,col)].to_f32());}
                    differences+=1;
                }
            }}
            // Bounded dyadics make the synthetic dot exactly representable in F32;
            // independent F64 products and the explicit BF16 tree verify the math.
            if case<3 {
                for row in [0,1,13,63,64,127] { for col in [0,1,7,8,119,120,1919,1920,3839] {
                    let mut partial=[0f32;4];
                    for block in 0..4 {
                        let mut sum=0f64;
                        for k in block*1024..(block+1)*1024 {sum+=cpu_inputs[order[g]][row*4096+k].to_f32() as f64*cpu_w[col*4096+k].to_f32() as f64;}
                        partial[block]=bf16::from_f32(sum as f32).to_f32();
                    }
                    let left=bf16::from_f32(partial[0]+partial[1]).to_f32();
                    let right=bf16::from_f32(partial[2]+partial[3]).to_f32();
                    let sum=bf16::from_f32(left+right).to_f32();
                    let result=bf16::from_f32(sum*cpu_s[col].to_f32());
                    assert_eq!(got[blocked_index(row,col)].to_bits(),result.to_bits(),"independent CPU sample case={case} group={g} row={row} col={col}");
                    cpu_checks+=1;
                }}
            }
        }
        println!("WEIGHT_STATIONARY_ORACLE case={case} order={order:?} values={} differences={differences} cpu_checks={cpu_checks}",4*128*3840);
        assert_eq!(differences,0,"original K1024 BF16 tree exact oracle");
    }
    let order=[0usize,1,2,3];
    for (round,candidate) in [false,true,true,false].into_iter().enumerate() {
        for _ in 0..3 {
            if candidate {call_views!(d,op::project_sliding_output_four,inputs,weights,scales,outputs,order);}
            else {call!(d,control::project_sliding_output_four,inputs,weights,scales,references,order);}
        }
        let start=std::time::Instant::now();
        for _ in 0..16 {
            if candidate {call_views!(d,op::project_sliding_output_four,inputs,weights,scales,outputs,order);}
            else {call!(d,control::project_sliding_output_four,inputs,weights,scales,references,order);}
        }
        println!("WEIGHT_STATIONARY_TIME case={case} round={round} candidate={candidate} complete_four_us={:.6}",start.elapsed().as_secs_f64()*1e6/16.0);
    }
    for g in 0..4 {assert!(read(&mut d,&inputs[g]).await.iter().zip(&cpu_inputs[g]).all(|(a,b)|a.to_bits()==b.to_bits()),"input mutated");}
    let reread: HostTensor<f8e4m3,m![H,Qs]>=weights.to_host(&mut d.pdma).await.unwrap();
    assert!(reread.into_vec().iter().zip(&cpu_w).all(|(a,b)|a.to_f32().to_bits()==b.to_f32().to_bits()),"weight mutated");
    assert!(read(&mut d,&scales).await.iter().zip(&cpu_s).all(|(a,b)|a.to_bits()==b.to_bits()),"scales mutated");
    for candidate in [false,true] {
        println!("WEIGHT_STATIONARY_PROFILE case={case} candidate={candidate}");
        PROFILE.with(|p|p.enable(true));
        if candidate {call_views!(d,op::project_sliding_output_four,inputs,weights,scales,outputs,order);}
        else {call!(d,control::project_sliding_output_four,inputs,weights,scales,references,order);}
        PROFILE.with(|p|p.enable(false));
    }
    println!("WEIGHT_STATIONARY_PASS case={case} immutable=true original_tree_exact=true");
}
#[tokio::main]
async fn main() {
    PROFILE.with(|p|tracing::subscriber::set_global_default(p.clone()).unwrap());
    let args:Vec<_>=std::env::args().collect();
    let first:usize=args.get(1).map(|s|s.parse().unwrap()).unwrap_or(0);
    let end:usize=args.get(2).map(|s|s.parse().unwrap()).unwrap_or(6);
    assert!(first<end && end<=6);
    for case in first..end {fixture(case).await;}
}
