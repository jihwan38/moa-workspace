use furiosa_opt_gemma4::{Chip,axes::*,ops_output_p128_four as op,ops_output_p128_parallel_gather as control};
use furiosa_opt_std::prelude::*;
fn hash(i:usize,seed:usize)->usize {let mut x=(i as u64).wrapping_add(seed as u64);x=(x^(x>>30)).wrapping_mul(0xbf58476d1ce4e5b9);x=(x^(x>>27)).wrapping_mul(0x94d049bb133111eb);(x^(x>>31))as usize}
async fn read<E:M>(device:&mut Device,x:&HbmTensor<bf16,Chip,E>)->Vec<bf16>{let h:HostTensor<bf16,E>=x.to_host(&mut device.pdma).await.unwrap();h.into_vec()}
macro_rules! fixture {($device:ident,$seed:ident,$input:ty,$axis:ident,$single:ident,$four:ident)=>{{
    let mut inputs=Vec::<HbmTensor<bf16,Chip,$input>>::new();
    let mut outputs=Vec::<HbmTensor<bf16,Chip,m![P128,H]>>::new();
    for row in 0..4 {
        let x:Vec<_>=(0..P128::SIZE*$axis::SIZE).map(|i|bf16::from_f32(if row==0&&i<$axis::SIZE {0.0}else{(hash(i,$seed+row*137)%255)as f32/128.0-127.0/128.0})).collect();
        inputs.push(HostTensor::<bf16,$input>::from_vec(x).to_hbm(&mut $device.pdma).await.unwrap());
        outputs.push(HostTensor::<bf16,m![P128,H]>::from_vec(vec![bf16::from_f32(f32::NAN);P128::SIZE*H::SIZE]).to_hbm(&mut $device.pdma).await.unwrap());
    }
    let weights=HostTensor::<f8e4m3,m![H,$axis]>::from_vec((0..H::SIZE*$axis::SIZE).map(|i|f8e4m3::from_f32((hash(i,$seed+739)%31)as f32/16.0-15.0/16.0)).collect::<Vec<_>>()).to_hbm(&mut $device.pdma).await.unwrap();
    let scales=HostTensor::<bf16,m![H]>::from_vec((0..H::SIZE).map(|i|bf16::from_f32([0.0,0.5,-0.5,1.0,-1.0,1.5,2.0][i%7])).collect::<Vec<_>>()).to_hbm(&mut $device.pdma).await.unwrap();
    for order in [[0,1,2,3],[3,1,0,2]] {
        let mut expected=Vec::new();
        for row in 0..4 {launch(control::$single,(&mut $device,&inputs[order[row]],&weights,&scales,&mut outputs[row])).await.unwrap();expected.push(read(&mut $device,&outputs[row]).await);}
        let [a,b,c,d]=outputs.as_mut_slice() else {unreachable!()};
        launch(op::$four,(&mut $device,&inputs[order[0]],&inputs[order[1]],&inputs[order[2]],&inputs[order[3]],&weights,&scales,a,b,c,d)).await.unwrap();
        let mut differences=0;
        for row in 0..4 {let actual=read(&mut $device,&outputs[row]).await;differences+=actual.iter().zip(&expected[row]).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();}
        println!("OUTPUT_FOUR axis={} seed={} order={order:?} checked={} differences={differences}",stringify!($axis),$seed,4*P128::SIZE*H::SIZE);assert_eq!(differences,0);
    }
    if $seed==31 {
        for variant in [0,1,1,0] {
            let start=std::time::Instant::now();
            for _ in 0..10 {
                if variant==0 {for row in 0..4 {launch(control::$single,(&mut $device,&inputs[row],&weights,&scales,&mut outputs[row])).await.unwrap();}}
                else {let[a,b,c,d]=outputs.as_mut_slice() else {unreachable!()};launch(op::$four,(&mut $device,&inputs[0],&inputs[1],&inputs[2],&inputs[3],&weights,&scales,a,b,c,d)).await.unwrap();}
            }
            println!("OUTPUT_FOUR_ABBA axis={} variant={variant} ms={} scope=10completedsets_nohostread",stringify!($axis),start.elapsed().as_secs_f64()*1000.0/10.0);
        }
    }
}};}
#[tokio::main]
async fn main(){
    let mut device=Device::new(op::project_sliding_output_four.topology()).unwrap();
    for seed in [31,79] {
        fixture!(device,seed,m![P128,Ns,Gs,Ds],Qs,project_sliding_output_batch,project_sliding_output_four);
        fixture!(device,seed,m![P128,Gf,Df],Qf,project_full_output_batch,project_full_output_four);
    }
}
