//! Stage diagnostic: seeded quantizer and every replicated input/scale copy.
use furiosa_opt_std::prelude::*;
use furiosa_opt_gemma4::{Chip,axes::{R64,H,L},ops_native_fp8_r64_split_switch};
fn word(i:usize)->u64 {let mut v=(i as u64).wrapping_add(47);v=(v^(v>>30)).wrapping_mul(0xbf58476d1ce4e5b9);v=(v^(v>>27)).wrapping_mul(0x94d049bb133111eb);v^(v>>31)}
macro_rules! check {
 ($device:ident,$axis:ident,$group:ident,$div:literal,$kernel:ident,$salt:expr)=>{{
    let width=$axis::SIZE;println!("INPUT_SALT {} axis={}",$salt,stringify!($axis));
    let vals:Vec<bf16>=(0..R64::SIZE*width).map(|i|bf16::from_f32(if i/width==0{0.}else if i/width==1{if i%2==0{2f32.powi(-127)}else{-2f32.powi(-127)}}else{(((word(i+$salt)>>40)as f32/16777216.)*2.-1.)*4.*2f32.powi(-((word(i+13+$salt)%9)as i32))})).collect();
    let mut expected=Vec::new();let mut scales=Vec::new();
    for row in vals.chunks(width){let amax=row.iter().map(|v|v.to_f32().abs()).fold(0f32,f32::max).max(2f32.powi(-112));let scale=f32::from_bits(amax.to_bits()&0x7f800000)/128.;scales.push(scale);for v in row{let x=v.to_f32();let x=if x.abs()<f32::MIN_POSITIVE{f32::from_bits(x.to_bits()&0x80000000)}else{x};expected.push(f8e4m3::from_f32(x/scale));}}
    let input=HostTensor::<bf16,m![R64,$axis]>::from_vec(vals).to_hbm(&mut $device.pdma).await.unwrap();
    let mut seedq:HbmTensor<f8e4m3,Chip,m![$group/$div,R64,$axis]>=HostTensor::<f8e4m3,m![$group/$div,R64,$axis]>::zero().to_hbm(&mut $device.pdma).await.unwrap();
    let mut seeds:HbmTensor<f32,Chip,m![$group/$div,R64]>=HostTensor::<f32,m![$group/$div,R64]>::zero().to_hbm(&mut $device.pdma).await.unwrap();
    let mut copyq:HbmTensor<f8e4m3,Chip,m![$group/120,R64,$axis]>=HostTensor::<f8e4m3,m![$group/120,R64,$axis]>::zero().to_hbm(&mut $device.pdma).await.unwrap();
    let mut copys:HbmTensor<f32,Chip,m![$group/120,R64]>=HostTensor::<f32,m![$group/120,R64]>::zero().to_hbm(&mut $device.pdma).await.unwrap();
    launch(ops_native_fp8_r64_split_switch::$kernel,(&mut $device,&input,&mut seedq,&mut seeds,&mut copyq,&mut copys)).await.unwrap();
    let q:HostTensor<f8e4m3,m![$group/$div,R64,$axis]>=seedq.to_host(&mut $device.pdma).await.unwrap();let q=q.into_vec();
    let s:HostTensor<f32,m![$group/$div,R64]>=seeds.to_host(&mut $device.pdma).await.unwrap();let s=s.into_vec();
    let cq:HostTensor<f8e4m3,m![$group/120,R64,$axis]>=copyq.to_host(&mut $device.pdma).await.unwrap();let cq=cq.into_vec();
    let cs:HostTensor<f32,m![$group/120,R64]>=copys.to_host(&mut $device.pdma).await.unwrap();let cs=cs.into_vec();
    for(name,actual)in[("seed",&q),("copies",&cq)]{let diff=actual.iter().enumerate().filter(|(i,v)|v.to_f32().to_bits()!=expected[i%expected.len()].to_f32().to_bits()).count();println!("seed_quant axis={} kind={name} checked={} bitdiff={diff}",stringify!($axis),actual.len());assert_eq!(diff,0,"seed or replicated values differ");for(i,v)in actual.iter().enumerate().filter(|(i,v)|v.to_f32().to_bits()!=expected[i%expected.len()].to_f32().to_bits()).take(12){println!("qdiff kind={name} group={} row={} k={} actual={} expected={}",i/expected.len(),i/width%64,i%width,v.to_f32(),expected[i%expected.len()].to_f32());}}
    for(name,actual)in[("seed",&s),("copies",&cs)]{let diff=actual.iter().enumerate().filter(|(i,v)|v.to_bits()!=scales[i%64].to_bits()).count();println!("seed_scale axis={} kind={name} checked={} bitdiff={diff}",stringify!($axis),actual.len());assert_eq!(diff,0,"seed or replicated values differ");for(i,v)in actual.iter().enumerate().filter(|(i,v)|v.to_bits()!=scales[i%64].to_bits()).take(12){println!("sdiff kind={name} group={} row={} actual={} expected={}",i/64,i%64,v,scales[i%64]);}}
 }};
}
#[tokio::main] async fn main(){let mut device=Device::new(ops_native_fp8_r64_split_switch::debug_up_quant.topology()).unwrap();for salt in [0,991]{check!(device,H,L,7680,debug_up_quant,salt);check!(device,L,H,1920,debug_down_quant,salt);}}
