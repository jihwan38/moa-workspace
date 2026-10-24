//! Every-value independent CPU gate for startup weight FP8 prepack.
use furiosa_opt_gemma4::{Chip,axes::{H,L},ops_batch_prepack8};
use furiosa_opt_std::prelude::*;
fn mix(i:usize,seed:u64)->u64{let mut x=(i as u64).wrapping_add(seed);x=(x^(x>>30)).wrapping_mul(0xbf58476d1ce4e5b9);x=(x^(x>>27)).wrapping_mul(0x94d049bb133111eb);x^(x>>31)}
fn value(p:u8)->f32{[0.,0.5,1.,1.5,2.,3.,4.,6.,-0.,-0.5,-1.,-1.5,-2.,-3.,-4.,-6.][p as usize]}
macro_rules! probe{($name:ident,$out:ident,$kin:ident,$kernel:ident)=>{async fn $name(device:&mut Device){
 let packed:Vec<u8>=(0..$out::SIZE*$kin::SIZE/2).map(|i|if(i/($kin::SIZE/2))%257==0{0x80}else{mix(i,13)as u8}).collect();
 let blocks:Vec<f8e4m3>=(0..$out::SIZE*$kin::SIZE/16).map(|i|{let bits=mix(i,17);f8e4m3::from_f32((1.0+((bits>>8)%8)as f32/8.0)*2f32.powi((bits%15)as i32-8))}).collect();
 let w:HbmTensor<f4e2m1,Chip,m![$out,$kin]>=HostTensor::<f4e2m1,m![$out,$kin]>::from_buf(packed.clone()).to_hbm(&mut device.pdma).await.unwrap();
 let s:HbmTensor<f8e4m3,Chip,m![$out,$kin / 16]>=HostTensor::<f8e4m3,m![$out,$kin / 16]>::from_vec(blocks.iter().copied()).to_hbm(&mut device.pdma).await.unwrap();
 let mut q:HbmTensor<f8e4m3,Chip,m![$out,$kin]>=HostTensor::<f8e4m3,m![$out,$kin]>::zero().to_hbm(&mut device.pdma).await.unwrap();
 let mut scales:HbmTensor<f32,Chip,m![$out]>=HostTensor::<f32,m![$out]>::zero().to_hbm(&mut device.pdma).await.unwrap();
 let start=std::time::Instant::now();launch(ops_batch_prepack8::$kernel,(device,&w,&s,&mut q,&mut scales)).await.unwrap();
 let aq:HostTensor<f8e4m3,m![$out,$kin]>=q.to_host(&mut device.pdma).await.unwrap();let aq=aq.into_vec();
 let asc:HostTensor<f32,m![$out]>=scales.to_host(&mut device.pdma).await.unwrap();let asc=asc.into_vec();let elapsed=start.elapsed().as_secs_f64()*1000.0;
 let(mut qdiff,mut sdiff,mut zero_sign_diff,mut error,mut norm)=(0,0,0,0f64,0f64);
 for row in 0..$out::SIZE {
    let weights:Vec<f32>=(0..$kin::SIZE).map(|k|{let i=row*$kin::SIZE+k;bf16::from_f32(value((packed[i/2]>>((i%2)*4))&15)*blocks[i/16].to_f32()).to_f32()}).collect();
    let amax=weights.iter().map(|x|x.abs()).fold(0f32,f32::max).max(2f32.powi(-112));let scale=f32::from_bits(amax.to_bits()&0x7f800000)/128.0;
    if scale.to_bits()!=asc[row].to_bits(){if sdiff<6{eprintln!("{} scale row={row} got={} expected={scale}",stringify!($out),asc[row]);}sdiff+=1;}
    for(k,w)in weights.into_iter().enumerate(){let expected=f8e4m3::from_f32(w/scale).to_f32();let got=aq[row*$kin::SIZE+k].to_f32();if got.to_bits()!=expected.to_bits(){if got==0.0&&expected==0.0{zero_sign_diff+=1;}else{if qdiff<6{eprintln!("{} q row={row} k={k} weight={w} scale={scale} got={got} expected={expected}",stringify!($out));}qdiff+=1;}}let e=got as f64*scale as f64-w as f64;error+=e*e;norm+=(w as f64).powi(2);}
 }
 println!("prepack {}x{} checked={} q_numeric_mismatches={qdiff} q_zero_sign_differences={zero_sign_diff} scale_bit_mismatches={sdiff} reconstruction_nrmse={} launch_plus_output_reads_ms={elapsed}",$out::SIZE,$kin::SIZE,aq.len(),(error/norm).sqrt());assert_eq!(sdiff,0);assert_eq!(qdiff,0);assert_eq!(zero_sign_diff,0);
}}}
probe!(up,L,H,prepack_up);probe!(down,H,L,prepack_down);
#[tokio::main]async fn main(){let mut device=Device::new(ops_batch_prepack8::prepack_up.topology()).unwrap();up(&mut device).await;down(&mut device).await;}
