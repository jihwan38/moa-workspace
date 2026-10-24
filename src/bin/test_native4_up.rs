//! Original-FP4 native blockdot correctness and throughput; activation quantization error excluded.
use furiosa_opt_gemma4::{Chip,axes::{P128,H,L},ops_batch_native4};
use furiosa_opt_std::prelude::*;
fn mix(i:usize,seed:u64)->u64{let mut x=(i as u64).wrapping_add(seed);x=(x^(x>>30)).wrapping_mul(0xbf58476d1ce4e5b9);x=(x^(x>>27)).wrapping_mul(0x94d049bb133111eb);x^(x>>31)}
fn lookup(x:u8)->f32{[0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,-0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0][x as usize]}
fn compare(label:&str,refv:&[bf16],got:&[bf16]){let(mut diff,mut fail,mut maxabs,mut sumerr,mut sumref)=(0,0,0f32,0f64,0f64);for(i,(r,g))in refv.iter().zip(got).enumerate(){let rf=r.to_f32();let gf=g.to_f32();let e=(rf-gf).abs();diff+=usize::from(r.to_bits()!=g.to_bits());maxabs=maxabs.max(e);sumerr+=(e as f64).powi(2);sumref+=(rf as f64).powi(2);if !gf.is_finite()||e>0.003+0.01*rf.abs(){if fail<6{eprintln!("{label} i={i} got={gf} expected={rf}");}fail+=1;}}println!("{label} checked={} bitdiff={diff} tolerance_failures={fail} max_abs={maxabs} relative_l2={}",refv.len(),(sumerr/sumref.max(1e-30)).sqrt());assert_eq!(fail,0);}
#[tokio::main]async fn main(){
 let mut device=Device::new(ops_batch_native4::project_up.topology()).unwrap();
 let xv:Vec<f8e4m3>=(0..P128::SIZE*H::SIZE).map(|i|{let w=mix(i,17);f8e4m3::from_f32((((w>>40)as f32/16777216.0)*2.0-1.0)*64.0*2f32.powi(-((w%9)as i32)))}).collect();
 let sv:Vec<f32>=(0..P128::SIZE).map(|i|2f32.powi((mix(i,19)%7)as i32-3)).collect();
 let xref:Vec<bf16>=xv.iter().enumerate().map(|(i,x)|bf16::from_f32(x.to_f32()*sv[i/H::SIZE])).collect();
 let packed:Vec<u8>=(0..L::SIZE*H::SIZE/2).map(|i|mix(i,23)as u8).collect();
 let scales:Vec<f8e4m3>=(0..L::SIZE*H::SIZE/16).map(|i|f8e4m3::from_f32((1+(mix(i,29)%31))as f32/512.0)).collect();
 let x:HbmTensor<f8e4m3,Chip,m![P128,H]>=HostTensor::<f8e4m3,m![P128,H]>::from_vec(xv.iter().copied()).to_hbm(&mut device.pdma).await.unwrap();
 let bx:HbmTensor<bf16,Chip,m![P128,H]>=HostTensor::<bf16,m![P128,H]>::from_vec(xref.iter().copied()).to_hbm(&mut device.pdma).await.unwrap();
 let xs:HbmTensor<f32,Chip,m![P128]>=HostTensor::<f32,m![P128]>::from_vec(sv.iter().copied()).to_hbm(&mut device.pdma).await.unwrap();
 let w:HbmTensor<f4e2m1,Chip,m![L,H]>=HostTensor::<f4e2m1,m![L,H]>::from_buf(packed.clone()).to_hbm(&mut device.pdma).await.unwrap();
 let s:HbmTensor<f8e4m3,Chip,m![L,H / 16]>=HostTensor::<f8e4m3,m![L,H / 16]>::from_vec(scales.iter().copied()).to_hbm(&mut device.pdma).await.unwrap();
 let mut out:HbmTensor<bf16,Chip,m![P128,L]>=HostTensor::<bf16,m![P128,L]>::from_vec((0..P128::SIZE*L::SIZE).map(|_|bf16::from_f32(f32::NAN))).to_hbm(&mut device.pdma).await.unwrap();
 launch(ops_batch_native4::project_up_reference,(&mut device,&bx,&w,&s,&mut out)).await.unwrap();let reference:HostTensor<bf16,m![P128,L]>=out.to_host(&mut device.pdma).await.unwrap();let reference=reference.into_vec();
 launch(ops_batch_native4::project_up,(&mut device,&x,&xs,&w,&s,&mut out)).await.unwrap();let result:HostTensor<bf16,m![P128,L]>=out.to_host(&mut device.pdma).await.unwrap();let result=result.into_vec();compare("native4_rows30_vs_weightedBF16",&reference,&result);
 launch(ops_batch_native4::project_up_rows6,(&mut device,&x,&xs,&w,&s,&mut out)).await.unwrap();let small:HostTensor<bf16,m![P128,L]>=out.to_host(&mut device.pdma).await.unwrap();let small=small.into_vec();compare("native4_rows6_vs_weightedBF16",&reference,&small);compare("native4_rows30_vs_rows6",&small,&result);
 let(mut scalar,mut got0,mut got1)=(Vec::new(),Vec::new(),Vec::new());for j in 0..512{let b=(mix(j,31)as usize)%P128::SIZE;let n=(mix(j,37)as usize)%L::SIZE;let dot:f64=(0..H::SIZE).map(|k|{let i=n*H::SIZE+k;let q=(packed[i/2]>>((i%2)*4))&15;let weight=bf16::from_f32(lookup(q)*scales[i/16].to_f32()).to_f32();xref[b*H::SIZE+k].to_f32()as f64*weight as f64}).sum();scalar.push(bf16::from_f32(dot as f32));got0.push(reference[b*L::SIZE+n]);got1.push(result[b*L::SIZE+n]);}compare("weightedBF16_vs_F64scalar",&scalar,&got0);compare("native4_vs_F64scalar",&scalar,&got1);
 let mut times=[Vec::<f64>::new(),Vec::<f64>::new(),Vec::<f64>::new()];for variant in [0,1,2,2,1,0]{let start=std::time::Instant::now();for _ in 0..5{if variant==0{launch(ops_batch_native4::project_up_reference,(&mut device,&bx,&w,&s,&mut out)).await.unwrap();}if variant==1{launch(ops_batch_native4::project_up,(&mut device,&x,&xs,&w,&s,&mut out)).await.unwrap();}if variant==2{launch(ops_batch_native4::project_up_rows6,(&mut device,&x,&xs,&w,&s,&mut out)).await.unwrap();}}let _:HostTensor<bf16,m![P128,L]>=out.to_host(&mut device.pdma).await.unwrap();times[variant].push(start.elapsed().as_secs_f64()*1000.0/5.0);}println!("native4_up timing weightedBF16_ms={:?} rows30_ms={:?} rows6_ms={:?} scope=5launches+1same_read_divided5",times[0],times[1],times[2]);
}
