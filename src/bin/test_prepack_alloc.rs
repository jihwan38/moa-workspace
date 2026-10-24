//! Every packed byte/scalar bit against unchanged prepack and blocking kernels.
use furiosa_opt_gemma4::{Chip, axes::{H,L}, ops_prepack_alloc as candidate,
    ops_batch_prepack8 as control, ops_native_fp8_blocked as block};
use furiosa_opt_std::prelude::*;
fn mix(i:usize,seed:u64)->u64 {
    let mut x=(i as u64).wrapping_add(seed);x=(x^(x>>30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x=(x^(x>>27)).wrapping_mul(0x94d049bb133111eb);x^(x>>31)
}
async fn bytes<D:MaterializableScalar,E:M>(d:&mut Device,x:&HbmTensor<D,Chip,E>)->Vec<u8> {
    let host:HostTensor<D,E>=x.to_host(&mut d.pdma).await.unwrap();
    host.into_inner().into_buf()
}
fn exact(label:&str,a:&[u8],b:&[u8]) {
    assert_eq!(a.len(),b.len());
    let mismatch=a.iter().zip(b).filter(|(a,b)|a!=b).count();
    println!("{label}: bytes={} mismatches={mismatch}",a.len());assert_eq!(mismatch,0);
}
macro_rules! probe { ($name:ident,$out:ident,$kin:ident,$pack:ident,$reblock:ident,$layout:ty)=>{
async fn $name(d:&mut Device) {
    let mut previous: Option<(HbmTensor<f8e4m3,Chip,$layout>,Vec<u8>)> = None;
    for seed in [13,71] {
        let packed:Vec<u8>=(0..$out::SIZE*$kin::SIZE/2).map(|i|
            if(i/($kin::SIZE/2))%257==0 {0x80} else {mix(i,seed)as u8}).collect();
        let scales:Vec<f8e4m3>=(0..$out::SIZE*$kin::SIZE/16).map(|i|{
            let w=mix(i,seed+4);f8e4m3::from_f32((1.0+((w>>8)%8)as f32/8.0)*2f32.powi((w%15)as i32-8))
        }).collect();
        let p=HostTensor::<f4e2m1,m![$out,$kin]>::from_buf(packed).to_hbm(&mut d.pdma).await.unwrap();
        let s=HostTensor::<f8e4m3,m![$out,$kin/16]>::from_vec(scales).to_hbm(&mut d.pdma).await.unwrap();
        let now=std::time::Instant::now();
        let mut cq=HostTensor::<f8e4m3,m![$out,$kin]>::zero().to_hbm(&mut d.pdma).await.unwrap();
        let mut cs=HostTensor::<f32,m![$out]>::zero().to_hbm(&mut d.pdma).await.unwrap();
        launch(control::$pack,(&mut *d,&p,&s,&mut cq,&mut cs)).await.unwrap();
        let mut cb=HostTensor::<f8e4m3,$layout>::zero().to_hbm(&mut d.pdma).await.unwrap();
        launch(block::$reblock,(&mut *d,&cq,&mut cb)).await.unwrap();
        let old_ms=now.elapsed().as_secs_f64()*1e3;
        let now=std::time::Instant::now();
        let (q,scale)=launch(candidate::$pack,(&mut *d,&p,&s)).await.unwrap();
        let blocked=launch(candidate::$reblock,(&mut *d,&q)).await.unwrap();
        let new_ms=now.elapsed().as_secs_f64()*1e3;
        println!("{} seed={seed} old_alloc_prepack_reblock_ms={old_ms:.3} returned_ms={new_ms:.3}",stringify!($name));
        exact("row_major_q",&bytes(d,&cq).await,&bytes(d,&q).await);
        exact("dequant_scale",&bytes(d,&cs).await,&bytes(d,&scale).await);
        let expected=bytes(d,&cb).await;exact("blocked_q",&expected,&bytes(d,&blocked).await);
        if let Some((old,expected))=&previous { exact("retained_previous_allocation",expected,&bytes(d,old).await); }
        previous=Some((blocked,expected));
    }
}}
probe!(up,L,H,prepack_up,reblock_up,block::UpWeight);
probe!(down,H,L,prepack_down,reblock_down,block::DownWeight);
#[tokio::main]
async fn main() {
    let mut d=Device::new(candidate::prepack_up.topology()).unwrap();
    up(&mut d).await;down(&mut d).await;
    println!("PASS returned FP8 prepack and blocked allocations; all bytes exact");
}
