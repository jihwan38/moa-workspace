//! Startup join byte oracle, independent of any matrix arithmetic.
use furiosa_opt_gemma4::{Chip,axes::L,ops_ffn_projection_join,ops_native_fp8_blocked::UpWeight};
use furiosa_opt_std::prelude::*;
type Result<T> = std::result::Result<T,Box<dyn std::error::Error>>;
fn bytes(n:usize,seed:u64)->Vec<u8> {
    (0..n).map(|i| {let mut x=(i as u64).wrapping_add(seed);
        x=(x^(x>>30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x=(x^(x>>27)).wrapping_mul(0x94d049bb133111eb);(x^(x>>31)) as u8}).collect()
}
async fn upload<D:MaterializableScalar,E:M>(d:&mut Device,b:Vec<u8>)->Result<HbmTensor<D,Chip,E>> {
    Ok(HostTensor::<D,E>::from_buf(b).to_hbm(&mut d.pdma).await?)
}
async fn read<D:MaterializableScalar,E:M>(d:&mut Device,x:&HbmTensor<D,Chip,E>)->Result<Vec<u8>> {
    let h:HostTensor<D,E>=x.to_host(&mut d.pdma).await?;Ok(h.into_inner().into_buf())
}
fn exact(label:&str,want:&[u8],got:&[u8]) {
    assert_eq!(want.len(),got.len());let diff=want.iter().zip(got).filter(|(a,b)|a!=b).count();
    println!("PROJECTION_JOIN_BYTES label={label} checked={} bitdiff={diff}",want.len());assert_eq!(diff,0);
}
#[tokio::main]
async fn main()->Result<()> {
    let mut d=Device::new(ops_ffn_projection_join::join.topology())?;
    for case in 0..2 {
        let up=bytes(UpWeight::SIZE,13+case*100);let gate=bytes(UpWeight::SIZE,71+case*100);
        let us=bytes(L::SIZE*4,31+case*100);let gs=bytes(L::SIZE*4,97+case*100);
        let du=upload::<f8e4m3,UpWeight>(&mut d,up.clone()).await?;
        let dg=upload::<f8e4m3,UpWeight>(&mut d,gate.clone()).await?;
        let ds=upload::<f32,m![L]>(&mut d,us.clone()).await?;
        let dt=upload::<f32,m![L]>(&mut d,gs.clone()).await?;
        let (weight,scale)=launch(ops_ffn_projection_join::join,(&mut d,&du,&dg,&ds,&dt)).await?;
        let got=read(&mut d,&weight).await?;
        exact(&format!("case{case}_up_half"),&up,&got[..up.len()]);
        exact(&format!("case{case}_gate_half"),&gate,&got[up.len()..]);
        let got=read(&mut d,&scale).await?;
        exact(&format!("case{case}_up_scale"),&us,&got[..us.len()]);
        exact(&format!("case{case}_gate_scale"),&gs,&got[us.len()..]);
        exact("immutable_up",&up,&read(&mut d,&du).await?);
        exact("immutable_gate",&gate,&read(&mut d,&dg).await?);
        exact("immutable_up_scale",&us,&read(&mut d,&ds).await?);
        exact("immutable_gate_scale",&gs,&read(&mut d,&dt).await?);
    }
    println!("PASS startup join two independent byte patterns, all output cells and immutable inputs; no per-request work or arithmetic");Ok(())
}
