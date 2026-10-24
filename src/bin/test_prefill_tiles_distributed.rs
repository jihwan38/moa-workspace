//! Raw-byte CPU oracle and warmed ABBA timing for P128/B16 tile transfers.
use furiosa_opt_gemma4::{Chip,axes::*,ops_prefill_tiles as candidate,ops_prefill_tiles_reference as control};
use furiosa_opt_std::prelude::*;
use std::time::Instant;
fn bits(n:usize,seed:u64)->Vec<u16>{(0..n).map(|i|{let mut x=(i as u64).wrapping_add(seed).wrapping_add(0x9e3779b97f4a7c15);x=(x^(x>>30)).wrapping_mul(0xbf58476d1ce4e5b9);x=(x^(x>>27)).wrapping_mul(0x94d049bb133111eb);(x^(x>>31)) as u16}).collect()}
async fn upload<Mp:M>(d:&mut Device,values:&[u16])->HbmTensor<bf16,Chip,Mp>{assert_eq!(values.len(),Mp::SIZE);HostTensor::<bf16,Mp>::from_buf(values.iter().flat_map(|x|x.to_le_bytes()).collect::<Vec<_>>()).to_hbm(&mut d.pdma).await.unwrap()}
async fn read<Mp:M>(d:&mut Device,x:&HbmTensor<bf16,Chip,Mp>)->Vec<u16>{let h:HostTensor<bf16,Mp>=x.to_host(&mut d.pdma).await.unwrap();h.into_vec().into_iter().map(|x|x.to_bits()).collect()}
fn exact_bits(label:&str,got:&[u16],expected:&[u16]){assert_eq!(got.len(),expected.len());let count=got.iter().zip(expected).filter(|(a,b)|a!=b).count();if let Some((i,(a,b)))=got.iter().zip(expected).enumerate().find(|(_, (a,b))|a!=b){println!("{label} firstdiff={i} actual={a:04x} expected={b:04x}");}assert_eq!(count,0,"{label} bit differences");}
fn median(v:&mut [f64])->f64{v.sort_by(f64::total_cmp);(v[(v.len()-1)/2]+v[v.len()/2])/2.}
macro_rules! measure {($samples:expr,$kernel:path,$args:expr)=>{{let start=Instant::now();launch($kernel,$args).await.unwrap();$samples.push(start.elapsed().as_secs_f64()*1e6);}};}
macro_rules! case {($d:ident,$name:literal,$payload:ty,$store_a:path,$store_b:path,$load_a:path,$load_b:path)=>{{
    type Payload=$payload;type Wide=m![P128,{Payload}];type Narrow=m![B,{Payload}];
    let mut expected=bits(Wide::SIZE,11);
    let mut a=upload::<Wide>(&mut $d,&expected).await;let mut b=upload::<Wide>(&mut $d,&expected).await;
    let mut la=upload::<Narrow>(&mut $d,&bits(Narrow::SIZE,22)).await;let mut lb=upload::<Narrow>(&mut $d,&bits(Narrow::SIZE,33)).await;
    let mut sa_time=Vec::new();let mut sb_time=Vec::new();let mut la_time=Vec::new();let mut lb_time=Vec::new();
    for tile in [7usize,0,3,1,6,2,5,4] {
        let index=HostTensor::<i32,m![1]>::from_vec([tile as i32]).to_hbm(&mut $d.pdma).await.unwrap();
        let source_bits=bits(Narrow::SIZE,100+tile as u64);let source=upload::<Narrow>(&mut $d,&source_bits).await;
        expected[tile*Narrow::SIZE..(tile+1)*Narrow::SIZE].copy_from_slice(&source_bits);
        launch($store_a,(&mut $d,&source,&index,&mut a)).await.unwrap();launch($store_b,(&mut $d,&source,&index,&mut b)).await.unwrap();
        for _ in 0..4 {
            measure!(sa_time,$store_a,(&mut $d,&source,&index,&mut a));
            measure!(sb_time,$store_b,(&mut $d,&source,&index,&mut b));
            measure!(sb_time,$store_b,(&mut $d,&source,&index,&mut b));
            measure!(sa_time,$store_a,(&mut $d,&source,&index,&mut a));
        }
        exact_bits("original store CPUbyteoracle",&read(&mut $d,&a).await,&expected);
        exact_bits("distributed store selected/nonselected rows CPUbyteoracle",&read(&mut $d,&b).await,&expected);
        launch($load_a,(&mut $d,&b,&index,&mut la)).await.unwrap();launch($load_b,(&mut $d,&b,&index,&mut lb)).await.unwrap();
        for _ in 0..4 {
            measure!(la_time,$load_a,(&mut $d,&b,&index,&mut la));
            measure!(lb_time,$load_b,(&mut $d,&b,&index,&mut lb));
            measure!(lb_time,$load_b,(&mut $d,&b,&index,&mut lb));
            measure!(la_time,$load_a,(&mut $d,&b,&index,&mut la));
        }
        exact_bits("original load CPUbyteoracle",&read(&mut $d,&la).await,&source_bits);
        exact_bits("distributed load CPUbyteoracle",&read(&mut $d,&lb).await,&source_bits);
        println!("{} tile={tile} allwidebytes={} selectedbytes={} exact",$name,Wide::SIZE*2,Narrow::SIZE*2);
    }
    println!("ABBA {} store_original_us={:.3} store_distributed_us={:.3} load_original_us={:.3} load_distributed_us={:.3} samples_per_kernel={}",$name,median(&mut sa_time),median(&mut sb_time),median(&mut la_time),median(&mut lb_time),sa_time.len());
}};}
#[tokio::main]
async fn main(){
    let mut d=Device::new(candidate::store_hidden_tile.topology()).unwrap();
    case!(d,"hidden",m![H],control::store_hidden_tile,candidate::store_hidden_tile,control::load_hidden_tile,candidate::load_hidden_tile);
    case!(d,"query_sliding",m![Qs],control::store_query_sliding_tile,candidate::store_query_sliding_tile,control::load_query_sliding_tile,candidate::load_query_sliding_tile);
    case!(d,"key_sliding",m![Ps],control::store_key_sliding_tile,candidate::store_key_sliding_tile,control::load_key_sliding_tile,candidate::load_key_sliding_tile);
    case!(d,"query_full",m![Qf],control::store_query_full_tile,candidate::store_query_full_tile,control::load_query_full_tile,candidate::load_query_full_tile);
    case!(d,"key_full",m![Pf],control::store_key_full_tile,candidate::store_key_full_tile,control::load_key_full_tile,candidate::load_key_full_tile);
    case!(d,"heads_sliding",m![Ns,Gs,Ds],control::store_heads_sliding_tile,candidate::store_heads_sliding_tile,control::load_heads_sliding_tile,candidate::load_heads_sliding_tile);
    case!(d,"heads_full",m![Gf,Df],control::store_heads_full_tile,candidate::store_heads_full_tile,control::load_heads_full_tile,candidate::load_heads_full_tile);
    println!("PASS:7payloads8tileindices allBF16bits CPUbyteexact, nonselectedrows preserved, warmedABBA");
}
