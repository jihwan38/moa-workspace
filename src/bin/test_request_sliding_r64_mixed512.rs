#[path="support/npu_profile.rs"]
mod npu_profile;
// Pooled sliding attention: distinct request rings, sparse updates and migration.
use furiosa_opt_gemma4::{Chip, axes::*, ops, ops_request_sliding_r64_mixed512 as candidate, ops_request_sliding_r64 as control};
use furiosa_opt_std::prelude::*;
use furiosa_opt_gemma4::ops_pool_views::Rc64;
use std::time::Instant;
static PROFILE_DONE:std::sync::atomic::AtomicBool=std::sync::atomic::AtomicBool::new(false);
static COLLECTOR:std::sync::OnceLock<npu_profile::Collector>=std::sync::OnceLock::new();
fn bf(x: f32) -> f32 {
    f32::from_bits((x.to_bits().wrapping_add(0x7fff + ((x.to_bits() >> 16) & 1))) & 0xffff0000)
}
fn values(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut x = (i as u64)
                .wrapping_add(seed)
                .wrapping_add(0x9e3779b97f4a7c15);
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            x ^= x >> 31;
            bf((((x >> 40) as f32) / 16777216.0 * 2.0 - 1.0) * scale)
        })
        .collect()
}
async fn bf_upload<Mp: M>(d: &mut Device, x: &[f32]) -> HbmTensor<bf16, Chip, Mp> {
    assert_eq!(x.len(), Mp::SIZE);
    let bytes: Vec<u8> = x
        .iter()
        .flat_map(|x| ((bf(*x).to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    HostTensor::<bf16, Mp>::from_buf(bytes)
        .to_hbm(&mut d.pdma)
        .await
        .unwrap()
}
async fn f_upload<Mp: M>(d: &mut Device, x: &[f32]) -> HbmTensor<f32, Chip, Mp> {
    assert_eq!(x.len(), Mp::SIZE);
    HostTensor::<f32, Mp>::from_buf(x.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
        .to_hbm(&mut d.pdma)
        .await
        .unwrap()
}
async fn zero<D: ScalarBytes + MaterializableScalar, Mp: M>(
    d: &mut Device,
) -> HbmTensor<D, Chip, Mp> {
    HostTensor::<D, Mp>::from_buf(vec![0u8; Mp::SIZE * D::BITS / 8])
        .to_hbm(&mut d.pdma)
        .await
        .unwrap()
}
async fn read<Mp: M>(d: &mut Device, x: &HbmTensor<bf16, Chip, Mp>) -> Vec<f32> {
    let h: HostTensor<bf16, Mp> = x.to_host(&mut d.pdma).await.unwrap();
    h.into_vec().into_iter().map(bf16::to_f32).collect()
}


fn slot(r:usize)->usize{(r*5+3)%64}
fn positions(policy:i32)->Vec<usize>{(0..64).map(|r|if policy&(1<<(r/8))!=0{[0,31,255,510,511,1,127,256][r%8]}else{[512,513,700,1023,1024,1535,2048,8191][r%8]}).collect()}
async fn indices<Mp:M>(d:&mut Device,x:Vec<i32>)->HbmTensor<i32,Chip,Mp>{HostTensor::<i32,Mp>::from_vec(x).to_hbm(&mut d.pdma).await.unwrap()}
async fn mask(d:&mut Device,pos:&[usize],count:usize)->HbmTensor<f32,Chip,m![R64,Ts]>{let x:Vec<f32>=(0..64).flat_map(|r|(0..1024).map(move|t|if r<count&&(pos[r]>=1023||t<=pos[r]){1.}else{0.})).collect();f_upload(d,&x).await}
async fn raw(d:&mut Device,x:&HbmTensor<bf16,Chip,m![R64,Ns,Gs,Ds]>)->Vec<u8>{x.to_host::<m![R64,Ns,Gs,Ds]>(&mut d.pdma).await.unwrap().into_inner().into_buf()}
fn as_f32(x:&[u8])->Vec<f32>{x.chunks_exact(2).map(|b|bf16::from_bits(u16::from_le_bytes([b[0],b[1]])).to_f32()).collect()}
fn exact(label:&str,actual:&[u8],expected:&[u8]){let mismatches=actual.chunks_exact(2).zip(expected.chunks_exact(2)).filter(|(a,b)|a!=b).count();if mismatches!=0{let first=actual.chunks_exact(2).zip(expected.chunks_exact(2)).position(|(a,b)|a!=b).unwrap();panic!("{label}: mismatches={mismatches} first={first} actual={:?} expected={:?}",&actual[first*2..first*2+2],&expected[first*2..first*2+2]);}}
fn composed(policy:i32,short:&[u8],long:&[u8])->Vec<u8>{let mut result=Vec::with_capacity(short.len());let n=8*4096*2;for group in 0..8{let src=if policy&(1<<group)!=0{short}else{long};result.extend_from_slice(&src[group*n..(group+1)*n]);}result}
async fn compare(
    d: &mut Device,
    q: &[f32],
    data: &[(Vec<f32>, Vec<f32>)],
    got: &[f32],
    count: usize,
    positions:&[usize],
) {
    for r in 0..count {
        let qr = bf_upload::<m![Ns, Gs, Ds]>(d, &q[r * 4096..(r + 1) * 4096]).await;
        let kr = bf_upload::<m![Ts, Ns, Ds]>(d, &data[r].0).await;
        let vr = bf_upload::<m![Ts, Ns, Ds]>(d, &data[r].1).await;
        let mask: Vec<f32> = (0..1024)
            .map(|t| {
                if positions[r] >= 1023 || t <= positions[r] {
                    1.
                } else {
                    0.
                }
            })
            .collect();
        let mask = f_upload::<m![Ts]>(d, &mask).await;
        let mut reference = zero::<bf16, m![Ns, Gs, Ds]>(d).await;
        launch(
            ops::sliding_attention,
            (d, &qr, &kr, &vr, &mask, &mut reference),
        )
        .await
        .unwrap();
        let reference = read(d, &reference).await;
        let n = got[r * 4096..(r + 1) * 4096]
            .iter()
            .zip(&reference)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let max = got[r * 4096..(r + 1) * 4096]
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!(
            "pooled sliding row={r} pos={} bit_mismatches={n} max_abs={max}",
            positions[r]
        );
        assert_eq!(n, 0, "pooled sliding vs scalar");
    }
    assert!(
        got[count * 4096..].iter().all(|x| *x == 0.),
        "inactive rows must be finite zero"
    );
}
async fn copy(
    d: &mut Device,
    k: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    v: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    r: usize,
    data: &(Vec<f32>, Vec<f32>),
) {
    let kr = bf_upload::<m![Ts, Ns, Ds]>(d, &data.0).await;
    let vr = bf_upload::<m![Ts, Ns, Ds]>(d, &data.1).await;
    let index = indices::<m![1]>(d, vec![(slot(r) * 1024 * 2048 * 2) as i32]).await;
    launch(
        furiosa_opt_gemma4::ops_request_sliding_r64::copy_sliding_cache_to_pool_rc64,
        (d, &kr, &vr, &index, k, v),
    )
    .await
    .unwrap();
}

#[tokio::main]
async fn main(){
 let collector=npu_profile::Collector::default();tracing::subscriber::set_global_default(collector.clone()).unwrap();COLLECTOR.set(collector).ok().unwrap();
 let mut d=Device::new(candidate::sliding_attention_r64.topology()).unwrap();
 let mut k=zero::<bf16,m![Rc64,Ts,Ns,Ds]>(&mut d).await;let mut v=zero::<bf16,m![Rc64,Ts,Ns,Ds]>(&mut d).await;
 let qv=values(64*4096,100,0.2);let q=bf_upload::<m![R64,Ns,Gs,Ds]>(&mut d,&qv).await;
 let mut data=Vec::new();for r in 0..64{data.push((values(1024*2048,200+r as u64,0.2),values(1024*2048,300+r as u64,1.)));copy(&mut d,&mut k,&mut v,r,&data[r]).await;}
 let index=indices::<m![R64]>(&mut d,(0..64).map(|r|(slot(r)*1024*2048*2)as i32).collect()).await;
 let mut out=zero::<bf16,m![R64,Ns,Gs,Ds]>(&mut d).await;let mut baseline=zero::<bf16,m![R64,Ns,Gs,Ds]>(&mut d).await;
 let short_positions=positions(255);let short_mask=mask(&mut d,&short_positions,64).await;
 launch(control::sliding_attention_r64,(&mut d,&q,&k,&v,&index,&short_mask,&mut baseline)).await.unwrap();let short=raw(&mut d,&baseline).await;
 compare(&mut d,&qv,&data,&as_f32(&short),64,&short_positions).await;
 let long_positions=positions(0);let long_mask=mask(&mut d,&long_positions,64).await;
 launch(control::sliding_attention_r64,(&mut d,&q,&k,&v,&index,&long_mask,&mut baseline)).await.unwrap();let long=raw(&mut d,&baseline).await;
 compare(&mut d,&qv,&data,&as_f32(&long),64,&long_positions).await;
 let mut policy=HbmScalar::<i32>::from_host(0,&mut d.pdma).await.unwrap();
 for bits in 0..256 {
  let pos=positions(bits);let visibility=mask(&mut d,&pos,64).await;policy.write(bits,&mut d.pdma).await.unwrap();
  launch(candidate::sliding_attention_r64,(&mut d,&q,&k,&v,&index,&visibility,&policy,&mut out)).await.unwrap();
  let got=raw(&mut d,&out).await;exact(&format!("policy={bits:02x}"),&got,&composed(bits,&short,&long));
  if bits==0||bits==255||bits==0x55||bits==0xaa||(bits>0 && bits&(bits-1)==0){println!("MIXED_POLICY_EXACT policy={bits:02x} checked=262144");}
 }
 println!("MIXED_ALL_256_POLICIES_PASS checked_values={}",256*64*4096);
 for bits in [0,255,0x55,0x80] {
  let pos=positions(bits);let visibility=mask(&mut d,&pos,64).await;policy.write(bits,&mut d.pdma).await.unwrap();
  let mut times=[Vec::new(),Vec::new()];for variant in [0,1,1,0]{let start=Instant::now();for _ in 0..3{if variant==0{launch(control::sliding_attention_r64,(&mut d,&q,&k,&v,&index,&visibility,&mut baseline)).await.unwrap();}else{launch(candidate::sliding_attention_r64,(&mut d,&q,&k,&v,&index,&visibility,&policy,&mut out)).await.unwrap();}}times[variant].push(start.elapsed().as_secs_f64()*1e6/3.);}
  println!("MIXED_TIMING policy={bits:02x} full_us={:?} mixed_us={:?}",times[0],times[1]);
  if bits==0x55&&std::env::var_os("FURIOSA_OPT_PROFILE").is_some(){println!("PROFILE_VARIANT mixed55");COLLECTOR.get().unwrap().enable(true);launch(candidate::sliding_attention_r64,(&mut d,&q,&k,&v,&index,&visibility,&policy,&mut out)).await.unwrap();COLLECTOR.get().unwrap().enable(false);println!("PROFILE_VARIANT end");}
 }
 let bits=0xa5;let pos=positions(bits);let visibility=mask(&mut d,&pos,64).await;policy.write(bits,&mut d.pdma).await.unwrap();
 for r in 0..64{if pos[r]<1023{for t in pos[r]+1..1024{for h in 0..8{for j in 0..256{let i=(t*8+h)*256+j;data[r].0[i]=bf(qv[(r*16+h*2)*256+j]*256.);data[r].1[i]=bf(-2000.+j as f32);}}}copy(&mut d,&mut k,&mut v,r,&data[r]).await;}}
 launch(candidate::sliding_attention_r64,(&mut d,&q,&k,&v,&index,&visibility,&policy,&mut out)).await.unwrap();let got=raw(&mut d,&out).await;exact("future poison invariance",&got,&composed(bits,&short,&long));
 launch(control::sliding_attention_r64,(&mut d,&q,&k,&v,&index,&visibility,&mut baseline)).await.unwrap();exact("future poison full control",&got,&raw(&mut d,&baseline).await);
 for r in 0..7{let p=pos[r]%1024;let nk=values(2048,400+r as u64,0.2);let nv=values(2048,500+r as u64,1.);data[r].0[p*2048..(p+1)*2048].copy_from_slice(&nk);data[r].1[p*2048..(p+1)*2048].copy_from_slice(&nv);copy(&mut d,&mut k,&mut v,r,&data[r]).await;}
 for count in [7,64]{let visibility=mask(&mut d,&pos,count).await;launch(control::sliding_attention_r64,(&mut d,&q,&k,&v,&index,&visibility,&mut baseline)).await.unwrap();launch(candidate::sliding_attention_r64,(&mut d,&q,&k,&v,&index,&visibility,&policy,&mut out)).await.unwrap();let got=raw(&mut d,&out).await;exact("sparse/group transitions",&got,&raw(&mut d,&baseline).await);if count==7{compare(&mut d,&qv,&data,&as_f32(&got),7,&pos).await;}println!("MIXED_SPARSE_EXACT count={count}");}
 println!("PASS mixedR8 all256policies, originalscalarshort/long, causalpoison, sparseupdates,64->7->64; samepoolABI/math");
}
