//! Exact shared-input QKV gate against frozen four-group projections and F64 dots.
use furiosa_opt_gemma4::{Chip,axes::*,ops_qkv_shared_ingress as candidate,ops_qkv_shared_blocked as control,ops_query_native_fp8_control as quant};
use furiosa_opt_std::prelude::*;
#[path="support/npu_profile.rs"] mod npu_profile;
fn hash(i: usize, seed: usize) -> usize {
    let mut v = (i as u64).wrapping_add(seed as u64);
    v = (v ^ (v >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    v = (v ^ (v >> 27)).wrapping_mul(0x94d049bb133111eb);
    (v ^ (v >> 31)) as usize
}
fn activation(b: usize, k: usize, seed: usize) -> bf16 {
    let unit = if seed == 0 {
        ((b * 17 + k * 13) % 31) as f32 / 16.0 - 15.0 / 16.0
    } else {
        ((hash(b * H::SIZE + k, seed) % 255) as f32 - 127.0) / 128.0
    };
    let value = match b {
        0 => 0.0,
        1 => unit * 2f32.powi(-110),
        2 => {
            if k % 2 == 0 {
                0.25
            } else {
                -0.25
            }
        }
        _ => unit * 2f32.powi((b % 11) as i32 - 5),
    };
    bf16::from_f32(value)
}
fn weight(n: usize, k: usize, seed: usize) -> f8e4m3 {
    let value = if seed == 0 {
        ((n * 5 + k * 7) % 17) as f32 / 8.0 - 1.0
    } else {
        ((hash(n * H::SIZE + k / 2, seed + 7) % 16) as f32 - 8.0) / 16.0
    };
    f8e4m3::from_f32(value)
}
fn channel(n: usize) -> bf16 {
    bf16::from_f32([0.0, 0.5, -0.5, 1.0, -1.0, 1.5, 2.0][n % 7])
}
fn quantize(x: &[bf16]) -> (Vec<f8e4m3>, Vec<f32>) {
    let mut q = Vec::with_capacity(x.len());
    let mut scales = Vec::with_capacity(P128::SIZE);
    for row in x.chunks_exact(H::SIZE) {
        let max = row
            .iter()
            .map(|x| x.to_f32().abs())
            .fold(1.925929944387236e-34f32, f32::max);
        let scale = f32::from_bits(max.to_bits() & 0x7f80_0000) * 0.0078125;
        scales.push(scale);
        q.extend(row.iter().map(|x| f8e4m3::from_f32(x.to_f32() / scale)));
    }
    (q, scales)
}
fn cpu_dot(q: &[f8e4m3], w: &[f8e4m3], b: usize, n: usize, scale: f32, channel: bf16) -> bf16 {
    // Bounded dyadic domains keep each H960 partial and its four-part sum exact F32.
    let dot = (0..H::SIZE)
        .map(|k| q[b * H::SIZE + k].to_f32() as f64 * w[n * H::SIZE + k].to_f32() as f64)
        .sum::<f64>();
    bf16::from_f32(bf16::from_f32(dot as f32 * scale).to_f32() * channel.to_f32())
}
async fn read<E: M>(d: &mut Device, x: &HbmTensor<bf16, Chip, E>) -> Vec<bf16> {
    let h: HostTensor<bf16, E> = x.to_host(&mut d.pdma).await.unwrap();
    h.into_vec()
}

async fn upload<D:MaterializableScalar,E:M>(d:&mut Device,x:impl IntoIterator<Item=D>)->HbmTensor<D,Chip,E>{HostTensor::<D,E>::from_vec(x).to_hbm(&mut d.pdma).await.unwrap()}
async fn zeros<D:MaterializableScalar,E:M>(d:&mut Device)->HbmTensor<D,Chip,E>{HostTensor::<D,E>::zero().to_hbm(&mut d.pdma).await.unwrap()}
async fn outputs<E:M>(d:&mut Device)->[HbmTensor<bf16,Chip,E>;4]{let mut v=Vec::new();for _ in 0..4{let n=HostTensor::<bf16,E>::zero().into_vec().len();v.push(upload(d,vec![bf16::from_f32(f32::NAN);n]).await);}v.try_into().ok().unwrap()}
async fn read_four<E:M>(d:&mut Device,x:&[HbmTensor<bf16,Chip,E>;4])->Vec<Vec<bf16>>{let mut h=Vec::new();for t in x{h.push(read(d,t).await);}h}
fn check(name:&str,actual:&[Vec<bf16>],expected:&[Vec<bf16>]){assert!(actual.iter().flatten().all(|x|x.to_f32().is_finite()));let count:usize=actual.iter().map(Vec::len).sum();let diff:usize=actual.iter().zip(expected).map(|(a,b)|a.iter().zip(b).filter(|(x,y)|x.to_bits()!=y.to_bits()).count()).sum();println!("INGRESS_QKV {name} checked={count} bitdiff={diff}");assert_eq!(diff,0);}
fn cpu_check(name:&str,actual:&[Vec<bf16>],qs:&[Vec<f8e4m3>],xs:&[Vec<f32>],w:&[f8e4m3],sc:&[bf16],order:[usize;4],seed:usize){
 let n=sc.len();let cols:Vec<usize>=if seed==0{(0..n).collect()}else{let mut v=vec![0,1,7,8,31,32,63,64,127,128,n/2-1,n/2,n-2,n-1];v.sort_unstable();v.dedup();v};let mut count=0;let mut diff=0;
 for group in 0..4{for row in 0..128{
  let cached:Vec<bf16>=(0..if seed==0{17}else{cols.len()}).map(|j|{let col=if seed==0{j}else{cols[j]};cpu_dot(&qs[order[group]],w,row,col,xs[order[group]][row],bf16::from_f32(1.0))}).collect();
  for(j,&col)in cols.iter().enumerate(){let expected=bf16::from_f32(cached[if seed==0{col%17}else{j}].to_f32()*sc[col].to_f32());count+=1;if expected.to_bits()!=actual[group][row*n+col].to_bits(){if diff<8{eprintln!("cpu_diff {name} group={group} row={row} col={col} expected={} actual={}",expected.to_f32(),actual[group][row*n+col].to_f32());}diff+=1;}}
 }}println!("INGRESS_QKV_CPU {name} checked={count} bitdiff={diff}");assert_eq!(diff,0);
}
fn unblock(groups:Vec<Vec<bf16>>,width:usize,block:usize)->Vec<Vec<bf16>> {
 groups.into_iter().map(|raw|{let mut canonical=vec![bf16::from_f32(0.0);raw.len()];for row in 0..128{for col in 0..width{canonical[row*width+col]=raw[(((row/64*(width/block)+col/block)*64+row%64)*block)+col%block];}}canonical}).collect()
}
#[tokio::main]
async fn main(){
 let collector=npu_profile::Collector::default();tracing::subscriber::set_global_default(collector.clone()).unwrap();
 let mut d=Device::new(candidate::project_sliding_qkv_four.topology()).unwrap();
 for seed in [0usize,101]{
  let mut inputs=Vec::new();let mut q=Vec::new();let mut xs=Vec::new();let mut cpu_q=Vec::new();let mut cpu_xs=Vec::new();
  for group in 0..4{
   let values:Vec<_>=(0..P128::SIZE).flat_map(|row|(0..H::SIZE).map(move|k|activation(row,k,seed+group*100))).collect();let(cq,cs)=quantize(&values);cpu_q.push(cq);cpu_xs.push(cs);
   inputs.push(upload::<bf16,m![P128,H]>(&mut d,values).await);q.push(zeros::<f8e4m3,m![P128,H]>(&mut d).await);xs.push(zeros::<f32,m![P128,1#8]>(&mut d).await);
   launch(quant::quantize_input,(&mut d,&inputs[group],&mut q[group],&mut xs[group])).await.unwrap();
   let hq:HostTensor<f8e4m3,m![P128,H]>=q[group].to_host(&mut d.pdma).await.unwrap();let hq=hq.into_vec();let hs:HostTensor<f32,m![P128,1#8]>=xs[group].to_host(&mut d.pdma).await.unwrap();let hs=hs.into_vec();
   let qdiff=hq.iter().zip(&cpu_q[group]).filter(|(a,b)|a.to_f32().to_bits()!=b.to_f32().to_bits()).count();let sdiff=(0..128).filter(|&r|hs[r*8].to_bits()!=cpu_xs[group][r].to_bits()).count();println!("INGRESS_QKV_quant seed={seed} group={group} checked={} qdiff={qdiff} scalediff={sdiff}",hq.len());assert_eq!(qdiff,0);assert_eq!(sdiff,0);
  }
  let qw_cpu:Vec<_>=(0..Qs::SIZE).flat_map(|n|(0..H::SIZE).map(move|k|weight(n,k,seed))).collect();
  let kw_cpu:Vec<_>=(0..Ps::SIZE).flat_map(|n|(0..H::SIZE).map(move|k|weight(n,k,if seed==0{0}else{seed+11}))).collect();
  let vw_cpu:Vec<_>=(0..Ps::SIZE).flat_map(|n|(0..H::SIZE).map(move|k|weight(n,k,if seed==0{0}else{seed+29}))).collect();
  let qs_cpu:Vec<_>=(0..Qs::SIZE).map(channel).collect();let ks_cpu:Vec<_>=(0..Ps::SIZE).map(|i|channel(i+3)).collect();let vs_cpu:Vec<_>=(0..Ps::SIZE).map(|i|channel(i+5)).collect();
  let qw:HbmTensor<f8e4m3,Chip,m![Qs,H]>=upload(&mut d,qw_cpu.iter().copied()).await;let kw:HbmTensor<f8e4m3,Chip,m![Ps,H]>=upload(&mut d,kw_cpu.iter().copied()).await;let vw:HbmTensor<f8e4m3,Chip,m![Ps,H]>=upload(&mut d,vw_cpu.iter().copied()).await;
  let qs:HbmTensor<bf16,Chip,m![Qs]>=upload(&mut d,qs_cpu.iter().copied()).await;let ks:HbmTensor<bf16,Chip,m![Ps]>=upload(&mut d,ks_cpu.iter().copied()).await;let vs:HbmTensor<bf16,Chip,m![Ps]>=upload(&mut d,vs_cpu.iter().copied()).await;
  let mut qo:[HbmTensor<bf16,Chip,candidate::QueryOutput>;4]=outputs(&mut d).await;let mut ko:[HbmTensor<bf16,Chip,candidate::KeyOutput>;4]=outputs(&mut d).await;let mut vo:[HbmTensor<bf16,Chip,candidate::KeyOutput>;4]=outputs(&mut d).await;
  let mut eqo:[HbmTensor<bf16,Chip,control::QueryOutput>;4]=outputs(&mut d).await;let mut eko:[HbmTensor<bf16,Chip,control::KeyOutput>;4]=outputs(&mut d).await;let mut evo:[HbmTensor<bf16,Chip,control::KeyOutput>;4]=outputs(&mut d).await;

  macro_rules! shared {($order:expr)=>{{let p=$order;let[qa,qb,qc,qd]=&mut qo;let[ka,kb,kc,kd]=&mut ko;let[va,vb,vc,vd]=&mut vo;launch(candidate::project_sliding_qkv_four,(&mut d,&q[p[0]],&xs[p[0]],&q[p[1]],&xs[p[1]],&q[p[2]],&xs[p[2]],&q[p[3]],&xs[p[3]],&qw,&kw,&vw,&qs,&ks,&vs,(qa,ka,va),(qb,kb,vb),(qc,kc,vc),(qd,kd,vd))).await.unwrap();}};}
  macro_rules! separate {($order:expr)=>{{let p=$order;let[qa,qb,qc,qd]=&mut eqo;let[ka,kb,kc,kd]=&mut eko;let[va,vb,vc,vd]=&mut evo;launch(control::project_sliding_qkv_four,(&mut d,&q[p[0]],&xs[p[0]],&q[p[1]],&xs[p[1]],&q[p[2]],&xs[p[2]],&q[p[3]],&xs[p[3]],&qw,&kw,&vw,&qs,&ks,&vs,(qa,ka,va),(qb,kb,vb),(qc,kc,vc),(qd,kd,vd))).await.unwrap();}};}
  for order in [[0,1,2,3],[2,0,3,1]]{
   qo=outputs(&mut d).await;ko=outputs(&mut d).await;vo=outputs(&mut d).await;eqo=outputs(&mut d).await;eko=outputs(&mut d).await;evo=outputs(&mut d).await;
   // Shared path first, on each new input order; control cannot seed stale DM.
   shared!(order);let aq=unblock(read_four(&mut d,&qo).await,Qs::SIZE,64);let ak=unblock(read_four(&mut d,&ko).await,Ps::SIZE,32);let av=unblock(read_four(&mut d,&vo).await,Ps::SIZE,32);
   separate!(order);let eq=unblock(read_four(&mut d,&eqo).await,Qs::SIZE,64);let ek=unblock(read_four(&mut d,&eko).await,Ps::SIZE,32);let ev=unblock(read_four(&mut d,&evo).await,Ps::SIZE,32);
   println!("INGRESS_QKV_case seed={seed} order={order:?}");check("Q",&aq,&eq);check("K",&ak,&ek);check("V",&av,&ev);
   if order==[2,0,3,1]{cpu_check("Q",&aq,&cpu_q,&cpu_xs,&qw_cpu,&qs_cpu,order,seed);cpu_check("K",&ak,&cpu_q,&cpu_xs,&kw_cpu,&ks_cpu,order,seed);cpu_check("V",&av,&cpu_q,&cpu_xs,&vw_cpu,&vs_cpu,order,seed);}
  }
  if seed==101{
   for include_quant in [true,false]{let mut times=[Vec::new(),Vec::new()];for variant in [0,1,1,0]{let start=std::time::Instant::now();for _ in 0..10{if include_quant{for g in 0..4{launch(quant::quantize_input,(&mut d,&inputs[g],&mut q[g],&mut xs[g])).await.unwrap();}}if variant==0{separate!([0,1,2,3]);}else{shared!([0,1,2,3]);}}times[variant].push(start.elapsed().as_secs_f64()*1000.0/10.0);}println!("INGRESS_QKV_timing include_one_quant_per_input={include_quant} blocked_control_ms={:?} ingress_ms={:?} scope=10_completed_QKV_only_all_outputs_no_host_read",times[0],times[1]);}
   if std::env::var_os("FURIOSA_OPT_PROFILE").is_some(){for variant in [0,1]{println!("PROFILE_VARIANT {variant}");collector.enable(true);for g in 0..4{launch(quant::quantize_input,(&mut d,&inputs[g],&mut q[g],&mut xs[g])).await.unwrap();}if variant==0{separate!([0,1,2,3]);}else{shared!([0,1,2,3]);}collector.enable(false);}}
  }
 }
}
