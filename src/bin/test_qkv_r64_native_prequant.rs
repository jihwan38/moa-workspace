//! Native Qs arithmetic/layout oracle; BF16 activation differences are reported separately.
use furiosa_opt_gemma4::{
    Chip, axes::*, ops_qkv_r64_native_prequant as op,
};
use furiosa_opt_std::prelude::*;
mod control {
 pub use furiosa_opt_gemma4::ops_qkv_r64_trf32::{project_query_batch,project_sliding_kv_batch,project_full_query_batch};
 pub use furiosa_opt_gemma4::ops_batch_r64::project_full_key_batch;
}
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
    let mut scales = Vec::with_capacity(R64::SIZE);
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

macro_rules! fixture {
 ($d:ident,$seed:ident,$salt:expr,$axis:ident,$candidate:ident,$reference:ident,$input:ident,$q:ident,$xs:ident,$cpu_q:ident,$cpu_scale:ident)=>{{
    let wseed=if $seed==0 {0}else{$seed+$salt};
    let w:Vec<_>=(0..$axis::SIZE).flat_map(|n|(0..H::SIZE).map(move|k|weight(n,k,wseed))).collect();
    let sc:Vec<_>=(0..$axis::SIZE).map(|n|channel(n+$salt)).collect();
    let weights=HostTensor::<f8e4m3,m![$axis,H]>::from_vec(w.clone()).to_hbm(&mut $d.pdma).await.unwrap();
    let scales=HostTensor::<bf16,m![$axis]>::from_vec(sc.clone()).to_hbm(&mut $d.pdma).await.unwrap();
    let mut output=HostTensor::<bf16,m![R64,$axis]>::from_vec(vec![bf16::from_f32(f32::NAN);R64::SIZE*$axis::SIZE]).to_hbm(&mut $d.pdma).await.unwrap();
    launch(op::$candidate,(&mut $d,&$q,&$xs,&weights,&scales,&mut output)).await.unwrap();
    let actual=read(&mut $d,&output).await;
    let columns:Vec<_>=if $seed==0 {(0..$axis::SIZE).collect()}else{
        let mut v=vec![0,1,7,8,31,32,63,64,127,128,255,256,$axis::SIZE/2-1,$axis::SIZE/2,$axis::SIZE-2,$axis::SIZE-1];
        v.sort_unstable();v.dedup();v.retain(|&n|n<$axis::SIZE);v
    };
    let mut diff=0;
    for b in 0..R64::SIZE {
        let dots:Vec<_>=(0..if $seed==0{17}else{columns.len()}).map(|j|{
            let n=if $seed==0{j}else{columns[j]};
            cpu_dot(&$cpu_q,&w,b,n,$cpu_scale[b],bf16::from_f32(1.0))
        }).collect();
        for (j,&n) in columns.iter().enumerate(){
            let expected=bf16::from_f32(dots[if $seed==0{n%17}else{j}].to_f32()*sc[n].to_f32());
            if actual[b*$axis::SIZE+n].to_bits()!=expected.to_bits(){
                if diff<5{println!("{} seed={} salt={} cpu_difference row={b} channel={n} actual={} expected={}",stringify!($axis),$seed,$salt,actual[b*$axis::SIZE+n].to_f32(),expected.to_f32());}
                diff+=1;
            }
        }
    }
    println!("{} seed={} salt={} quantized_F64_oracle checked={} differences={diff}",stringify!($axis),$seed,$salt,R64::SIZE*columns.len());assert_eq!(diff,0);
    launch(control::$reference,(&mut $d,&$input,&weights,&scales,&mut output)).await.unwrap();
    let reference=read(&mut $d,&output).await;
    let(mut max_abs,mut e2,mut r2,mut differences)=(0.0f32,0.0f64,0.0f64,0usize);
    for(a,b)in actual.iter().zip(&reference){assert!(a.to_f32().is_finite());let delta=a.to_f32()-b.to_f32();max_abs=max_abs.max(delta.abs());e2+=(delta as f64).powi(2);r2+=(b.to_f32()as f64).powi(2);differences+=usize::from(a.to_bits()!=b.to_bits());}
    println!("{} seed={} salt={} BF16_comparison checked={} differences={differences} max_abs={max_abs} relative_l2={}",stringify!($axis),$seed,$salt,actual.len(),(e2/r2.max(1e-300)).sqrt());
    (weights,scales,output)
 }};
}
#[tokio::main]
async fn main() {
    let mut d = Device::new(op::project_query_quantized.topology()).unwrap();
    for seed in [0, 31, 79] {
        let x: Vec<_> = (0..R64::SIZE)
            .flat_map(|b| (0..H::SIZE).map(move |k| activation(b, k, seed)))
            .collect();
        let (cpu_q, cpu_scale) = quantize(&x);
        let input = HostTensor::<bf16, m![R64, H]>::from_vec(x)
            .to_hbm(&mut d.pdma)
            .await
            .unwrap();
        let mut q = HostTensor::<f8e4m3, m![R64, H]>::zero()
            .to_hbm(&mut d.pdma)
            .await
            .unwrap();
        let mut xs = HostTensor::<f32, m![R64,1#8]>::zero()
            .to_hbm(&mut d.pdma)
            .await
            .unwrap();
        launch(op::quantize_input, (&mut d, &input, &mut q, &mut xs))
            .await
            .unwrap();
        let hq: HostTensor<f8e4m3, m![R64, H]> = q.to_host(&mut d.pdma).await.unwrap();
        let hq = hq.into_vec();
        let hs: HostTensor<f32, m![R64,1#8]> = xs.to_host(&mut d.pdma).await.unwrap();
        let hs = hs.into_vec();
        assert_eq!(hs.len(), R64::SIZE * 8);
        let qdiff = hq
            .iter()
            .zip(&cpu_q)
            .filter(|(a, b)| a.to_f32().to_bits() != b.to_f32().to_bits())
            .count();
        let sdiff = (0..R64::SIZE)
            .filter(|&b| hs[b * 8].to_bits() != cpu_scale[b].to_bits())
            .count();
        println!(
            "seed={seed} quantize checked={} qdiff={qdiff} scalediff={sdiff}",
            hq.len()
        );
        assert_eq!(qdiff, 0);
        assert_eq!(sdiff, 0);
        let (qw, qs, mut qo) = fixture!(
            d,
            seed,
            0,
            Qs,
            project_query_quantized,
            project_query_batch,
            input,
            q,
            xs,
            cpu_q,
            cpu_scale
        );
        let (kw, ks, mut ko) = fixture!(
            d,
            seed,
            13,
            Ps,
            project_sliding_kv_quantized,
            project_sliding_kv_batch,
            input,
            q,
            xs,
            cpu_q,
            cpu_scale
        );
        let (vw, vs, mut vo) = fixture!(
            d,
            seed,
            101,
            Ps,
            project_sliding_kv_quantized,
            project_sliding_kv_batch,
            input,
            q,
            xs,
            cpu_q,
            cpu_scale
        );
        let (fqw, fqs, mut fqo) = fixture!(
            d,
            seed,
            17,
            Qf,
            project_full_query_quantized,
            project_full_query_batch,
            input,
            q,
            xs,
            cpu_q,
            cpu_scale
        );
        let (fkw, fks, mut fko) = fixture!(
            d,
            seed,
            97,
            Pf,
            project_full_key_quantized,
            project_full_key_batch,
            input,
            q,
            xs,
            cpu_q,
            cpu_scale
        );
        if seed == 31 {
            for variant in [0, 1, 1, 0] {
                let now = std::time::Instant::now();
                for _ in 0..20 {
                    if variant == 0 {
                        launch(
                            control::project_query_batch,
                            (&mut d, &input, &qw, &qs, &mut qo),
                        )
                        .await
                        .unwrap();
                        launch(
                            control::project_sliding_kv_batch,
                            (&mut d, &input, &kw, &ks, &mut ko),
                        )
                        .await
                        .unwrap();
                        launch(
                            control::project_sliding_kv_batch,
                            (&mut d, &input, &vw, &vs, &mut vo),
                        )
                        .await
                        .unwrap();
                    } else {
                        launch(op::quantize_input, (&mut d, &input, &mut q, &mut xs))
                            .await
                            .unwrap();
                        launch(
                            op::project_query_quantized,
                            (&mut d, &q, &xs, &qw, &qs, &mut qo),
                        )
                        .await
                        .unwrap();
                        launch(
                            op::project_sliding_kv_quantized,
                            (&mut d, &q, &xs, &kw, &ks, &mut ko),
                        )
                        .await
                        .unwrap();
                        launch(
                            op::project_sliding_kv_quantized,
                            (&mut d, &q, &xs, &vw, &vs, &mut vo),
                        )
                        .await
                        .unwrap();
                    }
                }
                println!(
                    "sliding_QKV_ABBA variant={variant} warm_us={:.3}",
                    now.elapsed().as_secs_f64() * 1e6 / 20.0
                );
            }
            for variant in [0, 1, 1, 0] {
                let now = std::time::Instant::now();
                for _ in 0..20 {
                    if variant == 0 {
                        launch(
                            control::project_full_query_batch,
                            (&mut d, &input, &fqw, &fqs, &mut fqo),
                        )
                        .await
                        .unwrap();
                        launch(
                            control::project_full_key_batch,
                            (&mut d, &input, &fkw, &fks, &mut fko),
                        )
                        .await
                        .unwrap();
                    } else {
                        launch(op::quantize_input, (&mut d, &input, &mut q, &mut xs))
                            .await
                            .unwrap();
                        launch(
                            op::project_full_query_quantized,
                            (&mut d, &q, &xs, &fqw, &fqs, &mut fqo),
                        )
                        .await
                        .unwrap();
                        launch(
                            op::project_full_key_quantized,
                            (&mut d, &q, &xs, &fkw, &fks, &mut fko),
                        )
                        .await
                        .unwrap();
                    }
                }
                println!(
                    "full_QK_ABBA variant={variant} warm_us={:.3}",
                    now.elapsed().as_secs_f64() * 1e6 / 20.0
                );
            }
        }
    }
}
