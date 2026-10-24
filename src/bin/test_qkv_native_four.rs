//! Native Qs arithmetic/layout oracle; BF16 activation differences are reported separately.
use furiosa_opt_gemma4::{
    Chip, axes::*, ops_qkv_native_four as op, ops_qkv_native_prequant as control,
};
use furiosa_opt_std::prelude::*;
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

macro_rules! four {
    ($d:ident,$q:ident,$xs:ident,$w:ident,$s:ident,$out:ident,$kernel:ident,$order:expr) => {{
        let p = $order;
        let [a, b, c, e] = &mut $out;
        launch(
            op::$kernel,
            (
                &mut $d, &$q[p[0]], &$xs[p[0]], &$q[p[1]], &$xs[p[1]], &$q[p[2]], &$xs[p[2]],
                &$q[p[3]], &$xs[p[3]], &$w, &$s, a, b, c, e,
            ),
        )
        .await
        .unwrap();
    }};
}
macro_rules! fixture {
 ($d:ident,$seed:ident,$salt:expr,$axis:ident,$candidate:ident,$reference:ident,$q:ident,$xs:ident,$cpu_q:ident,$cpu_scale:ident)=>{{
    let wseed=if $seed==0{0}else{$seed+$salt};
    let w:Vec<_>=(0..$axis::SIZE).flat_map(|n|(0..H::SIZE).map(move|k|weight(n,k,wseed))).collect();
    let sc:Vec<_>=(0..$axis::SIZE).map(|n|channel(n+$salt)).collect();
    let weights=HostTensor::<f8e4m3,m![$axis,H]>::from_vec(w.clone()).to_hbm(&mut $d.pdma).await.unwrap();
    let scales=HostTensor::<bf16,m![$axis]>::from_vec(sc.clone()).to_hbm(&mut $d.pdma).await.unwrap();
    let mut v=Vec::new();for _ in 0..4{v.push(HostTensor::<bf16,m![P128,$axis]>::from_vec(vec![bf16::from_f32(f32::NAN);P128::SIZE*$axis::SIZE]).to_hbm(&mut $d.pdma).await.unwrap());}
    let mut out:[HbmTensor<bf16,Chip,m![P128,$axis]>;4]=v.try_into().ok().unwrap();
    let mut reference=Vec::new();for r in 0..4{launch(control::$reference,(&mut $d,&$q[r],&$xs[r],&weights,&scales,&mut out[r])).await.unwrap();reference.push(read(&mut $d,&out[r]).await);}
    for order in [[0,1,2,3],[2,0,3,1]] {
        four!($d,$q,$xs,weights,scales,out,$candidate,order);
        let mut total=0;let mut cpu_diff=0;let mut cpu_checked=0;
        for r in 0..4 {
            let actual=read(&mut $d,&out[r]).await;
            let diff=actual.iter().zip(&reference[order[r]]).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();total+=diff;
            if order==[2,0,3,1] {
                let columns:Vec<_>=if $seed==0{(0..$axis::SIZE).collect()}else{let mut v=vec![0,1,7,8,31,32,63,64,127,128,$axis::SIZE/2-1,$axis::SIZE/2,$axis::SIZE-2,$axis::SIZE-1];v.sort_unstable();v.dedup();v.retain(|&n|n<$axis::SIZE);v};
                for b in 0..P128::SIZE {
                    let dots:Vec<_>=(0..if $seed==0{17}else{columns.len()}).map(|j|{let n=if $seed==0{j}else{columns[j]};cpu_dot(&$cpu_q[order[r]],&w,b,n,$cpu_scale[order[r]][b],bf16::from_f32(1.0))}).collect();
                    for(j,&n)in columns.iter().enumerate(){let expected=bf16::from_f32(dots[if $seed==0{n%17}else{j}].to_f32()*sc[n].to_f32());cpu_diff+=usize::from(expected.to_bits()!=actual[b*$axis::SIZE+n].to_bits());cpu_checked+=1;}
                }
            }
        }
        println!("{} seed={} salt={} order={order:?} differential_checked={} differences={total} cpu_checked={cpu_checked} cpu_differences={cpu_diff}",stringify!($axis),$seed,$salt,4*P128::SIZE*$axis::SIZE);
        assert_eq!(total,0);assert_eq!(cpu_diff,0);
    }
    (weights,scales,out)
 }};
}
#[tokio::main]
async fn main() {
    let mut d = Device::new(op::project_query_four.topology()).unwrap();
    for seed in [0, 31, 79] {
        let mut input = Vec::new();
        let mut q = Vec::new();
        let mut xs = Vec::new();
        let mut cpu_q = Vec::new();
        let mut cpu_scale = Vec::new();
        for group in 0..4 {
            let s = if seed == 0 { 0 } else { seed + group * 43 };
            let x: Vec<_> = (0..P128::SIZE)
                .flat_map(|b| {
                    (0..H::SIZE).map(move |k| {
                        activation((b + group * 29) % P128::SIZE, (k + group * 17) % H::SIZE, s)
                    })
                })
                .collect();
            let (cq, cs) = quantize(&x);
            cpu_q.push(cq);
            cpu_scale.push(cs);
            input.push(
                HostTensor::<bf16, m![P128, H]>::from_vec(x)
                    .to_hbm(&mut d.pdma)
                    .await
                    .unwrap(),
            );
            q.push(
                HostTensor::<f8e4m3, m![P128, H]>::zero()
                    .to_hbm(&mut d.pdma)
                    .await
                    .unwrap(),
            );
            xs.push(
                HostTensor::<f32, m![P128,1#8]>::zero()
                    .to_hbm(&mut d.pdma)
                    .await
                    .unwrap(),
            );
            launch(
                op::quantize_input,
                (&mut d, &input[group], &mut q[group], &mut xs[group]),
            )
            .await
            .unwrap();
            let hq: HostTensor<f8e4m3, m![P128, H]> = q[group].to_host(&mut d.pdma).await.unwrap();
            let hq = hq.into_vec();
            let hs: HostTensor<f32, m![P128,1#8]> = xs[group].to_host(&mut d.pdma).await.unwrap();
            let hs = hs.into_vec();
            assert_eq!(hs.len(), P128::SIZE * 8);
            let qdiff = hq
                .iter()
                .zip(&cpu_q[group])
                .filter(|(a, b)| a.to_f32().to_bits() != b.to_f32().to_bits())
                .count();
            let sdiff = (0..P128::SIZE)
                .filter(|&b| hs[b * 8].to_bits() != cpu_scale[group][b].to_bits())
                .count();
            println!(
                "seed={seed} group={group} q_checked={} qdiff={qdiff} scalediff={sdiff}",
                hq.len()
            );
            assert_eq!(qdiff, 0);
            assert_eq!(sdiff, 0);
        }
        let (qw, qs, mut qo) = fixture!(
            d,
            seed,
            0,
            Qs,
            project_query_four,
            project_query_quantized,
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
            project_sliding_kv_four,
            project_sliding_kv_quantized,
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
            project_sliding_kv_four,
            project_sliding_kv_quantized,
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
            project_full_query_four,
            project_full_query_quantized,
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
            project_full_key_four,
            project_full_key_quantized,
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
                        for r in 0..4 {
                            launch(
                                control::quantize_input,
                                (&mut d, &input[r], &mut q[r], &mut xs[r]),
                            )
                            .await
                            .unwrap();
                            launch(
                                control::project_query_quantized,
                                (&mut d, &q[r], &xs[r], &qw, &qs, &mut qo[r]),
                            )
                            .await
                            .unwrap();
                            launch(
                                control::project_sliding_kv_quantized,
                                (&mut d, &q[r], &xs[r], &kw, &ks, &mut ko[r]),
                            )
                            .await
                            .unwrap();
                            launch(
                                control::project_sliding_kv_quantized,
                                (&mut d, &q[r], &xs[r], &vw, &vs, &mut vo[r]),
                            )
                            .await
                            .unwrap();
                        }
                    } else {
                        for r in 0..4 {
                            launch(
                                op::quantize_input,
                                (&mut d, &input[r], &mut q[r], &mut xs[r]),
                            )
                            .await
                            .unwrap();
                        }
                        four!(d, q, xs, qw, qs, qo, project_query_four, [0, 1, 2, 3]);
                        four!(d, q, xs, kw, ks, ko, project_sliding_kv_four, [0, 1, 2, 3]);
                        four!(d, q, xs, vw, vs, vo, project_sliding_kv_four, [0, 1, 2, 3]);
                    }
                }
                println!(
                    "four_sliding_QKV_ABBA variant={variant} warm_us={:.3}",
                    now.elapsed().as_secs_f64() * 1e6 / 20.0
                );
            }
            for variant in [0, 1, 1, 0] {
                let now = std::time::Instant::now();
                for _ in 0..20 {
                    if variant == 0 {
                        for r in 0..4 {
                            launch(
                                control::quantize_input,
                                (&mut d, &input[r], &mut q[r], &mut xs[r]),
                            )
                            .await
                            .unwrap();
                            launch(
                                control::project_full_query_quantized,
                                (&mut d, &q[r], &xs[r], &fqw, &fqs, &mut fqo[r]),
                            )
                            .await
                            .unwrap();
                            launch(
                                control::project_full_key_quantized,
                                (&mut d, &q[r], &xs[r], &fkw, &fks, &mut fko[r]),
                            )
                            .await
                            .unwrap();
                        }
                    } else {
                        for r in 0..4 {
                            launch(
                                op::quantize_input,
                                (&mut d, &input[r], &mut q[r], &mut xs[r]),
                            )
                            .await
                            .unwrap();
                        }
                        four!(
                            d,
                            q,
                            xs,
                            fqw,
                            fqs,
                            fqo,
                            project_full_query_four,
                            [0, 1, 2, 3]
                        );
                        four!(d, q, xs, fkw, fks, fko, project_full_key_four, [0, 1, 2, 3]);
                    }
                }
                println!(
                    "four_full_QK_ABBA variant={variant} warm_us={:.3}",
                    now.elapsed().as_secs_f64() * 1e6 / 20.0
                );
            }
        }
    }
}
