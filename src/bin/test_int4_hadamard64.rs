//! CPU-only fixed H64 projection diagnostic. Three unchanged runtime kernels
//! acquire BF16 inputs; no rotated device graph, timing or model-quality claim.
use furiosa_opt_gemma4::{
    axes::{H, P512},
    host::{int4_complete_ffn_weights::ProductionWeights, load},
    ops_batch_layouts::{P512H960, P512L960},
    ops_ffn_projection_clusters as prod, ops_int4_ffn_bridges as bridge,
    ops_norm_blocked_residual as tail, Chip,
};
use furiosa_opt_std::prelude::*;
use std::{fs::File, time::Instant};
#[path = "support/int4_grouped_activation_fixture.rs"]
mod cpu;
#[path = "support/int4_grouped_complete_fixture.rs"]
mod fixture;
#[path = "support/int4_hadamard64.rs"]
mod rotation;
use cpu::{Result, EXPANDED, HIDDEN, TOKENS};
type Sliding = furiosa_opt_gemma4::ops_output_p128_h64_blocked::Output;
async fn upload<D: MaterializableScalar, E: M>(
    d: &mut Device,
    b: Vec<u8>,
) -> Result<HbmTensor<D, Chip, E>> {
    assert_eq!(b.len() * 8, E::SIZE * D::BITS);
    Ok(HostTensor::<D, E>::from_buf(b).to_hbm(&mut d.pdma).await?)
}
async fn read<D: MaterializableScalar, E: M>(
    d: &mut Device,
    x: &HbmTensor<D, Chip, E>,
) -> Result<Vec<u8>> {
    Ok(x.to_host::<E>(&mut d.pdma).await?.into_inner().into_buf())
}
fn floats(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
async fn poison<E: M>(d: &mut Device, payload: u16) -> Result<HbmTensor<bf16, Chip, E>> {
    Ok(
        HostTensor::<bf16, E>::from_vec(std::iter::repeat_n(bf16::from_bits(payload), E::SIZE))
            .to_hbm(&mut d.pdma)
            .await?,
    )
}
struct Inputs {
    sliding: [HbmTensor<bf16, Chip, Sliding>; 4],
    canonical: Vec<u8>,
    wire: [Vec<u8>; 4],
}
impl Inputs {
    async fn new(d: &mut Device, permuted: bool) -> Result<Self> {
        let canonical = fixture::input(permuted, false);
        let wire = fixture::sliding(&canonical);
        let mut v = Vec::new();
        for b in &wire {
            v.push(upload(d, b.clone()).await?);
        }
        Ok(Self {
            sliding: v.try_into().ok().unwrap(),
            canonical,
            wire,
        })
    }
    async fn immutable(&self, d: &mut Device) -> Result<usize> {
        let mut n = 0;
        for (i, x) in self.sliding.iter().enumerate() {
            assert_eq!(read(d, x).await?, self.wire[i]);
            n += self.wire[i].len();
        }
        Ok(n)
    }
}
struct ProductionScratch {
    residual: HbmTensor<bf16, Chip, tail::Residual>,
    qi: HbmTensor<f8e4m3, Chip, P512H960>,
    xs: HbmTensor<f32, Chip, m![P512]>,
    joined: HbmTensor<bf16, Chip, prod::JoinedOutput>,
    qh: HbmTensor<f8e4m3, Chip, P512L960>,
    hs: HbmTensor<f32, Chip, m![P512]>,
    down: HbmTensor<bf16, Chip, furiosa_opt_gemma4::ops_native_fp8_p512_downblocked::Output>,
}
impl ProductionScratch {
    async fn new(d: &mut Device, r: &[u8]) -> Result<Self> {
        Ok(Self {
            residual: upload(d, r.to_vec()).await?,
            qi: upload(d, vec![0x7f; 512 * HIDDEN]).await?,
            xs: upload(d, floats(&vec![f32::NAN; 512])).await?,
            joined: poison(d, 0x7fc5).await?,
            qh: upload(d, vec![0x7f; 512 * EXPANDED]).await?,
            hs: upload(d, floats(&vec![f32::NAN; 512])).await?,
            down: poison(d, 0x7fc6).await?,
        })
    }
    async fn reset(&mut self, d: &mut Device, r: &[u8]) -> Result<()> {
        self.residual = upload(d, r.to_vec()).await?;
        Ok(())
    }
    async fn snapshot(&self, d: &mut Device) -> Result<Vec<Vec<u8>>> {
        Ok(vec![
            read(d, &self.residual).await?,
            read(d, &self.qi).await?,
            read(d, &self.xs).await?,
            read(d, &self.joined).await?,
            read(d, &self.qh).await?,
            read(d, &self.hs).await?,
            read(d, &self.down).await?,
        ])
    }
}
async fn production_chain(
    d: &mut Device,
    p: &ProductionWeights,
    x: &Inputs,
    s: &mut ProductionScratch,
) -> Result<()> {
    launch(
        furiosa_opt_gemma4::ops_group_ffn_projection_storegroup::sliding_projection_owned,
        (
            d,
            (&x.sliding[0], &x.sliding[1], &x.sliding[2], &x.sliding[3]),
            (&p.post_attention, &p.pre_ffn, &p.post_ffn, &p.layer_gate),
            &mut s.residual,
            &p.joined,
            &p.joined_scale,
            p.down(),
            p.down_scale(),
            p.ug(),
            p.gg(),
            p.dg(),
            &mut s.qi,
            &mut s.xs,
            &mut s.joined,
            &mut s.qh,
            &mut s.hs,
            &mut s.down,
        ),
    )
    .await?;
    Ok(())
}

struct SampledWeights {
    name: &'static str,
    rows: usize,
    cols: usize,
    global: f32,
    samples: Vec<rotation::Row>,
    source: Vec<(String, usize, u64)>,
}
impl SampledWeights {
    fn load(t: &safetensors::SafeTensors<'_>, name: &'static str) -> Result<Self> {
        let (rows, cols) = if name == "down_proj" {
            (HIDDEN, EXPANDED)
        } else {
            (EXPANDED, HIDDEN)
        };
        assert!(["up_proj", "gate_proj", "down_proj"].contains(&name));
        let stem = format!("model.language_model.layers.0.mlp.{name}");
        let names = [
            format!("{stem}.weight_packed"),
            format!("{stem}.weight_scale"),
            format!("{stem}.weight_global_scale"),
        ];
        let packed = t.tensor(&names[0])?;
        let scales = t.tensor(&names[1])?;
        let global = t.tensor(&names[2])?;
        assert_eq!(packed.dtype(), safetensors::Dtype::U8);
        assert_eq!(packed.shape(), [rows, cols / 2]);
        assert_eq!(scales.dtype(), safetensors::Dtype::F8_E4M3);
        assert_eq!(scales.shape(), [rows, cols / 16]);
        assert_eq!(global.dtype(), safetensors::Dtype::F32);
        assert_eq!(global.data().len(), 4);
        let stored_global = f32::from_le_bytes(global.data().try_into()?);
        assert!(stored_global.is_finite() && stored_global != 0.0);
        let global = 1.0f32 / stored_global;
        assert!(global.is_finite() && global != 0.0);
        let source = names
            .iter()
            .map(|n| {
                let b = t.tensor(n).unwrap();
                (n.clone(), b.data().len(), cpu::fingerprint(b.data()))
            })
            .collect();
        let mut samples = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        for owner in 0..TOKENS {
            let row = cpu::sample_row(owner, rows);
            assert!(seen.insert(row));
            let bits: Vec<u16> = (0..cols)
                .map(|k| {
                    let at = row * cols + k;
                    let nibble = (packed.data()[at / 2] >> (4 * (at % 2))) & 15;
                    let f4 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][(nibble & 7) as usize]
                        * if nibble & 8 == 0 { 1.0 } else { -1.0 };
                    let code = scales.data()[at / 16];
                    let e = (code >> 3) & 15;
                    let m = code & 7;
                    assert!(!(e == 15 && m == 7));
                    let magnitude = if e == 0 {
                        m as f32 / 512.0
                    } else {
                        (8 + m) as f32 * 2f32.powi(e as i32 - 10)
                    };
                    let scale = if code & 128 == 0 {
                        magnitude
                    } else {
                        -magnitude
                    };
                    // Original supplied BF16 boundary; native E4 row-dequant is absent.
                    let x = bf16::from_f32(f4 * scale).to_bits();
                    assert_eq!(x, rotation::bf16_rne(f4 * scale));
                    x
                })
                .collect();
            samples.push(rotation::Row::new(&bits)?);
        }
        Ok(Self {
            name,
            rows,
            cols,
            global,
            samples,
            source,
        })
    }
    fn verify(&self, t: &safetensors::SafeTensors<'_>) -> Result<usize> {
        let mut bytes = 0;
        for (name, n, hash) in &self.source {
            let v = t.tensor(name)?;
            assert_eq!(v.data().len(), *n);
            assert_eq!(
                cpu::fingerprint(v.data()),
                *hash,
                "immutable supplied tensor {name}"
            );
            bytes += n;
        }
        Ok(bytes)
    }
}

fn error_json(e: &rotation::Error) -> serde_json::Value {
    serde_json::json!({"samples":e.samples,"sse":e.sse,"reference_ss":e.reference_ss,"relative_l2":e.relative_l2(),"maxabs":e.maxabs})
}
struct Report {
    pair_bits: Vec<[u64; 11]>,
    passed: bool,
}
fn analyze(weights: &SampledWeights, input: &[u8], round: usize, step: usize) -> Result<Report> {
    assert_eq!(input.len(), TOKENS * weights.cols * 2);
    let before = cpu::fingerprint(input);
    let mut samples: Vec<Option<rotation::Sample>> = vec![None; TOKENS];
    let (mut xzero, mut xrzero, mut wzero, mut wrzero) = (0usize, 0usize, 0usize, 0usize);
    for p in 0..TOKENS {
        let owner = if round == 0 { p } else { cpu::perm(p) };
        let bits: Vec<u16> = input[p * weights.cols * 2..(p + 1) * weights.cols * 2]
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let x = rotation::Row::new(&bits)?;
        let w = &weights.samples[owner];
        xzero += x.original_q.zeros();
        xrzero += x.rotated_q.zeros();
        wzero += w.original_q.zeros();
        wrzero += w.rotated_q.zeros();
        assert!(samples[owner].is_none());
        samples[owner] = Some(rotation::compare(&x, w)?);
    }
    // Canonical-owner order makes ALL floating aggregates and pair bits identical
    // across the input row permutation, not merely statistically comparable.
    let samples: Vec<_> = samples.into_iter().map(Option::unwrap).collect();
    let mut errors: [rotation::Error; 9] = std::array::from_fn(|_| rotation::Error::default());
    let (mut max_identity, mut max_bound) = (0f64, 0f64);
    for s in &samples {
        for (i, e) in errors.iter_mut().enumerate() {
            e.add(s.dots[i], s.dots[0]);
        }
        max_identity = max_identity.max(s.identity_error);
        max_bound = max_bound.max(s.identity_bound);
    }
    if errors.iter().any(|e| {
        e.reference_ss <= 0.0
            || !e.reference_ss.is_finite()
            || !e.sse.is_finite()
            || !e.relative_l2().is_finite()
    }) {
        return Err(
            "zero/nonfinite sampled reference or ratio is outside the fixed relative gate".into(),
        );
    }
    let passed = rotation::decision(&errors);
    let values = TOKENS * weights.cols;
    let arms: serde_json::Map<String, serde_json::Value> = rotation::ARM_NAMES
        .iter()
        .zip(&errors)
        .map(|(n, e)| (n.to_string(), error_json(e)))
        .collect();
    println!(
        "I4_H64_DOTS {}",
        serde_json::json!({"round":round,"step":step,"projection":weights.name,"pairs":TOKENS,
        "terms_per_arm":values,"output_rows":weights.rows,"sample_row":"(317*canonical_owner+19)%rows","scope":"pre_global_FP64_projection_dots",
        "global_reported_not_applied":weights.global,"reference":"direct supplied BF16 weights and actual BF16 production activation",
        "matched_inputs":true,"feedback":"original q020 only; DOWN input is production GeGLU","arms":arms,
        "ideal_identity_maxabs":max_identity,"ideal_identity_max_bound":max_bound,
        "original_activation_zero_fraction":xzero as f64/values as f64,"rotated_activation_zero_fraction":xrzero as f64/values as f64,
        "original_weight_zero_fraction":wzero as f64/values as f64,"rotated_weight_zero_fraction":wrzero as f64/values as f64,
        "gate_rotation_only_max_l2":0.01,"gate_rotated_over_original_A4W4":0.75,"numerical_gate_passed":passed,
        "finite_normal_or_zero_scope_passed":true,"not_device_reduction_or_FFN_output":true,"no_task_quality_or_speed_claim":true})
    );
    assert_eq!(
        cpu::fingerprint(input),
        before,
        "immutable CPU activation buffer"
    );
    Ok(Report {
        pair_bits: samples.iter().map(rotation::Sample::bits).collect(),
        passed,
    })
}

async fn acquire(
    d: &mut Device,
    p: &ProductionWeights,
    x: &Inputs,
    residual: &[u8],
    production: &mut ProductionScratch,
    mid: &mut HbmTensor<bf16, Chip, tail::Residual>,
    prefix: &mut HbmTensor<bf16, Chip, bridge::InputBf16>,
    hidden: &mut HbmTensor<bf16, Chip, bridge::HiddenBf16>,
) -> Result<()> {
    *mid = upload(d, residual.to_vec()).await?;
    production.reset(d, residual).await?;
    launch(
        bridge::prefix_sliding_bf16,
        (
            &mut *d,
            &x.sliding[0],
            &x.sliding[1],
            &x.sliding[2],
            &x.sliding[3],
            &p.post_attention,
            &p.pre_ffn,
            mid,
            prefix,
        ),
    )
    .await?;
    production_chain(d, p, x, production).await?;
    launch(
        bridge::geglu_bf16,
        (&mut *d, &production.joined, p.ug(), p.gg(), hidden),
    )
    .await?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    if std::env::args().len() != 1 {
        return Err("fixed policy/layer0/512pairs/2orders/2steps; no tuning arguments".into());
    }
    let file = File::open(load::model_dir()?.join("model.safetensors"))?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let tensors = safetensors::SafeTensors::deserialize(&mmap)?;
    let start = Instant::now();
    let up = SampledWeights::load(&tensors, "up_proj")?;
    let gate = SampledWeights::load(&tensors, "gate_proj")?;
    let down = SampledWeights::load(&tensors, "down_proj")?;
    println!(
        "I4_H64_SETUP {}",
        serde_json::json!({"layer":0,"sampled_weight_rows":1536,"sampled_weight_values":11796480,
        "CPU_sampled_decode_rotate_quantize_ms":start.elapsed().as_secs_f64()*1e3,"no_all48_startup_or_runtime_speed_claim":true,
        "seed":"0x243f6a8885a308d3","policy":"R=D H64/8, six RN32 stages, single BF16 round","quantizer":"unchanged rational per960, no MSE",
        "source_fingerprints":[up.source,gate.source,down.source]})
    );
    let mut d = Device::new(bridge::prefix_sliding_bf16.topology())?;
    let p = ProductionWeights::load_layer0(&mut d, &tensors).await?;
    for (w, g) in [(&up, p.ug()), (&gate, p.gg()), (&down, p.dg())] {
        assert_eq!(read(&mut d, g).await?, w.global.to_le_bytes());
    }
    let weights_before = p.fingerprints(&mut d).await?;
    let (mut passed, mut pair_count, mut permutation_bytes, mut reuse_bytes, mut immutable_bytes) =
        (true, 0usize, 0usize, 0usize, 0usize);
    let mut pair_baseline: Vec<Vec<[u64; 11]>> = Vec::new();
    let mut state_baseline: Vec<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>)> = Vec::new();
    for round in 0..2 {
        let x = Inputs::new(&mut d, round == 1).await?;
        let mut incoming = fixture::input(round == 1, true);
        for step in 0..2 {
            let rb = fixture::block_residual(&incoming);
            let mut production = ProductionScratch::new(&mut d, &rb).await?;
            let mut mid = upload(&mut d, rb.clone()).await?;
            let mut prefix = poison::<bridge::InputBf16>(&mut d, 0x7fd1).await?;
            let mut hidden = poison::<bridge::HiddenBf16>(&mut d, 0x7fd2).await?;
            acquire(
                &mut d,
                &p,
                &x,
                &rb,
                &mut production,
                &mut mid,
                &mut prefix,
                &mut hidden,
            )
            .await?;
            let pb = read(&mut d, &prefix).await?;
            let hb = read(&mut d, &hidden).await?;
            let mb = read(&mut d, &mid).await?;
            let current = production.snapshot(&mut d).await?;
            let next = fixture::canonical_residual(&current[0]);
            for (w, input) in [(&up, &pb), (&gate, &pb), (&down, &hb)] {
                let report = analyze(w, input, round, step)?;
                passed &= report.passed;
                pair_count += report.pair_bits.len();
                if round == 0 {
                    pair_baseline.push(report.pair_bits);
                } else {
                    let index = step * 3
                        + match w.name {
                            "up_proj" => 0,
                            "gate_proj" => 1,
                            "down_proj" => 2,
                            _ => unreachable!(),
                        };
                    assert_eq!(
                        report.pair_bits, pair_baseline[index],
                        "owner-associated dot/identity bits under row permutation"
                    );
                    permutation_bytes += report.pair_bits.len() * 11 * 8;
                }
            }
            if round == 0 {
                state_baseline.push((pb.clone(), hb.clone(), incoming.clone(), next.clone()));
            } else {
                let base = &state_baseline[step];
                for (a, b, width, label) in [
                    (&pb, &base.0, HIDDEN, "prefix"),
                    (&hb, &base.1, EXPANDED, "production hidden"),
                    (&incoming, &base.2, HIDDEN, "incoming"),
                    (&next, &base.3, HIDDEN, "feedback"),
                ] {
                    permutation_bytes += fixture::compare_rows(a, b, width, 2, label);
                }
            }
            // These reads precede replay, so recomputation cannot hide mutation.
            for (got, old) in production.snapshot(&mut d).await?.iter().zip(&current) {
                assert_eq!(got, old, "production scratch unchanged during CPU analysis");
                immutable_bytes += got.len();
            }
            for (got, old) in [
                (read(&mut d, &prefix).await?, &pb),
                (read(&mut d, &hidden).await?, &hb),
                (read(&mut d, &mid).await?, &mb),
            ] {
                assert_eq!(&got, old);
                immutable_bytes += got.len();
            }
            immutable_bytes += x.immutable(&mut d).await?;
            acquire(
                &mut d,
                &p,
                &x,
                &rb,
                &mut production,
                &mut mid,
                &mut prefix,
                &mut hidden,
            )
            .await?;
            for (got, old) in production.snapshot(&mut d).await?.iter().zip(&current) {
                assert_eq!(got, old, "restored q020 replay");
                reuse_bytes += got.len();
            }
            for (got, old) in [
                (read(&mut d, &prefix).await?, &pb),
                (read(&mut d, &hidden).await?, &hb),
                (read(&mut d, &mid).await?, &mb),
            ] {
                assert_eq!(&got, old, "restored activation replay");
                reuse_bytes += got.len();
            }
            for w in [&up, &gate, &down] {
                w.verify(&tensors)?;
            }
            incoming = next;
            println!("I4_H64_STEP_PASS round={round} step={step} original_production_feedback=true full_owner_reuse=true");
        }
    }
    assert_eq!(
        p.fingerprints(&mut d).await?,
        weights_before,
        "all original retained production tensors immutable"
    );
    assert_eq!(pair_count, 6144);
    println!(
        "PASS_I4_H64_DIAGNOSTIC {}",
        serde_json::json!({"pairs_per_arm":pair_count,"arms":9,"terms_per_arm":47185920,
        "permutation_bytes":permutation_bytes,"reuse_bytes":reuse_bytes,"immutable_bytes_compared":immutable_bytes,
        "production_weights_full_read_fingerprints":weights_before,"numerical_gate_passed":passed,"decision":if passed{"SAMPLED_NUMERICAL_GATE_PASS_ONLY"}else{"HOLD"},
        "no_promotion":true,"no_task_quality_or_speed_claim":true,"device_rotation_unimplemented":true})
    );
    Ok(())
}
