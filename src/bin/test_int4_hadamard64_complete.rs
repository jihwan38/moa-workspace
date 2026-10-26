//! Complete provided-layer0 fixed H64 numerical diagnostic. CPU rotation and
//! D2H/H2D are diagnostic-only; no timing or model/task-quality claim.
use furiosa_opt_gemma4::{
    axes::{H, L, P128, P512},
    host::{int4_complete_ffn_weights::ProductionWeights, load},
    i4_grouped_activation_raw_adapter as raw,
    ops_batch_layouts::{P512H960, P512L960},
    ops_ffn_projection_clusters as prod, ops_int4_ffn_bridges as bridge,
    ops_int4_grouped_activation_projection as candidate,
    ops_int4_grouped_activation_reference as reference, ops_int4_grouped_activation_store as fused,
    ops_norm_blocked_residual as tail, Chip,
};
use furiosa_opt_std::prelude::*;
use std::{fs::File, time::Instant};
#[path = "support/int4_grouped_activation_fixture.rs"]
mod cpu;
#[path = "support/int4_grouped_complete_fixture.rs"]
mod fixture;
#[path = "support/int4_hadamard64_complete_fixture.rs"]
mod rotate;
#[path = "support/int4_hadamard64_transform.rs"]
mod transform;
use cpu::{InputCpu, Result, WeightCpu, EXPANDED, HIDDEN, TOKENS};
use fused::{Mantissa, Record};
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
fn joined(a: &[u8], b: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(a.len() + b.len());
    v.extend_from_slice(a);
    v.extend_from_slice(b);
    v
}
fn compare_joined(got: &[u8], a: &[u8], b: &[u8], label: &str) {
    assert_eq!(got.len(), a.len() + b.len());
    assert_eq!(&got[..a.len()], a, "immutable UP plane {label}");
    assert_eq!(&got[a.len()..], b, "immutable GATE plane {label}");
}
async fn poison<E: M>(d: &mut Device, payload: u16) -> Result<HbmTensor<bf16, Chip, E>> {
    Ok(
        HostTensor::<bf16, E>::from_vec(std::iter::repeat_n(bf16::from_bits(payload), E::SIZE))
            .to_hbm(&mut d.pdma)
            .await?,
    )
}
struct InputGpu<E: M, S: M> {
    packed: HbmTensor<i4, Chip, E>,
    dense: HbmTensor<f8e4m3, Chip, E>,
    scales: HbmTensor<f32, Chip, S>,
}
impl<E: M, S: M> InputGpu<E, S> {
    async fn new(d: &mut Device, x: &InputCpu) -> Result<Self> {
        Ok(Self {
            packed: upload(d, x.packed.clone()).await?,
            dense: upload(d, x.dense.clone()).await?,
            scales: upload(d, floats(&x.scales)).await?,
        })
    }
    async fn immutable(&self, d: &mut Device, x: &InputCpu) -> Result<usize> {
        assert_eq!(read(d, &self.packed).await?, x.packed, "I4 input immutable");
        assert_eq!(read(d, &self.dense).await?, x.dense, "E4 input immutable");
        assert_eq!(
            read(d, &self.scales).await?,
            floats(&x.scales),
            "input scales immutable"
        );
        Ok(x.packed.len() + x.dense.len() + x.scales.len() * 4)
    }
}
struct WeightsGpu {
    joined_i4: HbmTensor<i4, Chip, candidate::JoinedWeight>,
    joined_e4: HbmTensor<f8e4m3, Chip, reference::JoinedWeight>,
    joined_scales: HbmTensor<f32, Chip, candidate::JoinedBlockScale>,
    down_i4: HbmTensor<i4, Chip, candidate::DownWeight>,
    down_e4: HbmTensor<f8e4m3, Chip, candidate::DownWeight>,
    down_scales: HbmTensor<f32, Chip, candidate::DownBlockScale>,
    global: HbmTensor<f32, Chip, m![1]>,
}
impl WeightsGpu {
    async fn new(
        d: &mut Device,
        up: &WeightCpu,
        gate: &WeightCpu,
        down: &WeightCpu,
    ) -> Result<Self> {
        assert_eq!((up.rows, up.cols), (EXPANDED, HIDDEN));
        assert_eq!((gate.rows, gate.cols), (EXPANDED, HIDDEN));
        assert_eq!((down.rows, down.cols), (HIDDEN, EXPANDED));
        Ok(Self {
            joined_i4: upload(d, joined(&up.packed, &gate.packed)).await?,
            joined_e4: upload(d, joined(&up.dense, &gate.dense)).await?,
            joined_scales: upload(d, joined(&floats(&up.scales), &floats(&gate.scales))).await?,
            down_i4: upload(d, down.packed.clone()).await?,
            down_e4: upload(d, down.dense.clone()).await?,
            down_scales: upload(d, floats(&down.scales)).await?,
            global: upload(d, down.global.to_le_bytes().to_vec()).await?,
        })
    }
    async fn immutable(
        &self,
        d: &mut Device,
        up: &WeightCpu,
        gate: &WeightCpu,
        down: &WeightCpu,
    ) -> Result<usize> {
        compare_joined(
            &read(d, &self.joined_i4).await?,
            &up.packed,
            &gate.packed,
            "I4",
        );
        compare_joined(
            &read(d, &self.joined_e4).await?,
            &up.dense,
            &gate.dense,
            "E4",
        );
        compare_joined(
            &read(d, &self.joined_scales).await?,
            &floats(&up.scales),
            &floats(&gate.scales),
            "scales",
        );
        assert_eq!(
            read(d, &self.down_i4).await?,
            down.packed,
            "DOWN I4 immutable"
        );
        assert_eq!(
            read(d, &self.down_e4).await?,
            down.dense,
            "DOWN E4 immutable"
        );
        assert_eq!(
            read(d, &self.down_scales).await?,
            floats(&down.scales),
            "DOWN scales immutable"
        );
        assert_eq!(
            read(d, &self.global).await?,
            down.global.to_le_bytes(),
            "DOWN global immutable"
        );
        Ok(up.packed.len()
            + gate.packed.len()
            + down.packed.len()
            + up.dense.len()
            + gate.dense.len()
            + down.dense.len()
            + (up.scales.len() + gate.scales.len() + down.scales.len() + 1) * 4)
    }
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
struct Quantizer {
    table: HbmTensor<i32, Chip, m![Mantissa, Record]>,
    hshift: HbmTensor<i32, Chip, m![H % 8]>,
    lshift: HbmTensor<i32, Chip, m![L % 8]>,
    table_bytes: Vec<u8>,
    shift_bytes: Vec<u8>,
}
impl Quantizer {
    async fn new(d: &mut Device) -> Result<Self> {
        let mut table_bytes = Vec::new();
        for m in 128u32..256 {
            let u = 7 * ((1u32 << 20).div_ceil(m));
            for lane in 0..8u32 {
                let b = if lane == 0 {
                    (u as f32).to_bits()
                } else {
                    0x5a3c0000 | (m << 4) | lane
                };
                table_bytes.extend(b.to_le_bytes());
            }
        }
        let shift_bytes: Vec<u8> = (0..8i32).flat_map(|i| (4 * i).to_le_bytes()).collect();
        Ok(Self {
            table: upload(d, table_bytes.clone()).await?,
            hshift: upload(d, shift_bytes.clone()).await?,
            lshift: upload(d, shift_bytes.clone()).await?,
            table_bytes,
            shift_bytes,
        })
    }
    async fn immutable(&self, d: &mut Device) -> Result<usize> {
        assert_eq!(read(d, &self.table).await?, self.table_bytes);
        assert_eq!(read(d, &self.hshift).await?, self.shift_bytes);
        assert_eq!(read(d, &self.lshift).await?, self.shift_bytes);
        Ok(self.table_bytes.len() + 2 * self.shift_bytes.len())
    }
}
struct CandidateScratch {
    residual: HbmTensor<bf16, Chip, tail::Residual>,
    prefix: HbmTensor<bf16, Chip, bridge::InputBf16>,
    rotated_prefix: HbmTensor<bf16, Chip, bridge::InputBf16>,
    xp: HbmTensor<i32, Chip, raw::PackedInput>,
    xs: HbmTensor<f32, Chip, candidate::InputScale>,
    xo: HbmTensor<i32, Chip, candidate::InputScale>,
    joined: HbmTensor<bf16, Chip, candidate::JoinedOutput>,
    hidden: HbmTensor<bf16, Chip, bridge::HiddenBf16>,
    rotated_hidden: HbmTensor<bf16, Chip, bridge::HiddenBf16>,
    hp: HbmTensor<i32, Chip, raw::PackedHidden>,
    hs: HbmTensor<f32, Chip, candidate::HiddenScale>,
    ho: HbmTensor<i32, Chip, candidate::HiddenScale>,
    down: HbmTensor<bf16, Chip, candidate::Output>,
}
#[derive(Clone)]
struct Snapshot {
    residual: Vec<u8>,
    prefix: Vec<u8>,
    rotated_prefix: Vec<u8>,
    xp: Vec<u8>,
    xs: Vec<u8>,
    xo: Vec<u8>,
    joined: Vec<u8>,
    hidden: Vec<u8>,
    rotated_hidden: Vec<u8>,
    hp: Vec<u8>,
    hs: Vec<u8>,
    ho: Vec<u8>,
    down: Vec<u8>,
}
impl Snapshot {
    fn fields(&self) -> [(&str, &[u8]); 13] {
        [
            ("residual", &self.residual),
            ("prefix", &self.prefix),
            ("rotated_prefix", &self.rotated_prefix),
            ("xp", &self.xp),
            ("xs", &self.xs),
            ("xo", &self.xo),
            ("joined", &self.joined),
            ("hidden", &self.hidden),
            ("rotated_hidden", &self.rotated_hidden),
            ("hp", &self.hp),
            ("hs", &self.hs),
            ("ho", &self.ho),
            ("down", &self.down),
        ]
    }
    fn equal(&self, other: &Self, label: &str) -> usize {
        let mut n = 0;
        for ((a, x), (b, y)) in self.fields().into_iter().zip(other.fields()) {
            assert_eq!(a, b);
            assert_eq!(x, y, "{label} {a}");
            n += x.len();
        }
        n
    }
    fn permutation(&self, base: &Self) -> usize {
        fixture::compare_rows(
            &fixture::canonical_residual(&self.residual),
            &fixture::canonical_residual(&base.residual),
            HIDDEN,
            2,
            "residual",
        );
        for (a, b, width, label) in [
            (&self.prefix, &base.prefix, HIDDEN, "prefix"),
            (
                &self.rotated_prefix,
                &base.rotated_prefix,
                HIDDEN,
                "rotated_prefix",
            ),
            (&self.hidden, &base.hidden, EXPANDED, "hidden"),
            (
                &self.rotated_hidden,
                &base.rotated_hidden,
                EXPANDED,
                "rotated_hidden",
            ),
        ] {
            fixture::compare_rows(a, b, width, 2, label);
        }
        cpu::compare_permutation(&self.joined, &base.joined, EXPANDED, 2);
        cpu::compare_permutation(&self.down, &base.down, HIDDEN, 1);
        for (a, b, width, label) in [
            (&self.xs, &base.xs, HIDDEN, "Hscale"),
            (&self.xo, &base.xo, HIDDEN, "Hoffset"),
            (&self.hs, &base.hs, EXPANDED, "Lscale"),
            (&self.ho, &base.ho, EXPANDED, "Loffset"),
        ] {
            fixture::compare_group_rows(a, b, width, label);
        }
        let x = fixture::quantize(&self.rotated_prefix, HIDDEN);
        let bx = fixture::quantize(&base.rotated_prefix, HIDDEN);
        let h = fixture::quantize(&self.rotated_hidden, EXPANDED);
        let bh = fixture::quantize(&base.rotated_hidden, EXPANDED);
        assert_eq!(self.xp, x.packed);
        assert_eq!(base.xp, bx.packed);
        assert_eq!(self.hp, h.packed);
        assert_eq!(base.hp, bh.packed);
        for p in 0..512 {
            assert_eq!(
                &x.codes[p * HIDDEN..(p + 1) * HIDDEN],
                &bx.codes[cpu::perm(p) * HIDDEN..(cpu::perm(p) + 1) * HIDDEN]
            );
            assert_eq!(
                &h.codes[p * EXPANDED..(p + 1) * EXPANDED],
                &bh.codes[cpu::perm(p) * EXPANDED..(cpu::perm(p) + 1) * EXPANDED]
            );
        }
        // Physical owner extents, not sums of helper counters with mixed units.
        self.fields().iter().map(|(_, b)| b.len()).sum()
    }
}
impl CandidateScratch {
    async fn new(d: &mut Device, r: &[u8]) -> Result<Self> {
        Ok(Self {
            residual: upload(d, r.to_vec()).await?,
            prefix: poison(d, 0x7fc1).await?,
            rotated_prefix: poison(d, 0x7fd1).await?,
            xp: upload(d, vec![0x5a; 512 * HIDDEN / 2]).await?,
            xs: upload(d, floats(&vec![f32::NAN; 512 * HIDDEN / 960])).await?,
            xo: upload(d, vec![0xa7; 512 * HIDDEN / 960 * 4]).await?,
            joined: poison(d, 0x7fc2).await?,
            hidden: poison(d, 0x7fc3).await?,
            rotated_hidden: poison(d, 0x7fd3).await?,
            hp: upload(d, vec![0xa5; 512 * EXPANDED / 2]).await?,
            hs: upload(d, floats(&vec![f32::NAN; 512 * EXPANDED / 960])).await?,
            ho: upload(d, vec![0xb9; 512 * EXPANDED / 960 * 4]).await?,
            down: poison(d, 0x7fc4).await?,
        })
    }
    async fn reset(&mut self, d: &mut Device, r: &[u8]) -> Result<()> {
        self.residual = upload(d, r.to_vec()).await?;
        Ok(())
    }
    async fn snapshot(&self, d: &mut Device) -> Result<Snapshot> {
        Ok(Snapshot {
            residual: read(d, &self.residual).await?,
            prefix: read(d, &self.prefix).await?,
            rotated_prefix: read(d, &self.rotated_prefix).await?,
            xp: read(d, &self.xp).await?,
            xs: read(d, &self.xs).await?,
            xo: read(d, &self.xo).await?,
            joined: read(d, &self.joined).await?,
            hidden: read(d, &self.hidden).await?,
            rotated_hidden: read(d, &self.rotated_hidden).await?,
            hp: read(d, &self.hp).await?,
            hs: read(d, &self.hs).await?,
            ho: read(d, &self.ho).await?,
            down: read(d, &self.down).await?,
        })
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
async fn candidate_prefix(
    d: &mut Device,
    p: &ProductionWeights,
    x: &Inputs,
    s: &mut CandidateScratch,
) -> Result<()> {
    launch(
        bridge::prefix_sliding_bf16,
        (
            d,
            &x.sliding[0],
            &x.sliding[1],
            &x.sliding[2],
            &x.sliding[3],
            &p.post_attention,
            &p.pre_ffn,
            &mut s.residual,
            &mut s.prefix,
        ),
    )
    .await?;
    Ok(())
}
async fn candidate_rest(
    d: &mut Device,
    p: &ProductionWeights,
    w: &WeightsGpu,
    q: &Quantizer,
    s: &mut CandidateScratch,
    rotated: bool,
) -> Result<()> {
    let prefix = read(d, &s.prefix).await?;
    let quant_prefix = if rotated {
        transform::rotate_bytes(&prefix, HIDDEN, false)?
    } else {
        prefix
    };
    s.rotated_prefix = upload(d, quant_prefix).await?;
    launch(
        fused::quantize_pack_input,
        (
            &mut *d,
            &s.rotated_prefix,
            &q.table,
            &q.hshift,
            &mut s.xp,
            &mut s.xs,
            &mut s.xo,
        ),
    )
    .await?;
    raw::project_upgate(
        d,
        &s.xp,
        &s.xs,
        &w.joined_i4,
        &w.joined_scales,
        &mut s.joined,
    )
    .await?;
    launch(
        bridge::geglu_bf16,
        (&mut *d, &s.joined, p.ug(), p.gg(), &mut s.hidden),
    )
    .await?;
    let hidden = read(d, &s.hidden).await?;
    let quant_hidden = if rotated {
        transform::rotate_bytes(&hidden, EXPANDED, false)?
    } else {
        hidden
    };
    s.rotated_hidden = upload(d, quant_hidden).await?;
    launch(
        fused::quantize_pack_hidden,
        (
            &mut *d,
            &s.rotated_hidden,
            &q.table,
            &q.lshift,
            &mut s.hp,
            &mut s.hs,
            &mut s.ho,
        ),
    )
    .await?;
    raw::project_down(
        d,
        &s.hp,
        &s.hs,
        &w.down_i4,
        &w.down_scales,
        &w.global,
        &mut s.down,
    )
    .await?;
    launch(
        tail::post_feedforward,
        (
            &mut *d,
            &s.down,
            &p.post_ffn,
            &p.layer_gate,
            &mut s.residual,
        ),
    )
    .await?;
    Ok(())
}
async fn candidate_chain(
    d: &mut Device,
    p: &ProductionWeights,
    w: &WeightsGpu,
    q: &Quantizer,
    x: &Inputs,
    s: &mut CandidateScratch,
    rotated: bool,
) -> Result<()> {
    candidate_prefix(d, p, x, s).await?;
    candidate_rest(d, p, w, q, s, rotated).await
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

/// Separate original P128 norm paths, CPU quotient, E4 integer-code dots,
/// original P128 GeGLU and original P128 tail. All copies/downloads are untimed.
async fn intended_step(
    d: &mut Device,
    p: &ProductionWeights,
    w: &WeightsGpu,
    x: &Inputs,
    residual: &[u8],
    got: &Snapshot,
    mid_got: &[u8],
    up: &WeightCpu,
    gate: &WeightCpu,
    down: &WeightCpu,
    round: usize,
    step: usize,
    rotated: bool,
) -> Result<(Vec<u8>, usize, usize, usize)> {
    let mut prefix = Vec::new();
    let mut mid = Vec::new();
    for quarter in 0..4 {
        let begin = quarter * 128 * HIDDEN * 2;
        let end = begin + 128 * HIDDEN * 2;
        let projected: HbmTensor<bf16, Chip, m![P128, H]> =
            upload(d, x.canonical[begin..end].to_vec()).await?;
        let mut r: HbmTensor<bf16, Chip, m![P128, H]> =
            upload(d, residual[begin..end].to_vec()).await?;
        let mut n = poison::<m![P128, H]>(d, 0x7fc7).await?;
        launch(
            furiosa_opt_gemma4::ops_norm_wide::post_attention_p128,
            (&mut *d, &projected, &p.post_attention, &mut r),
        )
        .await?;
        launch(
            furiosa_opt_gemma4::ops_norm_wide::rmsnorm_p128,
            (&mut *d, &r, &p.pre_ffn, &mut n),
        )
        .await?;
        mid.extend(read(d, &r).await?);
        prefix.extend(read(d, &n).await?);
        assert_eq!(
            read(d, &projected).await?,
            x.canonical[begin..end],
            "immutable original prefix input"
        );
    }
    let mut exact = cpu::compare_bits(&got.prefix, &prefix, "independent prefix canonical BF16");
    exact += cpu::compare_bits(
        mid_got,
        &fixture::block_residual(&mid),
        "independent prefix persisted residual",
    );
    let quant_prefix = if rotated {
        transform::rotate_bytes(&prefix, HIDDEN, true)?
    } else {
        prefix.clone()
    };
    exact += cpu::compare_bits(
        &got.rotated_prefix,
        &quant_prefix,
        "independent CPU rotated/identity prefix BF16",
    );
    let qi = fixture::quantize(&quant_prefix, HIDDEN);
    assert_eq!(got.xp, qi.packed);
    assert_eq!(got.xs, floats(&qi.scales));
    assert_eq!(got.xo, fixture::offset_bytes(&quant_prefix, HIDDEN));
    fixture::quantization_metrics(
        &quant_prefix,
        &qi,
        round,
        step,
        if rotated {
            "rotated_prefix_own_coordinates"
        } else {
            "original_prefix"
        },
    );
    let xg = InputGpu::<P512H960, candidate::InputScale>::new(d, &qi).await?;
    let mut typed = poison::<candidate::JoinedOutput>(d, 0x7fc8).await?;
    let mut dots = poison::<reference::JoinedOutput>(d, 0x7fc9).await?;
    // Normal typed launches embed exactly the images used by the checked raw binding.
    launch(
        candidate::project_upgate,
        (
            &mut *d,
            &xg.packed,
            &xg.scales,
            &w.joined_i4,
            &w.joined_scales,
            &mut typed,
        ),
    )
    .await?;
    launch(
        reference::project_upgate,
        (
            &mut *d,
            &xg.dense,
            &xg.scales,
            &w.joined_e4,
            &w.joined_scales,
            &mut dots,
        ),
    )
    .await?;
    let jb = read(d, &dots).await?;
    exact += cpu::compare_bits(&got.joined, &jb, "intended E4-code UP/GATE");
    exact += cpu::compare_bits(&read(d, &typed).await?, &jb, "typed/raw UP/GATE image");
    let (mut sampled, mut immutable) = (0, 0);
    for (name, plane, weight) in [("UP", 0, up), ("GATE", 1, gate)] {
        let (n, radius) = cpu::sampled_bound(&got.joined, plane, &qi, weight, false);
        sampled += n;
        println!("I4_H64_COMPLETE_F64 round={round} step={step} projection={name} samples={n} max_preBF16_radius={radius:.12e} oracle=i64_local_F64_envelope");
    }
    let upcan = fixture::canonical_projection(&jb, EXPANDED, 0);
    let gatecan = fixture::canonical_projection(&jb, EXPANDED, 1);
    let mut hidden = Vec::with_capacity(512 * EXPANDED * 2);
    for quarter in 0..4 {
        let begin = quarter * 128 * EXPANDED * 2;
        let end = begin + 128 * EXPANDED * 2;
        let ub = fixture::reference_column_tile(&upcan[begin..end]);
        let gb = fixture::reference_column_tile(&gatecan[begin..end]);
        let u: HbmTensor<bf16, Chip, furiosa_opt_gemma4::ops_int4_complete_reference::Operand> =
            upload(d, ub.clone()).await?;
        let g: HbmTensor<bf16, Chip, furiosa_opt_gemma4::ops_int4_complete_reference::Operand> =
            upload(d, gb.clone()).await?;
        let mut h =
            poison::<furiosa_opt_gemma4::ops_int4_complete_reference::Operand>(d, 0x7fca).await?;
        launch(
            furiosa_opt_gemma4::ops_int4_complete_reference::geglu_p128,
            (&mut *d, &u, &g, p.ug(), p.gg(), &mut h),
        )
        .await?;
        hidden.extend(fixture::reference_canonical_tile(&read(d, &h).await?));
        assert_eq!(read(d, &u).await?, ub);
        assert_eq!(read(d, &g).await?, gb);
    }
    exact += cpu::compare_bits(&got.hidden, &hidden, "independent original P128 GeGLU BF16");
    let quant_hidden = if rotated {
        transform::rotate_bytes(&hidden, EXPANDED, true)?
    } else {
        hidden.clone()
    };
    exact += cpu::compare_bits(
        &got.rotated_hidden,
        &quant_hidden,
        "independent CPU rotated/identity OWN hidden BF16",
    );
    let qh = fixture::quantize(&quant_hidden, EXPANDED);
    assert_eq!(got.hp, qh.packed);
    assert_eq!(got.hs, floats(&qh.scales));
    assert_eq!(got.ho, fixture::offset_bytes(&quant_hidden, EXPANDED));
    fixture::quantization_metrics(
        &quant_hidden,
        &qh,
        round,
        step,
        if rotated {
            "rotated_candidate_own_hidden_coordinates"
        } else {
            "original_candidate_own_hidden"
        },
    );
    let hg = InputGpu::<P512L960, candidate::HiddenScale>::new(d, &qh).await?;
    let mut typed_down = poison::<candidate::Output>(d, 0x7fcb).await?;
    let mut ref_down = poison::<candidate::Output>(d, 0x7fcc).await?;
    launch(
        candidate::project_down,
        (
            &mut *d,
            &hg.packed,
            &hg.scales,
            &w.down_i4,
            &w.down_scales,
            &w.global,
            &mut typed_down,
        ),
    )
    .await?;
    launch(
        reference::project_down,
        (
            &mut *d,
            &hg.dense,
            &hg.scales,
            &w.down_e4,
            &w.down_scales,
            &w.global,
            &mut ref_down,
        ),
    )
    .await?;
    let db = read(d, &ref_down).await?;
    exact += cpu::compare_bits(&got.down, &db, "intended E4-code DOWN");
    exact += cpu::compare_bits(&read(d, &typed_down).await?, &db, "typed/raw DOWN image");
    let (n, radius) = cpu::sampled_bound(&got.down, 0, &qh, down, true);
    sampled += n;
    println!("I4_H64_COMPLETE_F64 round={round} step={step} projection=DOWN samples={n} max_preBF16_radius={radius:.12e} oracle=i64_local_F64_envelope");
    let downcan = fixture::canonical_projection(&db, HIDDEN, 0);
    let mut final_residual = Vec::new();
    for quarter in 0..4 {
        let begin = quarter * 128 * HIDDEN * 2;
        let end = begin + 128 * HIDDEN * 2;
        let y: HbmTensor<bf16, Chip, m![P128, H]> = upload(d, downcan[begin..end].to_vec()).await?;
        let mut r: HbmTensor<bf16, Chip, m![P128, H]> = upload(d, mid[begin..end].to_vec()).await?;
        launch(
            furiosa_opt_gemma4::ops_norm_wide::post_feedforward_p128,
            (&mut *d, &y, &p.post_ffn, &p.layer_gate, &mut r),
        )
        .await?;
        final_residual.extend(read(d, &r).await?);
        assert_eq!(read(d, &y).await?, downcan[begin..end]);
    }
    exact += cpu::compare_bits(
        &got.residual,
        &fixture::block_residual(&final_residual),
        "independent complete residual/layer scalar",
    );
    immutable += xg.immutable(d, &qi).await? + hg.immutable(d, &qh).await?;
    assert_eq!(read(d, &dots).await?, jb);
    assert_eq!(read(d, &ref_down).await?, db);
    Ok((final_residual, exact, sampled, immutable))
}

#[tokio::main]
async fn main() -> Result<()> {
    if std::env::args().len() != 1 {
        return Err(
            "fixed layer0/two roworders/two common-production steps; no tuning arguments".into(),
        );
    }
    cpu::validate_integer_encodings();
    cpu::check_output_layout(EXPANDED);
    cpu::check_output_layout(HIDDEN);
    let file = File::open(load::model_dir()?.join("model.safetensors"))?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let tensors = safetensors::SafeTensors::deserialize(&mmap)?;
    let t = Instant::now();
    let up = WeightCpu::load(&tensors, "up_proj")?;
    let gate = WeightCpu::load(&tensors, "gate_proj")?;
    let down = WeightCpu::load(&tensors, "down_proj")?;
    println!("I4_H64_COMPLETE_SETUP original_decode_encode_ms={:.6} layer=0 no_all48_or_runtime_claim=true",t.elapsed().as_secs_f64()*1e3);
    let t = Instant::now();
    let rotated_up = rotate::weights(&tensors, "up_proj", &up)?;
    let rotated_gate = rotate::weights(&tensors, "gate_proj", &gate)?;
    let rotated_down = rotate::weights(&tensors, "down_proj", &down)?;
    println!("I4_H64_COMPLETE_SETUP rotated_decode_encode_allrow_independent_audit_ms={:.6} layer=0 no_all48_or_runtime_claim=true",t.elapsed().as_secs_f64()*1e3);
    let mut d = Device::new(candidate::project_upgate.topology())?;
    let p = ProductionWeights::load_layer0(&mut d, &tensors).await?;
    for (w, g) in [(&up, p.ug()), (&gate, p.gg()), (&down, p.dg())] {
        assert_eq!(read(&mut d, g).await?, w.global.to_le_bytes());
    }
    let old_weights = WeightsGpu::new(&mut d, &up, &gate, &down).await?;
    let rotated_weights =
        WeightsGpu::new(&mut d, &rotated_up, &rotated_gate, &rotated_down).await?;
    let q = Quantizer::new(&mut d).await?;
    let production_before = p.fingerprints(&mut d).await?;
    println!("I4_H64_COMPLETE_OWNERS additional_packed={} additional_dense_reference={} additional_scales={} activation_rotation=CPU_D2H_H2D no_timing=true",3*HIDDEN*EXPANDED/2,3*HIDDEN*EXPANDED,3*HIDDEN*EXPANDED/960*4+4);
    let (mut exact, mut samples, mut permutation_bytes, mut immutable, mut same_input, mut reuse) =
        (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    let (mut rotated_base, mut original_base) = (Vec::new(), Vec::new());
    let mut numerical_gate = true;
    for round in 0..2 {
        let x = Inputs::new(&mut d, round == 1).await?;
        let mut incoming = fixture::input(round == 1, true);
        for step in 0..2 {
            let rb = fixture::block_residual(&incoming);
            let mut rotated = CandidateScratch::new(&mut d, &rb).await?;
            let mut original = CandidateScratch::new(&mut d, &rb).await?;
            let mut production = ProductionScratch::new(&mut d, &rb).await?;
            candidate_prefix(&mut d, &p, &x, &mut rotated).await?;
            let midpoint = read(&mut d, &rotated.residual).await?;
            candidate_rest(&mut d, &p, &rotated_weights, &q, &mut rotated, true).await?;
            let got = rotated.snapshot(&mut d).await?;
            candidate_prefix(&mut d, &p, &x, &mut original).await?;
            assert_eq!(read(&mut d, &original.residual).await?, midpoint);
            same_input += midpoint.len();
            candidate_rest(&mut d, &p, &old_weights, &q, &mut original, false).await?;
            let old = original.snapshot(&mut d).await?;
            assert_eq!(got.prefix, old.prefix, "same raw original BF16 prefix");
            same_input += got.prefix.len();
            println!("I4_H64_COMPLETE_INTENDED_BEGIN arm=rotated round={round} step={step}");
            let (_, n, ns, ni) = intended_step(
                &mut d,
                &p,
                &rotated_weights,
                &x,
                &incoming,
                &got,
                &midpoint,
                &rotated_up,
                &rotated_gate,
                &rotated_down,
                round,
                step,
                true,
            )
            .await?;
            exact += n;
            samples += ns;
            immutable += ni;
            println!("I4_H64_COMPLETE_INTENDED_BEGIN arm=original round={round} step={step}");
            let (_, n, ns, ni) = intended_step(
                &mut d,
                &p,
                &old_weights,
                &x,
                &incoming,
                &old,
                &midpoint,
                &up,
                &gate,
                &down,
                round,
                step,
                false,
            )
            .await?;
            exact += n;
            samples += ns;
            immutable += ni;
            production_chain(&mut d, &p, &x, &mut production).await?;
            let current = production.snapshot(&mut d).await?;
            let mut production_hidden = poison::<bridge::HiddenBf16>(&mut d, 0x7fda).await?;
            launch(
                bridge::geglu_bf16,
                (
                    &mut d,
                    &production.joined,
                    p.ug(),
                    p.gg(),
                    &mut production_hidden,
                ),
            )
            .await?;
            let ph = read(&mut d, &production_hidden).await?;
            let mut gate = true;
            for (label, new, old, prod) in [
                ("joined", &got.joined, &old.joined, &current[3]),
                ("hidden", &got.hidden, &old.hidden, &ph),
                ("down", &got.down, &old.down, &current[6]),
                ("residual", &got.residual, &old.residual, &current[0]),
            ] {
                let a = rotate::error(new, prod);
                let b = rotate::error(old, prod);
                rotate::report(&a, &format!("rotated_vs_q020_{label}"), round, step);
                rotate::report(&b, &format!("original_vs_q020_{label}"), round, step);
                if label == "residual" {
                    gate &= a.relative_l2 <= 0.75 * b.relative_l2;
                }
                if label == "joined" || label == "down" {
                    gate &= a.relative_l2 <= b.relative_l2;
                }
            }
            numerical_gate &= gate;
            if round == 0 {
                rotated_base.push(got.clone());
                original_base.push(old.clone());
            } else {
                permutation_bytes +=
                    got.permutation(&rotated_base[step]) + old.permutation(&original_base[step]);
            }
            // All post-oracle/CPU inputs are read BEFORE replay can overwrite them.
            immutable += rotated
                .snapshot(&mut d)
                .await?
                .equal(&got, "rotated before replay");
            immutable += original
                .snapshot(&mut d)
                .await?
                .equal(&old, "original before replay");
            for (a, b) in production.snapshot(&mut d).await?.iter().zip(&current) {
                assert_eq!(a, b);
                immutable += a.len();
            }
            assert_eq!(read(&mut d, &production_hidden).await?, ph);
            immutable += ph.len();
            rotated.reset(&mut d, &rb).await?;
            candidate_chain(&mut d, &p, &rotated_weights, &q, &x, &mut rotated, true).await?;
            reuse += rotated
                .snapshot(&mut d)
                .await?
                .equal(&got, "rotated full reuse");
            reuse += original
                .snapshot(&mut d)
                .await?
                .equal(&old, "original owners unchanged by rotated replay");
            for (a, b) in production.snapshot(&mut d).await?.iter().zip(&current) {
                assert_eq!(a, b);
                reuse += a.len();
            }
            assert_eq!(read(&mut d, &production_hidden).await?, ph);
            reuse += ph.len();
            incoming = fixture::canonical_residual(&current[0]);
            immutable += x.immutable(&mut d).await?;
            println!("I4_H64_COMPLETE_STEP_PASS round={round} step={step} intended_code_parity=true matched_incoming=true common_q020_feedback=true numerical_gate_passed={gate}");
        }
        immutable += q.immutable(&mut d).await?
            + old_weights.immutable(&mut d, &up, &gate, &down).await?
            + rotated_weights
                .immutable(&mut d, &rotated_up, &rotated_gate, &rotated_down)
                .await?;
        for w in [&up, &gate, &down, &rotated_up, &rotated_gate, &rotated_down] {
            w.verify_source(&tensors)?;
        }
    }
    assert_eq!(
        p.fingerprints(&mut d).await?,
        production_before,
        "original retained tensors immutable"
    );
    assert_eq!(exact, 471_859_200);
    assert_eq!(samples, 12_288);
    assert_eq!(same_input, 31_457_280);
    println!(
        "PASS_I4_H64_COMPLETE {}",
        serde_json::json!({"layer":0,"intended_bf16_values":exact,"i64_F64_samples":samples,
        "permutation_compared_owner_bytes":permutation_bytes,"same_raw_input_bytes":same_input,"immutable_exact_bytes":immutable,"reuse_bytes":reuse,
        "numerical_gate_passed":numerical_gate,"decision":if numerical_gate{"FULL_FFN_NUMERICAL_GATE_PASS_ONLY"}else{"HOLD"},
        "no_timing":true,"CPU_rotation_roundtrips":true,"no_model_quality_claim":true,"no_promotion":true})
    );
    Ok(())
}
