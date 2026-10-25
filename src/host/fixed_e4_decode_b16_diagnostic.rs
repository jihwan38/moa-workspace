//! Rc32 decode storage port gate. Synthetic history; supplied O weights.
//! Candidate append receives q022-rounded BF16 codes; original raw-BF16 drift is diagnostic only.
use super::{fixed_e4_codec as codec, load};
use crate::{
    Chip, axes::*, ops_b16_attention_output as original, ops_fixed_e4_b16 as fixed,
    ops_pool_views::Rc64,
};
use furiosa_opt_std::prelude::*;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
type BPool = HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>;
type EPool = HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>;
type Kv = HbmTensor<bf16, Chip, m![B, Ns, Ds]>;
type Query = HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>;
const POOL: usize = Rc64::SIZE * 1024 * 8 * 256;
const KV: usize = 8 * 256;
const Q: usize = 8 * 2 * 256;
const SLOTS: [usize; 16] = [0, 15, 16, 31, 1, 14, 17, 30, 2, 13, 18, 29, 3, 12, 19, 28];
const POS: [usize; 16] = [
    256, 1023, 1024, 8191, 511, 767, 1535, 2047, 255, 512, 768, 4095, 257, 1025, 8190, 0,
];
async fn ub<Mp: M>(d: &mut Device, b: &[u8]) -> Result<HbmTensor<bf16, Chip, Mp>> {
    assert_eq!(b.len(), Mp::SIZE * 2);
    Ok(HostTensor::<bf16, Mp>::from_buf(b.to_vec())
        .to_hbm(&mut d.pdma)
        .await?)
}
async fn ue<Mp: M>(d: &mut Device, b: &[u8]) -> Result<HbmTensor<f8e4m3, Chip, Mp>> {
    assert_eq!(b.len(), Mp::SIZE);
    Ok(HostTensor::<f8e4m3, Mp>::from_buf(b.to_vec())
        .to_hbm(&mut d.pdma)
        .await?)
}
async fn rb<Mp: M>(d: &mut Device, x: &HbmTensor<bf16, Chip, Mp>) -> Result<Vec<u8>> {
    Ok(x.to_host::<Mp>(&mut d.pdma).await?.into_inner().into_buf())
}
async fn re<Mp: M>(d: &mut Device, x: &HbmTensor<f8e4m3, Chip, Mp>) -> Result<Vec<u8>> {
    Ok(x.to_host::<Mp>(&mut d.pdma).await?.into_inner().into_buf())
}
fn exact(label: &str, a: &[u8], b: &[u8]) -> usize {
    assert_eq!(a.len(), b.len(), "{label} length");
    let count = a.iter().zip(b).filter(|(x, y)| x != y).count();
    if count != 0 {
        let first: Vec<_> = a
            .iter()
            .zip(b)
            .enumerate()
            .filter(|(_, (x, y))| x != y)
            .take(8)
            .map(|(i, (a, b))| (i, *a, *b))
            .collect();
        panic!("{label}: byte_diff={count} first={first:?}");
    }
    a.len()
}
fn poison_bf16(n: usize, seed: usize) -> Vec<u8> {
    (0..n)
        .flat_map(|i| {
            let x = i.wrapping_mul(0x9e3779b9).wrapping_add(seed);
            ((0x3b00 | (x & 0x3ff) | ((x & 0x400) << 5)) as u16).to_le_bytes()
        })
        .collect()
}
fn row_data(n: usize, seed: usize, live: usize, poison: bool) -> Vec<u8> {
    (0..16 * n)
        .flat_map(|i| {
            let row = i / n;
            let b = if row >= live {
                if poison { 0x7fc1 + (i % 31) as u16 } else { 0 }
            } else {
                let mut x = (i as u64).wrapping_add(seed as u64 * 991);
                x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
                x ^= x >> 31;
                let v = if row == 0 && seed % 4 == 0 {
                    0.
                } else if row == 1 {
                    ((x >> 40) as f32 / 8388608. - 1.) * 1e-8
                } else {
                    ((x >> 40) as f32 / 8388608. - 1.) * 2f32.powi((row % 5) as i32 - 2)
                };
                bf16::from_f32(v).to_bits()
            };
            b.to_le_bytes()
        })
        .collect()
}
fn permute(input: &[u8], stride: usize, order: &[usize]) -> Vec<u8> {
    let mut result = input.to_vec();
    for (r, &s) in order.iter().enumerate() {
        result[r * stride..(r + 1) * stride].copy_from_slice(&input[s * stride..(s + 1) * stride]);
    }
    result
}
fn scatter(
    expected: &mut [u8],
    input: &[u8],
    slots: &[usize],
    pos: &[usize],
    live: usize,
    bytes: usize,
) {
    for r in 0..live {
        let dst = (slots[r] * 1024 + pos[r] % 1024) * KV * bytes;
        expected[dst..dst + KV * bytes]
            .copy_from_slice(&input[r * KV * bytes..(r + 1) * KV * bytes]);
    }
}
struct OWeights {
    w: HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    s: HbmTensor<bf16, Chip, m![H]>,
    wb: Vec<u8>,
    sb: Vec<u8>,
}
impl OWeights {
    async fn load(d: &mut Device, layer: usize) -> Result<Self> {
        let file = std::fs::File::open(load::model_dir()?.join("model.safetensors"))?;
        let map = unsafe { memmap2::Mmap::map(&file)? };
        let t = safetensors::SafeTensors::deserialize(&map)?;
        let prefix = format!("model.language_model.layers.{layer}.self_attn.o_proj");
        let wt = t.tensor(&format!("{prefix}.weight"))?;
        let st = t.tensor(&format!("{prefix}.weight_scale"))?;
        assert_eq!(wt.shape(), [3840, 4096]);
        assert_eq!(st.shape(), [3840, 1]);
        assert_eq!(st.dtype(), safetensors::Dtype::BF16);
        let wb = wt.data().to_vec();
        let sb = st.data().to_vec();
        let w = HostTensor::<f8e4m3, m![H, Qs]>::from_safetensors(&wt)?
            .to_hbm(&mut d.pdma)
            .await?;
        let s = ub(d, &sb).await?;
        Ok(Self { w, s, wb, sb })
    }
    async fn immutable(&self, d: &mut Device) -> Result<usize> {
        Ok(
            exact("O weights immutable", &re(d, &self.w).await?, &self.wb)
                + exact("O scale immutable", &rb(d, &self.s).await?, &self.sb),
        )
    }
}
enum Offsets {
    N1(HbmTensor<i32, Chip, m![B = 1]>),
    N2(HbmTensor<i32, Chip, m![B = 2]>),
    N3(HbmTensor<i32, Chip, m![B = 3]>),
    N4(HbmTensor<i32, Chip, m![B = 4]>),
    N5(HbmTensor<i32, Chip, m![B = 5]>),
    N6(HbmTensor<i32, Chip, m![B = 6]>),
    N7(HbmTensor<i32, Chip, m![B = 7]>),
    N8(HbmTensor<i32, Chip, m![B = 8]>),
    N9(HbmTensor<i32, Chip, m![B = 9]>),
    N10(HbmTensor<i32, Chip, m![B = 10]>),
    N11(HbmTensor<i32, Chip, m![B = 11]>),
    N12(HbmTensor<i32, Chip, m![B = 12]>),
    N13(HbmTensor<i32, Chip, m![B = 13]>),
    N14(HbmTensor<i32, Chip, m![B = 14]>),
    N15(HbmTensor<i32, Chip, m![B = 15]>),
    N16(HbmTensor<i32, Chip, m![B = 16]>),
}
struct Inputs {
    q: Query,
    k: Kv,
    v: Kv,
    off: Offsets,
    ring: HbmTensor<i32, Chip, m![B]>,
    mask: HbmTensor<f32, Chip, m![B, Ts]>,
    qb: Vec<u8>,
    kb: Vec<u8>,
    vb: Vec<u8>,
    offset_bytes: Vec<u8>,
    ring_bytes: Vec<u8>,
    mask_bytes: Vec<u8>,
}
impl Inputs {
    async fn new(
        d: &mut Device,
        qb: Vec<u8>,
        kb: Vec<u8>,
        vb: Vec<u8>,
        slots: &[usize],
        pos: &[usize],
        live: usize,
        e4: bool,
        all_masked: bool,
    ) -> Result<Self> {
        let bytes = if e4 { 1 } else { 2 };
        let offsets: Vec<_> = (0..live)
            .map(|r| ((slots[r] * 1024 + pos[r] % 1024) * KV * bytes) as i32)
            .collect();
        let offset_bytes = offsets.iter().flat_map(|x| x.to_le_bytes()).collect();
        let off = match live {
            1 => Offsets::N1(
                HostTensor::<i32, m![B = 1]>::from_vec(offsets)
                    .to_hbm(&mut d.pdma)
                    .await?,
            ),
            2 => Offsets::N2(
                HostTensor::<i32, m![B = 2]>::from_vec(offsets)
                    .to_hbm(&mut d.pdma)
                    .await?,
            ),
            3 => Offsets::N3(
                HostTensor::<i32, m![B = 3]>::from_vec(offsets)
                    .to_hbm(&mut d.pdma)
                    .await?,
            ),
            4 => Offsets::N4(
                HostTensor::<i32, m![B = 4]>::from_vec(offsets)
                    .to_hbm(&mut d.pdma)
                    .await?,
            ),
            5 => Offsets::N5(
                HostTensor::<i32, m![B = 5]>::from_vec(offsets)
                    .to_hbm(&mut d.pdma)
                    .await?,
            ),
            6 => Offsets::N6(
                HostTensor::<i32, m![B = 6]>::from_vec(offsets)
                    .to_hbm(&mut d.pdma)
                    .await?,
            ),
            7 => Offsets::N7(
                HostTensor::<i32, m![B = 7]>::from_vec(offsets)
                    .to_hbm(&mut d.pdma)
                    .await?,
            ),
            8 => Offsets::N8(
                HostTensor::<i32, m![B = 8]>::from_vec(offsets)
                    .to_hbm(&mut d.pdma)
                    .await?,
            ),
            9 => Offsets::N9(
                HostTensor::<i32, m![B = 9]>::from_vec(offsets)
                    .to_hbm(&mut d.pdma)
                    .await?,
            ),
            10 => Offsets::N10(
                HostTensor::<i32, m![B = 10]>::from_vec(offsets)
                    .to_hbm(&mut d.pdma)
                    .await?,
            ),
            11 => Offsets::N11(
                HostTensor::<i32, m![B = 11]>::from_vec(offsets)
                    .to_hbm(&mut d.pdma)
                    .await?,
            ),
            12 => Offsets::N12(
                HostTensor::<i32, m![B = 12]>::from_vec(offsets)
                    .to_hbm(&mut d.pdma)
                    .await?,
            ),
            13 => Offsets::N13(
                HostTensor::<i32, m![B = 13]>::from_vec(offsets)
                    .to_hbm(&mut d.pdma)
                    .await?,
            ),
            14 => Offsets::N14(
                HostTensor::<i32, m![B = 14]>::from_vec(offsets)
                    .to_hbm(&mut d.pdma)
                    .await?,
            ),
            15 => Offsets::N15(
                HostTensor::<i32, m![B = 15]>::from_vec(offsets)
                    .to_hbm(&mut d.pdma)
                    .await?,
            ),
            16 => Offsets::N16(
                HostTensor::<i32, m![B = 16]>::from_vec(offsets)
                    .to_hbm(&mut d.pdma)
                    .await?,
            ),
            _ => unreachable!(),
        };
        let ring_host: Vec<_> = slots
            .iter()
            .map(|s| (s * 1024 * KV * bytes) as i32)
            .collect();
        let ring_bytes = ring_host.iter().flat_map(|x| x.to_le_bytes()).collect();
        let ring = HostTensor::<i32, m![B]>::from_vec(ring_host)
            .to_hbm(&mut d.pdma)
            .await?;
        let mask_host: Vec<f32> = (0..16)
            .flat_map(|r| {
                (0..1024).map(move |t| {
                    if !all_masked && r < live && (pos[r] >= 1023 || t <= pos[r]) {
                        1.0
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        let mask_bytes = mask_host.iter().flat_map(|x| x.to_le_bytes()).collect();
        let mask = HostTensor::<f32, m![B, Ts]>::from_vec(mask_host)
            .to_hbm(&mut d.pdma)
            .await?;
        Ok(Self {
            q: ub(d, &qb).await?,
            k: ub(d, &kb).await?,
            v: ub(d, &vb).await?,
            off,
            ring,
            mask,
            qb,
            kb,
            vb,
            offset_bytes,
            ring_bytes,
            mask_bytes,
        })
    }
    async fn immutable(&self, d: &mut Device) -> Result<usize> {
        let off = match &self.off {
            Offsets::N1(o) => o
                .to_host::<m![B = 1]>(&mut d.pdma)
                .await?
                .into_inner()
                .into_buf(),
            Offsets::N2(o) => o
                .to_host::<m![B = 2]>(&mut d.pdma)
                .await?
                .into_inner()
                .into_buf(),
            Offsets::N3(o) => o
                .to_host::<m![B = 3]>(&mut d.pdma)
                .await?
                .into_inner()
                .into_buf(),
            Offsets::N4(o) => o
                .to_host::<m![B = 4]>(&mut d.pdma)
                .await?
                .into_inner()
                .into_buf(),
            Offsets::N5(o) => o
                .to_host::<m![B = 5]>(&mut d.pdma)
                .await?
                .into_inner()
                .into_buf(),
            Offsets::N6(o) => o
                .to_host::<m![B = 6]>(&mut d.pdma)
                .await?
                .into_inner()
                .into_buf(),
            Offsets::N7(o) => o
                .to_host::<m![B = 7]>(&mut d.pdma)
                .await?
                .into_inner()
                .into_buf(),
            Offsets::N8(o) => o
                .to_host::<m![B = 8]>(&mut d.pdma)
                .await?
                .into_inner()
                .into_buf(),
            Offsets::N9(o) => o
                .to_host::<m![B = 9]>(&mut d.pdma)
                .await?
                .into_inner()
                .into_buf(),
            Offsets::N10(o) => o
                .to_host::<m![B = 10]>(&mut d.pdma)
                .await?
                .into_inner()
                .into_buf(),
            Offsets::N11(o) => o
                .to_host::<m![B = 11]>(&mut d.pdma)
                .await?
                .into_inner()
                .into_buf(),
            Offsets::N12(o) => o
                .to_host::<m![B = 12]>(&mut d.pdma)
                .await?
                .into_inner()
                .into_buf(),
            Offsets::N13(o) => o
                .to_host::<m![B = 13]>(&mut d.pdma)
                .await?
                .into_inner()
                .into_buf(),
            Offsets::N14(o) => o
                .to_host::<m![B = 14]>(&mut d.pdma)
                .await?
                .into_inner()
                .into_buf(),
            Offsets::N15(o) => o
                .to_host::<m![B = 15]>(&mut d.pdma)
                .await?
                .into_inner()
                .into_buf(),
            Offsets::N16(o) => o
                .to_host::<m![B = 16]>(&mut d.pdma)
                .await?
                .into_inner()
                .into_buf(),
        };
        let ring = self
            .ring
            .to_host::<m![B]>(&mut d.pdma)
            .await?
            .into_inner()
            .into_buf();
        let mask = self
            .mask
            .to_host::<m![B, Ts]>(&mut d.pdma)
            .await?
            .into_inner()
            .into_buf();
        Ok(exact("offsets immutable", &off, &self.offset_bytes)
            + exact("ring immutable", &ring, &self.ring_bytes)
            + exact("mask immutable", &mask, &self.mask_bytes)
            + exact("Q immutable", &rb(d, &self.q).await?, &self.qb)
            + exact("K immutable", &rb(d, &self.k).await?, &self.kb)
            + exact("V immutable", &rb(d, &self.v).await?, &self.vb))
    }
}
struct BScratch {
    k: BPool,
    v: BPool,
    a: Query,
    o: HbmTensor<bf16, Chip, m![B, H]>,
}
struct EScratch {
    k: EPool,
    v: EPool,
    a: Query,
    o: HbmTensor<bf16, Chip, m![B, H]>,
}
impl BScratch {
    async fn new(d: &mut Device, k: &[u8], v: &[u8], nan: u16) -> Result<Self> {
        Ok(Self {
            k: ub(d, k).await?,
            v: ub(d, v).await?,
            a: ub(d, &nan_bytes(16 * Q, nan)).await?,
            o: ub(d, &nan_bytes(16 * 3840, nan)).await?,
        })
    }
}
impl EScratch {
    async fn new(d: &mut Device, k: &[u8], v: &[u8], nan: u16) -> Result<Self> {
        Ok(Self {
            k: ue(d, k).await?,
            v: ue(d, v).await?,
            a: ub(d, &nan_bytes(16 * Q, nan)).await?,
            o: ub(d, &nan_bytes(16 * 3840, nan)).await?,
        })
    }
}
fn nan_bytes(n: usize, b: u16) -> Vec<u8> {
    (0..n).flat_map(|_| b.to_le_bytes()).collect()
}
macro_rules! invoke {
    ($fun:path,$d:expr,$i:expr,$s:expr,$w:expr,$off:expr) => {
        launch(
            $fun,
            (
                $d,
                &$i.q,
                &$i.k,
                &$i.v,
                $off,
                &$i.ring,
                &$i.mask,
                &mut $s.k,
                &mut $s.v,
                &$w.w,
                &$w.s,
                &mut $s.a,
                unsafe { $s.o.view_mut().reshape() },
            ),
        )
        .await?
    };
}
async fn bchain(d: &mut Device, i: &Inputs, s: &mut BScratch, w: &OWeights) -> Result<()> {
    match &i.off {
        Offsets::N1(o) => invoke!(original::sliding_1, d, i, s, w, o),
        Offsets::N2(o) => invoke!(original::sliding_2, d, i, s, w, o),
        Offsets::N3(o) => invoke!(original::sliding_3, d, i, s, w, o),
        Offsets::N4(o) => invoke!(original::sliding_4, d, i, s, w, o),
        Offsets::N5(o) => invoke!(original::sliding_5, d, i, s, w, o),
        Offsets::N6(o) => invoke!(original::sliding_6, d, i, s, w, o),
        Offsets::N7(o) => invoke!(original::sliding_7, d, i, s, w, o),
        Offsets::N8(o) => invoke!(original::sliding_8, d, i, s, w, o),
        Offsets::N9(o) => invoke!(original::sliding_9, d, i, s, w, o),
        Offsets::N10(o) => invoke!(original::sliding_10, d, i, s, w, o),
        Offsets::N11(o) => invoke!(original::sliding_11, d, i, s, w, o),
        Offsets::N12(o) => invoke!(original::sliding_12, d, i, s, w, o),
        Offsets::N13(o) => invoke!(original::sliding_13, d, i, s, w, o),
        Offsets::N14(o) => invoke!(original::sliding_14, d, i, s, w, o),
        Offsets::N15(o) => invoke!(original::sliding_15, d, i, s, w, o),
        Offsets::N16(o) => invoke!(original::sliding_16, d, i, s, w, o),
    };
    Ok(())
}
async fn echain(d: &mut Device, i: &Inputs, s: &mut EScratch, w: &OWeights) -> Result<()> {
    match &i.off {
        Offsets::N1(o) => invoke!(fixed::sliding_1, d, i, s, w, o),
        Offsets::N2(o) => invoke!(fixed::sliding_2, d, i, s, w, o),
        Offsets::N3(o) => invoke!(fixed::sliding_3, d, i, s, w, o),
        Offsets::N4(o) => invoke!(fixed::sliding_4, d, i, s, w, o),
        Offsets::N5(o) => invoke!(fixed::sliding_5, d, i, s, w, o),
        Offsets::N6(o) => invoke!(fixed::sliding_6, d, i, s, w, o),
        Offsets::N7(o) => invoke!(fixed::sliding_7, d, i, s, w, o),
        Offsets::N8(o) => invoke!(fixed::sliding_8, d, i, s, w, o),
        Offsets::N9(o) => invoke!(fixed::sliding_9, d, i, s, w, o),
        Offsets::N10(o) => invoke!(fixed::sliding_10, d, i, s, w, o),
        Offsets::N11(o) => invoke!(fixed::sliding_11, d, i, s, w, o),
        Offsets::N12(o) => invoke!(fixed::sliding_12, d, i, s, w, o),
        Offsets::N13(o) => invoke!(fixed::sliding_13, d, i, s, w, o),
        Offsets::N14(o) => invoke!(fixed::sliding_14, d, i, s, w, o),
        Offsets::N15(o) => invoke!(fixed::sliding_15, d, i, s, w, o),
        Offsets::N16(o) => invoke!(fixed::sliding_16, d, i, s, w, o),
    };
    Ok(())
}
fn drift(label: &str, a: &[u8], b: &[u8], live: usize, stride: usize) {
    let (mut err, mut norm, mut max, mut diff, mut nonfinite) = (0f64, 0f64, 0f64, 0usize, 0usize);
    for (a, b) in a[..live * stride * 2]
        .chunks_exact(2)
        .zip(b[..live * stride * 2].chunks_exact(2))
    {
        let ab = u16::from_le_bytes([a[0], a[1]]);
        let bb = u16::from_le_bytes([b[0], b[1]]);
        diff += usize::from(ab != bb);
        let x = f32::from_bits((ab as u32) << 16) as f64;
        let y = f32::from_bits((bb as u32) << 16) as f64;
        if !x.is_finite() || !y.is_finite() {
            nonfinite += 1;
            continue;
        }
        err += (x - y) * (x - y);
        norm += y * y;
        max = max.max((x - y).abs());
    }
    println!(
        "FIXED_E4_DRIFT label={label} bitdiff={diff} nonfinite={nonfinite} relative_l2={:.9} max_abs={max:.9} synthetic_history=true",
        (err / norm.max(1e-300)).sqrt()
    );
    assert_eq!(nonfinite, 0, "active output nonfinite");
}
async fn codec_gate(d: &mut Device, w: &OWeights, lut: &[u8]) -> Result<usize> {
    let mut kb = Vec::with_capacity(16 * KV * 2);
    let mut vb = Vec::with_capacity(16 * KV * 2);
    let mut finite = 0usize;
    let mut saturated = 0usize;
    for i in 0..65536usize {
        let mut b = i as u16;
        if b & 0x7f80 == 0x7f80 {
            b = (b & 0x8000) | 0x7f7f;
        } else {
            finite += 1;
        }
        if f32::from_bits(((b & 0x7fff) as u32) << 16) > 448. {
            saturated += 1;
        }
        if i < 32768 {
            kb.extend_from_slice(&b.to_le_bytes());
        } else {
            vb.extend_from_slice(&b.to_le_bytes());
        }
    }
    let mut ek = vec![0x25; POOL];
    let mut ev = vec![0xa5; POOL];
    let mut s = EScratch::new(d, &ek, &ev, 0x7fc9).await?;
    let i = Inputs::new(
        d,
        vec![0; 16 * Q * 2],
        kb.clone(),
        vb.clone(),
        &SLOTS,
        &POS,
        16,
        true,
        true,
    )
    .await?;
    scatter(&mut ek, &codec::encode_bytes(&kb, lut), &SLOTS, &POS, 16, 1);
    scatter(&mut ev, &codec::encode_bytes(&vb, lut), &SLOTS, &POS, 16, 1);
    // Reuse the unchanged complete candidate graph; only the append bytes are the
    // codec oracle. No accuracy assertion is made about all-masked codec outputs.
    echain(d, &i, &mut s, w).await?;
    let bytes = exact(
        "exhaustive codec K plus untouched pool",
        &re(d, &s.k).await?,
        &ek,
    ) + exact(
        "exhaustive codec V plus untouched pool",
        &re(d, &s.v).await?,
        &ev,
    );
    i.immutable(d).await?;
    println!(
        "FIXED_E4_CODEC_PASS finite_distinct={finite} input_values=65536 overrange_inputs={saturated} expected_overflow=saturate_signed448 rounding=nearest_even signed_zero=true all_pool_bytes={bytes}"
    );
    Ok(bytes)
}
pub async fn run(mut profile: impl FnMut(bool)) -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let first = args
        .first()
        .map(|x| x.parse::<usize>())
        .transpose()?
        .unwrap_or(0);
    let end = args
        .get(1)
        .map(|x| x.parse::<usize>())
        .transpose()?
        .unwrap_or(first + 1);
    assert!(first < end && end <= 16);
    let start = std::time::Instant::now();
    let lut = codec::table();
    let mut d = Device::new(fixed::sliding_16.topology())?;
    let (mut cache_bytes, mut output_bytes, mut immutable_bytes) = (0usize, 0usize, 0usize);
    let mut timings = Vec::new();
    for case in first..end {
        let live = case + 1;
        let layer = [0, 6, 46, 0][case % 4];
        let w = OWeights::load(&mut d, layer).await?;
        if case == first {
            cache_bytes += codec_gate(&mut d, &w, &lut).await?;
        }
        let mut ok = poison_bf16(POOL, 17 + case);
        let mut ov = poison_bf16(POOL, 987 + case);
        let mut ek = codec::encode_bytes(&ok, &lut);
        let mut ev = codec::encode_bytes(&ov, &lut);
        let mut ik = codec::decode_bytes(&ek);
        let mut iv = codec::decode_bytes(&ev);
        let mut cand = EScratch::new(&mut d, &ek, &ev, 0x7fc1).await?;
        let mut intent = BScratch::new(&mut d, &ik, &iv, 0x7fd2).await?;
        let mut orig = BScratch::new(&mut d, &ok, &ov, 0x7fe3).await?;
        let mut qb = row_data(Q, case + 1, live, false);
        let kb0 = row_data(KV, case + 17, live, true);
        let vb0 = row_data(KV, case + 97, live, true);
        let mut slots = SLOTS.to_vec();
        let mut positions = POS.to_vec();
        for step in 0..2 {
            let order: Vec<_> = (0..live)
                .map(|r| if step == 0 { r } else { live - 1 - r })
                .collect();
            let kb = permute(&kb0, KV * 2, &order);
            let vb = permute(&vb0, KV * 2, &order);
            if step == 1 {
                qb = permute(&qb, Q * 2, &order);
                slots = (0..16)
                    .map(|r| if r < live { SLOTS[order[r]] } else { SLOTS[r] })
                    .collect();
                positions = (0..16)
                    .map(|r| if r < live { POS[order[r]] + 1 } else { POS[r] })
                    .collect();
            }
            let kcode = codec::encode_bytes(&kb, &lut);
            let vcode = codec::encode_bytes(&vb, &lut);
            let kr = codec::decode_bytes(&kcode);
            let vr = codec::decode_bytes(&vcode);
            scatter(&mut ek, &kcode, &slots, &positions, live, 1);
            scatter(&mut ev, &vcode, &slots, &positions, live, 1);
            scatter(&mut ik, &kr, &slots, &positions, live, 2);
            scatter(&mut iv, &vr, &slots, &positions, live, 2);
            scatter(&mut ok, &kb, &slots, &positions, live, 2);
            scatter(&mut ov, &vb, &slots, &positions, live, 2);
            let ci = Inputs::new(
                &mut d,
                qb.clone(),
                kr.clone(),
                vr.clone(),
                &slots,
                &positions,
                live,
                true,
                false,
            )
            .await?;
            let ii = Inputs::new(
                &mut d,
                qb.clone(),
                kr,
                vr,
                &slots,
                &positions,
                live,
                false,
                false,
            )
            .await?;
            let oi = Inputs::new(
                &mut d,
                qb.clone(),
                kb,
                vb,
                &slots,
                &positions,
                live,
                false,
                false,
            )
            .await?;
            // Re-poison every output at each feedback step; candidate runs first.
            cand.a = ub(&mut d, &nan_bytes(16 * Q, 0x7fc1)).await?;
            cand.o = ub(&mut d, &nan_bytes(16 * 3840, 0x7fc1)).await?;
            intent.a = ub(&mut d, &nan_bytes(16 * Q, 0x7fd2)).await?;
            intent.o = ub(&mut d, &nan_bytes(16 * 3840, 0x7fd2)).await?;
            orig.a = ub(&mut d, &nan_bytes(16 * Q, 0x7fe3)).await?;
            orig.o = ub(&mut d, &nan_bytes(16 * 3840, 0x7fe3)).await?;
            echain(&mut d, &ci, &mut cand, &w).await?;
            let ca = rb(&mut d, &cand.a).await?;
            let co = rb(&mut d, &cand.o).await?;
            cache_bytes += exact("fixed whole K pool", &re(&mut d, &cand.k).await?, &ek)
                + exact("fixed whole V pool", &re(&mut d, &cand.v).await?, &ev);
            bchain(&mut d, &ii, &mut intent, &w).await?;
            output_bytes += exact(
                "intended attention all rows",
                &ca,
                &rb(&mut d, &intent.a).await?,
            ) + exact("intended O all rows", &co, &rb(&mut d, &intent.o).await?);
            cache_bytes += exact("intended whole K pool", &rb(&mut d, &intent.k).await?, &ik)
                + exact("intended whole V pool", &rb(&mut d, &intent.v).await?, &iv);
            bchain(&mut d, &oi, &mut orig, &w).await?;
            cache_bytes += exact("original whole K pool", &rb(&mut d, &orig.k).await?, &ok)
                + exact("original whole V pool", &rb(&mut d, &orig.v).await?, &ov);
            drift(
                &format!("case{case}_step{step}_attention"),
                &ca,
                &rb(&mut d, &orig.a).await?,
                live,
                Q,
            );
            drift(
                &format!("case{case}_step{step}_O"),
                &co,
                &rb(&mut d, &orig.o).await?,
                live,
                3840,
            );
            assert!(
                co[live * 3840 * 2..]
                    .chunks_exact(2)
                    .all(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7fff == 0),
                "inactive rows not zero"
            );
            immutable_bytes += ci.immutable(&mut d).await?
                + ii.immutable(&mut d).await?
                + oi.immutable(&mut d).await?;
            println!(
                "FIXED_E4_EXACT case={case} layer={layer} live={live} step={step} cumulative_cache_bytes={cache_bytes} cumulative_output_bytes={output_bytes} synthetic_history=true inactive_kv_nan=true future_unwritten_finite_poison=true rounded_append_idempotent=true physical_slots=32"
            );
            let (mut old, mut new) = (Vec::new(), Vec::new());
            for (round, candidate) in [false, true, true, false].into_iter().enumerate() {
                let begin = std::time::Instant::now();
                for _ in 0..6 {
                    if candidate {
                        echain(&mut d, &ci, &mut cand, &w).await?;
                    } else {
                        bchain(&mut d, &ii, &mut intent, &w).await?;
                    }
                }
                let us = begin.elapsed().as_secs_f64() * 1e6 / 6.;
                if candidate {
                    new.push(us);
                } else {
                    old.push(us);
                }
                println!(
                    "FIXED_E4_TIME case={case} step={step} live={live} round={round} candidate={candidate} complete_us={us:.6} samples=6 includes=append_attention_O excludes=upload_allocation_readback_metadata_cpu"
                );
            }
            output_bytes += exact("timed intended attention", &rb(&mut d, &cand.a).await?, &ca)
                + exact("timed intended O", &rb(&mut d, &cand.o).await?, &co);
            cache_bytes += exact("timed fixed K pool", &re(&mut d, &cand.k).await?, &ek)
                + exact("timed fixed V pool", &re(&mut d, &cand.v).await?, &ev)
                + exact("timed intended K pool", &rb(&mut d, &intent.k).await?, &ik)
                + exact("timed intended V pool", &rb(&mut d, &intent.v).await?, &iv);
            let a = old.iter().sum::<f64>() / 2.;
            let b = new.iter().sum::<f64>() / 2.;
            timings.push((case, step, a, b));
            if step == 1 && std::env::var_os("FURIOSA_OPT_PROFILE").is_some() {
                for candidate in [false, true] {
                    println!("PROFILE_VARIANT case={case} live={live} candidate={candidate}");
                    profile(true);
                    if candidate {
                        echain(&mut d, &ci, &mut cand, &w).await?;
                    } else {
                        bchain(&mut d, &ii, &mut intent, &w).await?;
                    }
                    profile(false);
                    println!("PROFILE_END");
                }
            }
            immutable_bytes += ci.immutable(&mut d).await?
                + ii.immutable(&mut d).await?
                + oi.immutable(&mut d).await?;
            // Matched synthetic feedback: every arm receives the exact same
            // candidate attention as its next query, followed by row reversal.
            qb = ca;
        }
        immutable_bytes += w.immutable(&mut d).await?;
    }
    let old = timings.iter().map(|x| x.2).sum::<f64>() / timings.len() as f64;
    let new = timings.iter().map(|x| x.3).sum::<f64>() / timings.len() as f64;
    let gain = 1. - new / old;
    println!(
        "FIXED_E4_ALL_CORRECT cases={first}..{end} cache_bytes={cache_bytes} output_bytes={output_bytes} immutable_bytes={immutable_bytes} intended_bf16_us={old:.6} candidate_us={new:.6} gain={gain:.9} elapsed_s={:.3} synthetic_history=true no_model_quality_claim=true",
        start.elapsed().as_secs_f64()
    );
    // This is an Rc32 storage-parity gate for every live width; no per-width speed threshold.
    // Timings describe only this complete device graph, not serving latency.
    println!(
        "PASS fixedE4 complete append+attention+suppliedO; exact intended codec/pools/output; original BF16 drift separately reported; timing diagnostic only; Rc32 all-width storage parity"
    );
    Ok(())
}
