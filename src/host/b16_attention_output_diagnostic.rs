//! Complete three-call versus one-graph gate with actual supplied O weights.
use super::load::{self, Layer, SlidingLayer};
use crate::{
    Chip, axes::*, ops_b16_attention_output as fused, ops_commit_r64 as commit,
    ops_output_b16_hsplit as output, ops_pool_views::Rc64,
    ops_request_sliding_b16_rc64 as attention,
};
use furiosa_opt_std::prelude::*;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
type Pool = HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>;
type NewKv = HbmTensor<bf16, Chip, m![B, Ns, Ds]>;
type Query = HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]>;
const ROW_BYTES: usize = 8 * 256 * 2;
const SLOTS: [usize; 16] = [63, 0, 31, 1, 62, 32, 2, 61, 33, 3, 60, 34, 4, 59, 35, 7];
const POS: [usize; 16] = [
    0, 1, 31, 255, 511, 512, 767, 768, 1023, 1024, 1025, 1535, 2047, 2048, 8191, 32767,
];
async fn upload<Mp: M>(d: &mut Device, bytes: &[u8]) -> Result<HbmTensor<bf16, Chip, Mp>> {
    assert_eq!(bytes.len(), Mp::SIZE * 2);
    Ok(HostTensor::<bf16, Mp>::from_buf(bytes.to_vec())
        .to_hbm(&mut d.pdma)
        .await?)
}
async fn read<Mp: M>(d: &mut Device, x: &HbmTensor<bf16, Chip, Mp>) -> Result<Vec<u8>> {
    Ok(x.to_host::<Mp>(&mut d.pdma).await?.into_inner().into_buf())
}
fn exact(label: &str, a: &[u8], b: &[u8]) -> usize {
    if a != b {
        let first = a.iter().zip(b).position(|(a, b)| a != b);
        panic!("{label}: first={first:?} sizes={}/{}", a.len(), b.len());
    }
    a.len()
}
fn finite(n: usize, seed: usize) -> Vec<u8> {
    (0..n)
        .flat_map(|i| {
            let x = i.wrapping_mul(0x9e3779b9).wrapping_add(seed);
            ((0x3b00 | (x & 0x3ff) | ((x & 0x400) << 5)) as u16).to_le_bytes()
        })
        .collect()
}
fn random_rows(n: usize, seed: usize, live: usize, poison_tail: bool) -> Vec<u8> {
    (0..16 * n)
        .flat_map(|i| {
            let row = i / n;
            let col = i % n;
            let raw = if poison_tail && row >= live {
                0x7fc1 + (col % 31) as u16
            } else {
                let mut x = (i as u64).wrapping_add(seed as u64 * 991);
                x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
                x ^= x >> 31;
                let value = if row == 0 {
                    0.0
                } else if row == 1 {
                    ((x >> 40) as f32 / 8388608. - 1.) * 1e-8
                } else {
                    ((x >> 40) as f32 / 8388608. - 1.) * 2f32.powi((row % 5) as i32 - 2)
                };
                bf16::from_f32(value).to_bits()
            };
            raw.to_le_bytes()
        })
        .collect()
}
fn update(expected: &mut [u8], input: &[u8], slots: &[usize], positions: &[usize], live: usize) {
    for row in 0..live {
        for head in 0..8 {
            for channel in 0..256 {
                let src = ((row * 8 + head) * 256 + channel) * 2;
                let dst =
                    (((slots[row] * 1024 + positions[row] % 1024) * 8 + head) * 256 + channel) * 2;
                expected[dst..dst + 2].copy_from_slice(&input[src..src + 2]);
            }
        }
    }
}
struct Scratch {
    k: Pool,
    v: Pool,
    attention: Query,
    output: HbmTensor<bf16, Chip, m![B, H]>,
}
impl Scratch {
    async fn new(d: &mut Device, k: &[u8], v: &[u8], poison: u8) -> Result<Self> {
        Ok(Self {
            k: upload(d, k).await?,
            v: upload(d, v).await?,
            attention: upload(d, &vec![poison; 16 * 8 * 2 * 256 * 2]).await?,
            output: upload(d, &vec![poison; 16 * 3840 * 2]).await?,
        })
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

struct Input {
    q: Query,
    k: NewKv,
    v: NewKv,
    offset: Offsets,
    ring: HbmTensor<i32, Chip, m![B]>,
    mask: HbmTensor<f32, Chip, m![B, Ts]>,
    qb: Vec<u8>,
    kb: Vec<u8>,
    vb: Vec<u8>,
    slots: Vec<usize>,
    positions: Vec<usize>,
}
impl Input {
    async fn new(d: &mut Device, case: usize, live: usize) -> Result<Self> {
        let slots: Vec<_> = (0..16).map(|r| SLOTS[(r * 5 + case) % 16]).collect();
        let positions: Vec<_> = (0..16)
            .map(|r| if case == 15 { 0 } else { POS[(r + case) % 16] })
            .collect();
        let qb = random_rows(8 * 2 * 256, case + 1, live, false);
        let kb = random_rows(8 * 256, case + 17, live, true);
        let vb = random_rows(8 * 256, case + 97, live, true);
        let offsets: Vec<_> = (0..live)
            .map(|r| ((slots[r] * 1024 + positions[r] % 1024) * ROW_BYTES) as i32)
            .collect();
        let offset = match live {
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
        let ring =
            HostTensor::<i32, m![B]>::from_vec(slots.iter().map(|s| (s * 1024 * ROW_BYTES) as i32))
                .to_hbm(&mut d.pdma)
                .await?;
        let mask = HostTensor::<f32, m![B, Ts]>::from_vec((0..16).flat_map(|r| {
            let p = positions[r];
            (0..1024).map(move |t| {
                if r < live && (p >= 1023 || t <= p) {
                    1.0
                } else {
                    0.0
                }
            })
        }))
        .to_hbm(&mut d.pdma)
        .await?;
        Ok(Self {
            q: upload(d, &qb).await?,
            k: upload(d, &kb).await?,
            v: upload(d, &vb).await?,
            offset,
            ring,
            mask,
            qb,
            kb,
            vb,
            slots,
            positions,
        })
    }
    async fn immutable(&self, d: &mut Device) -> Result<()> {
        exact("Q immutable", &read(d, &self.q).await?, &self.qb);
        exact("K immutable", &read(d, &self.k).await?, &self.kb);
        exact("V immutable", &read(d, &self.v).await?, &self.vb);
        Ok(())
    }
}
async fn chain(
    d: &mut Device,
    i: &Input,
    s: &mut Scratch,
    l: &SlidingLayer,
    candidate: bool,
) -> Result<()> {
    if candidate {
        match &i.offset {
            Offsets::N1(o) => {
                launch(
                    fused::sliding_1,
                    (
                        d,
                        &i.q,
                        &i.k,
                        &i.v,
                        o,
                        &i.ring,
                        &i.mask,
                        &mut s.k,
                        &mut s.v,
                        &l.o_weight,
                        &l.o_weight_scale,
                        &mut s.attention,
                        unsafe { s.output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            Offsets::N2(o) => {
                launch(
                    fused::sliding_2,
                    (
                        d,
                        &i.q,
                        &i.k,
                        &i.v,
                        o,
                        &i.ring,
                        &i.mask,
                        &mut s.k,
                        &mut s.v,
                        &l.o_weight,
                        &l.o_weight_scale,
                        &mut s.attention,
                        unsafe { s.output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            Offsets::N3(o) => {
                launch(
                    fused::sliding_3,
                    (
                        d,
                        &i.q,
                        &i.k,
                        &i.v,
                        o,
                        &i.ring,
                        &i.mask,
                        &mut s.k,
                        &mut s.v,
                        &l.o_weight,
                        &l.o_weight_scale,
                        &mut s.attention,
                        unsafe { s.output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            Offsets::N4(o) => {
                launch(
                    fused::sliding_4,
                    (
                        d,
                        &i.q,
                        &i.k,
                        &i.v,
                        o,
                        &i.ring,
                        &i.mask,
                        &mut s.k,
                        &mut s.v,
                        &l.o_weight,
                        &l.o_weight_scale,
                        &mut s.attention,
                        unsafe { s.output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            Offsets::N5(o) => {
                launch(
                    fused::sliding_5,
                    (
                        d,
                        &i.q,
                        &i.k,
                        &i.v,
                        o,
                        &i.ring,
                        &i.mask,
                        &mut s.k,
                        &mut s.v,
                        &l.o_weight,
                        &l.o_weight_scale,
                        &mut s.attention,
                        unsafe { s.output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            Offsets::N6(o) => {
                launch(
                    fused::sliding_6,
                    (
                        d,
                        &i.q,
                        &i.k,
                        &i.v,
                        o,
                        &i.ring,
                        &i.mask,
                        &mut s.k,
                        &mut s.v,
                        &l.o_weight,
                        &l.o_weight_scale,
                        &mut s.attention,
                        unsafe { s.output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            Offsets::N7(o) => {
                launch(
                    fused::sliding_7,
                    (
                        d,
                        &i.q,
                        &i.k,
                        &i.v,
                        o,
                        &i.ring,
                        &i.mask,
                        &mut s.k,
                        &mut s.v,
                        &l.o_weight,
                        &l.o_weight_scale,
                        &mut s.attention,
                        unsafe { s.output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            Offsets::N8(o) => {
                launch(
                    fused::sliding_8,
                    (
                        d,
                        &i.q,
                        &i.k,
                        &i.v,
                        o,
                        &i.ring,
                        &i.mask,
                        &mut s.k,
                        &mut s.v,
                        &l.o_weight,
                        &l.o_weight_scale,
                        &mut s.attention,
                        unsafe { s.output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            Offsets::N9(o) => {
                launch(
                    fused::sliding_9,
                    (
                        d,
                        &i.q,
                        &i.k,
                        &i.v,
                        o,
                        &i.ring,
                        &i.mask,
                        &mut s.k,
                        &mut s.v,
                        &l.o_weight,
                        &l.o_weight_scale,
                        &mut s.attention,
                        unsafe { s.output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            Offsets::N10(o) => {
                launch(
                    fused::sliding_10,
                    (
                        d,
                        &i.q,
                        &i.k,
                        &i.v,
                        o,
                        &i.ring,
                        &i.mask,
                        &mut s.k,
                        &mut s.v,
                        &l.o_weight,
                        &l.o_weight_scale,
                        &mut s.attention,
                        unsafe { s.output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            Offsets::N11(o) => {
                launch(
                    fused::sliding_11,
                    (
                        d,
                        &i.q,
                        &i.k,
                        &i.v,
                        o,
                        &i.ring,
                        &i.mask,
                        &mut s.k,
                        &mut s.v,
                        &l.o_weight,
                        &l.o_weight_scale,
                        &mut s.attention,
                        unsafe { s.output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            Offsets::N12(o) => {
                launch(
                    fused::sliding_12,
                    (
                        d,
                        &i.q,
                        &i.k,
                        &i.v,
                        o,
                        &i.ring,
                        &i.mask,
                        &mut s.k,
                        &mut s.v,
                        &l.o_weight,
                        &l.o_weight_scale,
                        &mut s.attention,
                        unsafe { s.output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            Offsets::N13(o) => {
                launch(
                    fused::sliding_13,
                    (
                        d,
                        &i.q,
                        &i.k,
                        &i.v,
                        o,
                        &i.ring,
                        &i.mask,
                        &mut s.k,
                        &mut s.v,
                        &l.o_weight,
                        &l.o_weight_scale,
                        &mut s.attention,
                        unsafe { s.output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            Offsets::N14(o) => {
                launch(
                    fused::sliding_14,
                    (
                        d,
                        &i.q,
                        &i.k,
                        &i.v,
                        o,
                        &i.ring,
                        &i.mask,
                        &mut s.k,
                        &mut s.v,
                        &l.o_weight,
                        &l.o_weight_scale,
                        &mut s.attention,
                        unsafe { s.output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            Offsets::N15(o) => {
                launch(
                    fused::sliding_15,
                    (
                        d,
                        &i.q,
                        &i.k,
                        &i.v,
                        o,
                        &i.ring,
                        &i.mask,
                        &mut s.k,
                        &mut s.v,
                        &l.o_weight,
                        &l.o_weight_scale,
                        &mut s.attention,
                        unsafe { s.output.view_mut().reshape() },
                    ),
                )
                .await?
            }
            Offsets::N16(o) => {
                launch(
                    fused::sliding_16,
                    (
                        d,
                        &i.q,
                        &i.k,
                        &i.v,
                        o,
                        &i.ring,
                        &i.mask,
                        &mut s.k,
                        &mut s.v,
                        &l.o_weight,
                        &l.o_weight_scale,
                        &mut s.attention,
                        unsafe { s.output.view_mut().reshape() },
                    ),
                )
                .await?
            }
        }
    } else {
        match &i.offset {
            Offsets::N1(o) => {
                launch(
                    commit::commit_sliding_pool_1,
                    (d, &i.k, &i.v, o, &mut s.k, &mut s.v),
                )
                .await?
            }
            Offsets::N2(o) => {
                launch(
                    commit::commit_sliding_pool_2,
                    (d, &i.k, &i.v, o, &mut s.k, &mut s.v),
                )
                .await?
            }
            Offsets::N3(o) => {
                launch(
                    commit::commit_sliding_pool_3,
                    (d, &i.k, &i.v, o, &mut s.k, &mut s.v),
                )
                .await?
            }
            Offsets::N4(o) => {
                launch(
                    commit::commit_sliding_pool_4,
                    (d, &i.k, &i.v, o, &mut s.k, &mut s.v),
                )
                .await?
            }
            Offsets::N5(o) => {
                launch(
                    commit::commit_sliding_pool_5,
                    (d, &i.k, &i.v, o, &mut s.k, &mut s.v),
                )
                .await?
            }
            Offsets::N6(o) => {
                launch(
                    commit::commit_sliding_pool_6,
                    (d, &i.k, &i.v, o, &mut s.k, &mut s.v),
                )
                .await?
            }
            Offsets::N7(o) => {
                launch(
                    commit::commit_sliding_pool_7,
                    (d, &i.k, &i.v, o, &mut s.k, &mut s.v),
                )
                .await?
            }
            Offsets::N8(o) => {
                launch(
                    commit::commit_sliding_pool_8,
                    (d, &i.k, &i.v, o, &mut s.k, &mut s.v),
                )
                .await?
            }
            Offsets::N9(o) => {
                launch(
                    commit::commit_sliding_pool_9,
                    (d, &i.k, &i.v, o, &mut s.k, &mut s.v),
                )
                .await?
            }
            Offsets::N10(o) => {
                launch(
                    commit::commit_sliding_pool_10,
                    (d, &i.k, &i.v, o, &mut s.k, &mut s.v),
                )
                .await?
            }
            Offsets::N11(o) => {
                launch(
                    commit::commit_sliding_pool_11,
                    (d, &i.k, &i.v, o, &mut s.k, &mut s.v),
                )
                .await?
            }
            Offsets::N12(o) => {
                launch(
                    commit::commit_sliding_pool_12,
                    (d, &i.k, &i.v, o, &mut s.k, &mut s.v),
                )
                .await?
            }
            Offsets::N13(o) => {
                launch(
                    commit::commit_sliding_pool_13,
                    (d, &i.k, &i.v, o, &mut s.k, &mut s.v),
                )
                .await?
            }
            Offsets::N14(o) => {
                launch(
                    commit::commit_sliding_pool_14,
                    (d, &i.k, &i.v, o, &mut s.k, &mut s.v),
                )
                .await?
            }
            Offsets::N15(o) => {
                launch(
                    commit::commit_sliding_pool_15,
                    (d, &i.k, &i.v, o, &mut s.k, &mut s.v),
                )
                .await?
            }
            Offsets::N16(o) => {
                launch(
                    commit::commit_sliding_pool_16,
                    (d, &i.k, &i.v, o, &mut s.k, &mut s.v),
                )
                .await?
            }
        }
        launch(
            attention::sliding_attention_b16,
            (d, &i.q, &s.k, &s.v, &i.ring, &i.mask, &mut s.attention),
        )
        .await?;
        launch(
            output::project_sliding_output_batch,
            (d, &s.attention, &l.o_weight, &l.o_weight_scale, unsafe {
                s.output.view_mut().reshape()
            }),
        )
        .await?;
    }
    Ok(())
}
pub async fn run(mut profile: impl FnMut(bool)) -> Result<()> {
    let all_start = std::time::Instant::now();
    let mut d = Device::new(fused::sliding_16.topology())?;
    let weights = load::load_weights(&mut d).await?;
    let mut ek = finite(64 * 1024 * 8 * 256, 17);
    let mut ev = finite(64 * 1024 * 8 * 256, 987);
    let mut control = Scratch::new(&mut d, &ek, &ev, 0xff).await?;
    let mut candidate = Scratch::new(&mut d, &ek, &ev, 0x7f).await?;
    let mut cache_bytes = 0usize;
    let mut output_bytes = 0usize;
    for case in 0usize..16 {
        let live = case + 1;
        let layer = [0usize, 6, 46][case % 3];
        let Layer::Sliding(l) = &weights.layers[layer] else {
            unreachable!()
        };
        let input = Input::new(&mut d, case, live).await?;
        update(&mut ek, &input.kb, &input.slots, &input.positions, live);
        update(&mut ev, &input.vb, &input.slots, &input.positions, live);
        candidate.attention = upload(&mut d, &vec![0x7f; 16 * 8 * 2 * 256 * 2]).await?;
        candidate.output = upload(&mut d, &vec![0x7f; 16 * 3840 * 2]).await?;
        control.attention = upload(&mut d, &vec![0xff; 16 * 8 * 2 * 256 * 2]).await?;
        control.output = upload(&mut d, &vec![0xff; 16 * 3840 * 2]).await?;
        // Independent candidate first; all HBM stages start separately poisoned.
        chain(&mut d, &input, &mut candidate, l, true).await?;
        let ca = read(&mut d, &candidate.attention).await?;
        let co = read(&mut d, &candidate.output).await?;
        cache_bytes += exact(
            "candidate K all bytes CPU",
            &read(&mut d, &candidate.k).await?,
            &ek,
        );
        cache_bytes += exact(
            "candidate V all bytes CPU",
            &read(&mut d, &candidate.v).await?,
            &ev,
        );
        input.immutable(&mut d).await?;
        chain(&mut d, &input, &mut control, l, false).await?;
        output_bytes += exact(
            "attention all rows",
            &ca,
            &read(&mut d, &control.attention).await?,
        );
        output_bytes += exact(
            "output all rows",
            &co,
            &read(&mut d, &control.output).await?,
        );
        cache_bytes += exact(
            "control K all bytes CPU",
            &read(&mut d, &control.k).await?,
            &ek,
        );
        cache_bytes += exact(
            "control V all bytes CPU",
            &read(&mut d, &control.v).await?,
            &ev,
        );
        assert!(
            co[..live * 3840 * 2]
                .chunks_exact(2)
                .all(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7f80 != 0x7f80),
            "active outputs nonfinite"
        );
        if live < 16 {
            assert!(
                co[live * 3840 * 2..]
                    .chunks_exact(2)
                    .all(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7fff == 0),
                "inactive output not masked zero"
            );
        }
        println!(
            "B16_ATTN_O_EXACT case={case} layer={layer} live={live} cumulative_cache_bytes={cache_bytes} cumulative_output_bytes={output_bytes} padding_nan_inputs=true reset={}",
            case == 15
        );
        for (round, new) in [false, true, true, false].into_iter().enumerate() {
            let s = if new { &mut candidate } else { &mut control };
            let now = std::time::Instant::now();
            for _ in 0..12 {
                chain(&mut d, &input, s, l, new).await?;
            }
            println!(
                "B16_ATTN_O_TIME case={case} live={live} round={round} candidate={new} complete_us={:.6} samples=12 excludes_metadata_readbacks=true",
                now.elapsed().as_secs_f64() * 1e6 / 12.
            );
        }
        exact(
            "timed attention",
            &read(&mut d, &candidate.attention).await?,
            &read(&mut d, &control.attention).await?,
        );
        exact(
            "timed output",
            &read(&mut d, &candidate.output).await?,
            &read(&mut d, &control.output).await?,
        );
        input.immutable(&mut d).await?;
        if matches!(live, 14 | 16) && std::env::var_os("FURIOSA_OPT_PROFILE").is_some() {
            for new in [false, true] {
                println!("PROFILE_VARIANT live={live} candidate={new}");
                profile(true);
                chain(
                    &mut d,
                    &input,
                    if new { &mut candidate } else { &mut control },
                    l,
                    new,
                )
                .await?;
                profile(false);
                println!("PROFILE_END");
            }
        }
    }
    println!(
        "PASS original B16 commit+attention+HsplitO3to1; supplied layers0/6/46; live1..16; full cache CPU bytes={cache_bytes}; all intermediate/output bytes={output_bytes}; paddingNaN/slots/wrap/reset/immutability; elapsed_s={:.3}",
        all_start.elapsed().as_secs_f64()
    );
    Ok(())
}
