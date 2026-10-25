//! Rc32 storage parity: unchanged BF16 readers versus fixed-E4 readers.
//! Histories are synthetic finite E4, independently decoded on CPU. No model-quality claim.
#[path = "support/npu_profile.rs"]
mod npu_profile;
use furiosa_opt_gemma4::{Chip, axes::*, host::fixed_e4_codec as codec, ops_pool_views::Rc64};
use furiosa_opt_gemma4::{
    ops_decode32_attention as b32, ops_fixed_e4_decode32 as e32, ops_fixed_e4_r64 as efull,
    ops_fixed_e4_r64_head_rows as eheads, ops_fixed_e4_r64_mixed512 as emixed,
    ops_fixed_e4_r64_prefix512 as e512, ops_fixed_e4_r64_prefix768 as e768,
    ops_request_sliding_r64 as bfull, ops_request_sliding_r64_head_rows as bheads,
    ops_request_sliding_r64_mixed512 as bmixed, ops_request_sliding_r64_prefix512 as b512,
    ops_request_sliding_r64_prefix768 as b768,
};
use furiosa_opt_std::prelude::*;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
type Q = HbmTensor<bf16, Chip, m![R64, Ns, Gs, Ds]>;
type BP = HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>;
type EP = HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>;
type Rings = HbmTensor<i32, Chip, m![R64]>;
type Mask = HbmTensor<f32, Chip, m![R64, Ts]>;
const KV: usize = 2048;
const QR: usize = 4096;
const POOL: usize = 32 * 1024 * KV;
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
    assert_eq!(a.len(), b.len(), "{label}");
    let n = a.iter().zip(b).filter(|(a, b)| a != b).count();
    if n != 0 {
        let first: Vec<_> = a
            .iter()
            .zip(b)
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .take(8)
            .collect();
        panic!("{label} mismatch={n} first={first:?}");
    }
    a.len()
}
fn pool(seed: usize) -> Vec<u8> {
    (0..POOL)
        .map(|i| {
            let h = i % KV;
            let t = (i / KV) % 1024;
            let s = i / (KV * 1024);
            let x = h.wrapping_mul(17) ^ t.wrapping_mul(29) ^ s.wrapping_mul(73) ^ seed;
            // Signed zeros, E4 subnormals and moderate normals; no nonfinite read payloads.
            let mag = if x % 31 == 0 {
                0
            } else if x % 13 == 0 {
                (x % 8) as u8
            } else {
                0x20 + (x % 49) as u8
            };
            mag | if x & 256 == 0 { 0 } else { 128 }
        })
        .collect()
}
fn query(seed: usize, live: usize) -> Vec<u8> {
    (0..64 * QR)
        .flat_map(|i| {
            let row = i / QR;
            let x = if row >= live {
                0.0
            } else {
                (((i * 37 + seed * 101) % 257) as f32 - 128.0) / 512.0
            };
            bf16::from_f32(x).to_bits().to_le_bytes()
        })
        .collect()
}
fn poison() -> Vec<u8> {
    (0..64 * QR)
        .flat_map(|i| (0x3d00u16 + (i % 127) as u16).to_le_bytes())
        .collect()
}
fn permutation(b: &[u8], live: usize) -> Vec<u8> {
    let mut o = b.to_vec();
    for r in 0..live {
        o[r * QR * 2..(r + 1) * QR * 2]
            .copy_from_slice(&b[(live - 1 - r) * QR * 2..(live - r) * QR * 2]);
    }
    o
}
fn shape(case: usize) -> (usize, usize) {
    match case {
        0 => (0, 17),
        1 => (0, 31),
        2 => (0, 32),
        3 => (1, 32),
        4 => (2, 31),
        5 => (3, 32),
        6 => (4, 32),
        7 => (5, 32),
        8 => (4, 64),
        _ => (0, 17),
    }
}
fn positions(case: usize, live: usize) -> Vec<usize> {
    (0..64)
        .map(|r| {
            if r >= live {
                0
            } else {
                match case {
                    0 | 3 => [0, 255, 256, 510, 511][r % 5],
                    4 => [256, 511, 512, 767][r % 4],
                    1 | 5 => {
                        if (r / 8) % 2 == 0 {
                            [0, 255, 511][r % 3]
                        } else {
                            [512, 767, 768, 1023, 1024, 8191][r % 6]
                        }
                    }
                    _ => [512, 767, 768, 1023, 1024, 8191][r % 6],
                }
            }
        })
        .collect()
}
macro_rules! invoke {
    ($f:path,$d:expr,$q:expr,$k:expr,$v:expr,$r:expr,$m:expr,$o:expr) => {
        launch($f, ($d, $q, $k, $v, $r, $m, $o)).await?
    };
}
macro_rules! policy {
    ($f:path,$d:expr,$q:expr,$k:expr,$v:expr,$r:expr,$m:expr,$p:expr,$o:expr) => {
        launch($f, ($d, $q, $k, $v, $r, $m, $p, $o)).await?
    };
}
async fn br(
    d: &mut Device,
    kind: usize,
    q: &Q,
    k: &BP,
    v: &BP,
    r: &Rings,
    m: &Mask,
    p: &HbmScalar<i32>,
    o: &mut Q,
) -> Result<()> {
    match kind {
        0 => policy!(b32::sliding_attention_r64, d, q, k, v, r, m, p, o),
        1 => invoke!(b512::sliding_attention_r64, d, q, k, v, r, m, o),
        2 => invoke!(b768::sliding_attention_r64, d, q, k, v, r, m, o),
        3 => policy!(bmixed::sliding_attention_r64, d, q, k, v, r, m, p, o),
        4 => invoke!(bheads::sliding_attention_r64, d, q, k, v, r, m, o),
        _ => invoke!(bfull::sliding_attention_r64, d, q, k, v, r, m, o),
    };
    Ok(())
}
async fn er(
    d: &mut Device,
    kind: usize,
    q: &Q,
    k: &EP,
    v: &EP,
    r: &Rings,
    m: &Mask,
    p: &HbmScalar<i32>,
    o: &mut Q,
) -> Result<()> {
    match kind {
        0 => policy!(e32::sliding_attention_r64, d, q, k, v, r, m, p, o),
        1 => invoke!(e512::sliding_attention_r64, d, q, k, v, r, m, o),
        2 => invoke!(e768::sliding_attention_r64, d, q, k, v, r, m, o),
        3 => policy!(emixed::sliding_attention_r64, d, q, k, v, r, m, p, o),
        4 => invoke!(eheads::sliding_attention_r64, d, q, k, v, r, m, o),
        _ => invoke!(efull::sliding_attention_r64, d, q, k, v, r, m, o),
    };
    Ok(())
}
#[tokio::main]
async fn main() -> Result<()> {
    let profile = npu_profile::Collector::default();
    tracing::subscriber::set_global_default(profile.clone())?;
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
    assert!(first < end && end <= 9);
    assert_eq!(Rc64::SIZE, 32);
    let mut d = Device::new(e32::sliding_attention_r64.topology())?;
    let (mut output_bytes, mut immutable_bytes) = (0usize, 0usize);
    for case in first..end {
        let (kind, live) = shape(case);
        let ec_k = pool(31 + case);
        let ec_v = pool(97 + case);
        let bc_k = codec::decode_bytes(&ec_k);
        let bc_v = codec::decode_bytes(&ec_v);
        let ek: EP = ue(&mut d, &ec_k).await?;
        let ev: EP = ue(&mut d, &ec_v).await?;
        let bk: BP = ub(&mut d, &bc_k).await?;
        let bv: BP = ub(&mut d, &bc_v).await?;
        let mut qb = query(case + 5, live);
        for step in 0..2 {
            let mut pos = positions(case, live);
            let mut slots: Vec<_> = (0..64).map(|r| (r * 13 + case * 3) % 32).collect();
            if step == 1 {
                qb = permutation(&qb, live);
                pos[..live].reverse();
                slots[..live].reverse();
            }
            let ph = pos.iter().take(live).enumerate().fold(255u8, |p, (r, &t)| {
                if t >= 512 { p & !(1 << (r / 8)) } else { p }
            });
            let p = HbmScalar::from_host(i32::from(ph), &mut d.pdma).await?;
            let erows: Vec<i32> = slots.iter().map(|&s| (s * 1024 * KV) as i32).collect();
            let brows: Vec<i32> = erows.iter().map(|&o| o * 2).collect();
            let rr = HostTensor::<i32, m![R64]>::from_vec(erows.clone())
                .to_hbm(&mut d.pdma)
                .await?;
            let rbases = HostTensor::<i32, m![R64]>::from_vec(brows.clone())
                .to_hbm(&mut d.pdma)
                .await?;
            let mut mh = Vec::with_capacity(64 * 1024);
            for r in 0..64 {
                for t in 0..1024 {
                    mh.push(if r < live && t <= pos[r].min(1023) {
                        1f32
                    } else {
                        0.
                    });
                }
            }
            let m = HostTensor::<f32, m![R64, Ts]>::from_vec(mh.clone())
                .to_hbm(&mut d.pdma)
                .await?;
            let q: Q = ub(&mut d, &qb).await?;
            let pb = poison();
            let mut out: Q = ub(&mut d, &pb).await?;
            let mut control: Q = ub(&mut d, &pb).await?;
            er(&mut d, kind, &q, &ek, &ev, &rr, &m, &p, &mut out).await?;
            let got = rb(&mut d, &out).await?;
            br(&mut d, kind, &q, &bk, &bv, &rbases, &m, &p, &mut control).await?;
            output_bytes += exact(
                "same-policy intended BF16",
                &got,
                &rb(&mut d, &control).await?,
            );
            if kind == 0 {
                output_bytes += exact(
                    "R32 inactive physical tail",
                    &got[32 * QR * 2..],
                    &pb[32 * QR * 2..],
                );
            }
            // Unchanged full-window BF16 reader additionally proves the prefix/mixed policy's omitted region is masked.
            let mut full: Q = ub(&mut d, &pb).await?;
            br(&mut d, 5, &q, &bk, &bv, &rbases, &m, &p, &mut full).await?;
            let full_bytes = rb(&mut d, &full).await?;
            let count = if kind == 0 { 32 } else { 64 };
            output_bytes += exact(
                "full-window intended reference",
                &got[..count * QR * 2],
                &full_bytes[..count * QR * 2],
            );
            assert!(
                got[..live * QR * 2]
                    .chunks_exact(2)
                    .all(|x| u16::from_le_bytes([x[0], x[1]]) & 0x7f80 != 0x7f80),
                "active nonfinite"
            );
            println!(
                "FIXED_E4_READER_EXACT case={case} step={step} kind={kind} live={live} policy={ph} physical_slots=32 output_bytes={output_bytes} tail_preserved={}",
                kind == 0
            );
            for (round, candidate) in [false, true, true, false].into_iter().enumerate() {
                let start = std::time::Instant::now();
                for _ in 0..6 {
                    if candidate {
                        er(&mut d, kind, &q, &ek, &ev, &rr, &m, &p, &mut out).await?;
                    } else {
                        br(&mut d, kind, &q, &bk, &bv, &rbases, &m, &p, &mut control).await?;
                    }
                }
                println!(
                    "FIXED_E4_READER_TIME case={case} step={step} round={round} candidate={candidate} us={:.6} calls=6",
                    start.elapsed().as_secs_f64() * 1e6 / 6.
                );
            }
            output_bytes += exact("timed output", &got, &rb(&mut d, &out).await?);
            if step == 1 && std::env::var_os("FURIOSA_OPT_PROFILE").is_some() {
                for candidate in [false, true] {
                    println!("PROFILE_VARIANT case={case} candidate={candidate}");
                    profile.enable(true);
                    if candidate {
                        er(&mut d, kind, &q, &ek, &ev, &rr, &m, &p, &mut out).await?;
                    } else {
                        br(&mut d, kind, &q, &bk, &bv, &rbases, &m, &p, &mut control).await?;
                    }
                    profile.enable(false);
                    println!("PROFILE_END");
                }
            }
            immutable_bytes += exact("query immutable", &qb, &rb(&mut d, &q).await?);
            immutable_bytes += exact(
                "E4 offsets immutable",
                &rr.to_host::<m![R64]>(&mut d.pdma)
                    .await?
                    .into_inner()
                    .into_buf(),
                &erows
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
            immutable_bytes += exact(
                "BF16 offsets immutable",
                &rbases
                    .to_host::<m![R64]>(&mut d.pdma)
                    .await?
                    .into_inner()
                    .into_buf(),
                &brows
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
            immutable_bytes += exact(
                "mask immutable",
                &m.to_host::<m![R64, Ts]>(&mut d.pdma)
                    .await?
                    .into_inner()
                    .into_buf(),
                &mh.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
            );
            qb = got;
            if kind == 0 {
                qb[32 * QR * 2..].fill(0);
            }
        }
        immutable_bytes += exact("E4 K whole immutable", &ec_k, &re(&mut d, &ek).await?)
            + exact("E4 V whole immutable", &ec_v, &re(&mut d, &ev).await?)
            + exact("BF16 K whole immutable", &bc_k, &rb(&mut d, &bk).await?)
            + exact("BF16 V whole immutable", &bc_v, &rb(&mut d, &bv).await?);
    }
    println!(
        "PASS fixed E4 Rc32 readers cases={first}..{end} output_bytes={output_bytes} immutable_bytes={immutable_bytes} synthetic_history=true no_model_quality_claim=true"
    );
    Ok(())
}
