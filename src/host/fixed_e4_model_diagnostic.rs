//! Read-only numerical-surrogate diagnostics. Summaries/samples are not an
//! all-element cross-program error oracle; same-program equality is separate.
use super::{runtime::Workspace, sampling::{sample,SamplingConfig}};
use crate::{axes::W, host_logit_bytes::bf16_bits_to_f32};
use furiosa_opt_std::prelude::*;
use rand::{SeedableRng,rngs::StdRng};
use std::collections::HashSet;
type Result<T> = std::result::Result<T,Box<dyn std::error::Error>>;

pub fn report(label:&str,bits:&[u16]) {
    assert_eq!(bits.len(),W::SIZE);
    let values:Vec<_>=bits.iter().map(|&b|bf16_bits_to_f32(b)).collect();
    assert!(values.iter().all(|x|x.is_finite()),"nonfinite full vocabulary: {label}");
    let (mut sum,mut l1,mut sq)=(0f64,0f64,0f64);
    let (mut lo,mut hi)=(f32::INFINITY,f32::NEG_INFINITY);
    let mut checksum=0xcbf29ce484222325u64;
    for (&b,&x) in bits.iter().zip(&values) {
        sum+=f64::from(x);l1+=f64::from(x).abs();sq+=f64::from(x).powi(2);lo=lo.min(x);hi=hi.max(x);
        for byte in b.to_le_bytes(){checksum=(checksum^u64::from(byte)).wrapping_mul(0x100000001b3);}
    }
    let none=HashSet::new();let banned=[0,1,106,107].into_iter().collect();
    let top1=sample(&values,&SamplingConfig::DEFAULT,&none,&mut StdRng::seed_from_u64(3));
    let greedy=sample(&values,&SamplingConfig::DEFAULT,&banned,&mut StdRng::seed_from_u64(71));
    let stochastic=sample(&values,&SamplingConfig{temperature:0.8,top_k:37,top_p:0.9},&banned,&mut StdRng::seed_from_u64(71));
    let samples=(0..128usize).map(|i|{let p=(17+i*2053)%W::SIZE;format!("{p}:{:04x}",bits[p])}).collect::<Vec<_>>().join(",");
    println!("FIXED_E4_ROW label={label} enabled={} values={} nonfinite=0 top1={top1} top1_bits={:04x} banned_greedy={greedy} seeded_stochastic={stochastic} sum={sum:.17e} l1={l1:.17e} squared_l2={sq:.17e} min={lo} max={hi} checksum_fnv64={checksum:016x} samples={samples} cross_program_scope=summary_and_128_samples_only quality_gate=false",crate::ops_fixed_e4_quality::ENABLED,bits.len(),bits[top1]);
}
pub fn exact(label:&str,a:&[u16],b:&[u16]) {
    assert_eq!(a.len(),b.len());
    let different=a.iter().zip(b).filter(|(a,b)|a!=b).count();
    println!("FIXED_E4_REPLAY label={label} values={} bitdiff={different}",a.len());
    assert_eq!(different,0,"same-program all-vocabulary replay: {label}");
    report(label,a);
    let av:Vec<_>=a.iter().map(|&x|bf16_bits_to_f32(x)).collect();
    let bv:Vec<_>=b.iter().map(|&x|bf16_bits_to_f32(x)).collect();
    let bans=[0,1,106,107].into_iter().collect();
    for cfg in [SamplingConfig::DEFAULT,SamplingConfig{temperature:0.8,top_k:37,top_p:0.9}] {
        assert_eq!(sample(&av,&cfg,&bans,&mut StdRng::seed_from_u64(71)),sample(&bv,&cfg,&bans,&mut StdRng::seed_from_u64(71)));
    }
}
pub fn exact_f32(label:&str,a:&[f32],b:&[f32]) {
    let convert=|row:&[f32]|row.iter().map(|x|{assert!(x.is_finite());assert_eq!(x.to_bits()&0xffff,0);(x.to_bits()>>16)as u16}).collect::<Vec<_>>();
    exact(label,&convert(a),&convert(b));
}
/// Read actual owning-buffer metadata before any model work. This catches a
/// stale shared-target library even when the requested admission count is32.
pub fn verify_physical32_identity(w: &Workspace, expected_context: usize) -> Result<()> {
    use crate::{axes::*, ops_pool_views::Rc64, ops_request_attention::Fc};
    assert!(w.is_canonical());
    assert_eq!((Rc64::SIZE, R64::SIZE, Fc::SIZE), (32, 64, 256));
    assert_eq!((w.requests(), w.physical_full_pages(), w.capacity()), (32, 256, expected_context));
    #[cfg(backend = "npu")]
    {
        use crate::host::runtime::KvCache;
        use furiosa_opt_std::runtime::{Buffers, DeviceSend};
        let sliding_bytes = 32 * Ts::SIZE * Ns::SIZE * Ds::SIZE * 2;
        let full_bytes = 256 * Tf::SIZE * Df::SIZE * 2;
        let (mut sliding, mut full, mut bytes) = (0, 0, 0usize);
        for cache in &w.kv_cache {
            let mut bound = Buffers::new();
            let expected = match cache {
                KvCache::SlidingCanonical { k, v } => {
                    DeviceSend::bind(k, &mut bound)?;
                    DeviceSend::bind(v, &mut bound)?;
                    sliding += 1;
                    sliding_bytes
                }
                KvCache::FullCanonical { k, v } => {
                    DeviceSend::bind(k, &mut bound)?;
                    DeviceSend::bind(v, &mut bound)?;
                    full += 1;
                    full_bytes
                }
                _ => return Err("diagnostic requires canonical BF16 owning pools".into()),
            };
            assert_eq!(bound.len(), 2);
            for buffer in bound.iter() {
                assert_eq!(buffer.size(), expected, "actual owning KV buffer extent");
                bytes += buffer.size();
            }
        }
        assert_eq!((sliding, full, bytes), (40, 8, 12_884_901_888));
        println!("FIXED_E4_STORAGE_IDENTITY physical_rings=32 admission=32 query_rows=64 full_pages=256 context={expected_context} sliding_layers={sliding} full_layers={full} allocated_kv_bytes={bytes} verified_owning_buffers=true");
    }
    Ok(())
}

