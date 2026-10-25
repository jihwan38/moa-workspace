//! Complete cache-byte oracle for sequential BF16/E4 model passes. No hashes,
//! tolerance, second device model, or persistent production selector is used.
use super::{runtime::{Workspace,KvCache},fixed_e4_codec};
use crate::{Chip,axes::*};
use furiosa_opt_std::prelude::*;
type Result<T>=std::result::Result<T,Box<dyn std::error::Error>>;


pub struct CacheSnapshot { pools:Vec<Vec<u8>>, pub bytes:usize }
async fn raw<D:MaterializableScalar,Mp:M>(d:&mut Device,x:&HbmTensor<D,Chip,Mp>)->Result<Vec<u8>> {
    Ok(x.to_host::<Mp>(&mut d.pdma).await?.into_inner().into_buf())
}
impl CacheSnapshot {
    /// Every cell in all32 sliding slots and all256 full pages, including
    /// unused owners, padding and future cells. Host memory is intentionally
    /// outside device allocation/timing and released after comparison.
    pub async fn capture_bf16(d:&mut Device,w:&Workspace)->Result<Self> {
        if w.uses_fixed_e4(){return Err("reference must be q022 BF16 storage".into());}
        let mut pools=Vec::with_capacity(96);let mut bytes=0;
        for cache in &w.kv_cache {
            let pair=match cache {
                KvCache::SlidingCanonical{k,v}=>(raw(d,k).await?,raw(d,v).await?),
                KvCache::FullCanonical{k,v}=>(raw(d,k).await?,raw(d,v).await?),
                _=>return Err("reference cache is not canonical".into()),
            };
            bytes+=pair.0.len()+pair.1.len();pools.push(pair.0);pools.push(pair.1);
        }
        assert_eq!(bytes,12_884_901_888usize);
        println!("STORAGE_REFERENCE_CACHE bytes={bytes} pools={} complete_owners=true",pools.len());
        Ok(Self{pools,bytes})
    }
    pub async fn compare_e4(self,d:&mut Device,w:&Workspace)->Result<usize> {
        if !w.uses_fixed_e4(){return Err("candidate must use fixed E4 storage".into());}
        let mut expected=self.pools.into_iter();let mut checked=0usize;
        let lut:Vec<_>=(0..=255u8).map(|c|fixed_e4_codec::decode(c).to_le_bytes()).collect();
        for (layer,cache) in w.kv_cache.iter().enumerate() {
            let (pair,e4)=match cache {
                KvCache::SlidingFixedE4{k,v}=>((raw(d,k).await?,raw(d,v).await?),true),
                KvCache::FullCanonical{k,v}=>((raw(d,k).await?,raw(d,v).await?),false),
                _=>return Err("candidate cache is not canonical".into()),
            };
            for (kind,got) in [pair.0,pair.1].into_iter().enumerate() {
                let reference=expected.next().ok_or("missing reference pool")?;
                if e4 {
                    assert_eq!(reference.len(),got.len()*2);
                    let first=got.iter().zip(reference.chunks_exact(2)).position(|(&code,b)|lut[code as usize]!=b);
                    assert!(first.is_none(),"decoded cache mismatch layer={layer} kind={kind} element={first:?}");
                } else {
                    assert_eq!(reference.len(),got.len());
                    let first=reference.iter().zip(&got).position(|(a,b)|a!=b);
                    assert!(first.is_none(),"full cache mismatch layer={layer} kind={kind} byte={first:?}");
                }
                checked+=reference.len();
            }
        }
        assert!(expected.next().is_none());assert_eq!(checked,self.bytes);
        println!("PASS STORAGE_ALL_CACHE_BYTES compared_bf16_bytes={checked} full_pool_cells=true decoded_e4_exact=true");
        Ok(checked)
    }
}

pub fn verify_identity(w:&Workspace,context:usize,e4:bool)->Result<()> {
    use crate::{ops_pool_views::Rc64,ops_request_attention::Fc};
    assert_eq!((Rc64::SIZE,R64::SIZE,Fc::SIZE),(32,64,256));
    assert_eq!((w.requests(),w.capacity(),w.physical_full_pages(),w.uses_fixed_e4()),(32,context,256,e4));
    #[cfg(backend="npu")]
    {
        use furiosa_opt_std::runtime::{Buffers,DeviceSend};
        let mut bytes=0;let(mut sliding,mut full)=(0,0);
        for cache in &w.kv_cache {
            let mut buffers=Buffers::new();
            let expected=match cache {
                KvCache::SlidingCanonical{k,v}=>{assert!(!e4);DeviceSend::bind(k,&mut buffers)?;DeviceSend::bind(v,&mut buffers)?;sliding+=1;134_217_728},
                KvCache::SlidingFixedE4{k,v}=>{assert!(e4);DeviceSend::bind(k,&mut buffers)?;DeviceSend::bind(v,&mut buffers)?;sliding+=1;67_108_864},
                KvCache::FullCanonical{k,v}=>{DeviceSend::bind(k,&mut buffers)?;DeviceSend::bind(v,&mut buffers)?;full+=1;134_217_728},
                _=>return Err("identity requires canonical owning pools".into()),
            };
            assert_eq!(buffers.len(),2);for b in buffers.iter(){assert_eq!(b.size(),expected);bytes+=b.size();}
        }
        assert_eq!((sliding,full),(40,8));assert_eq!(bytes,if e4{7_516_192_768}else{12_884_901_888});
        println!("STORAGE_IDENTITY e4={e4} physical_rings=32 query_rows=64 admission=32 full_pages=256 context={context} allocated_kv_bytes={bytes} exact_pool_saving_bytes={} verified_owning_buffers=true extra_persistent_ring_scratch_bytes=0",if e4{5_368_709_120usize}else{0});
    }
    Ok(())
}
