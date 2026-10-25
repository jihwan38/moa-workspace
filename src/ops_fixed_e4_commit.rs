//! Fixed scale1 E4 append. Caller supplies byte offsets for the E4 pools.
use crate::{Chip, axes::{B,Ns,Ds,Ts,R64},ops_pool_views::Rc64};
use furiosa_opt_std::prelude::*;
fn encode<const VALID: usize>(device:&mut Device,x:&DmTensor<bf16,Chip,crate::device::layout::Cluster,crate::device::layout::Slice,m![B={VALID},Ns,Ds]>)->DmTensor<f8e4m3,Chip,crate::device::layout::Cluster,crate::device::layout::Slice,m![B={VALID},Ns,Ds]>{
    device.main.begin(x.view())
        .fetch::<m![B={VALID},Ns,Ds/16],m![Ds%16]>()
        .fetch_cast::<f32>()
        .collect::<m![B={VALID},Ns,Ds/8],m![Ds%8]>()
        .cast::<f8e4m3,m![Ds%8 #32]>()
        .commit_trim::<m![Ds%8]>().commit()
}
pub(crate) fn commit_sliding_pool<const VALID: usize>(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    byte_offsets: &HbmTensor<i32, Chip, m![B = { VALID }]>,
    pool_k: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
) {
    let k: DmTensor<bf16, Chip, crate::device::layout::Cluster, crate::device::layout::Slice, m![B = { VALID }, Ns, Ds]> = k
        .view()
        .tile::<m![B], VALID, m![B={VALID} # 16,Ns,Ds]>(0)
        .to_dm(&mut device.tdma);
    let v: DmTensor<bf16, Chip, crate::device::layout::Cluster, crate::device::layout::Slice, m![B = { VALID }, Ns, Ds]> = v
        .view()
        .tile::<m![B], VALID, m![B={VALID} # 16,Ns,Ds]>(0)
        .to_dm(&mut device.tdma);
    let k = encode::<VALID>(device, &k);
    let v = encode::<VALID>(device, &v);
    pool_k
        .view_mut()
        .scatter::<m![Rc64, Ts], m![Ns, Ds]>()
        .by_byte_offsets(byte_offsets.view())
        .from_dm(&mut device.tdma, k);
    pool_v
        .view_mut()
        .scatter::<m![Rc64, Ts], m![Ns, Ds]>()
        .by_byte_offsets(byte_offsets.view())
        .from_dm(&mut device.tdma, v);
}

#[device(chip=1)]
pub fn commit_sliding_pool_1(device:&mut Device,k:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,offsets:&HbmTensor<i32,Chip,m![B=1]>,pool_k:&mut HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>,pool_v:&mut HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>){commit_sliding_pool::<1>(device,k,v,offsets,pool_k,pool_v);}

#[device(chip=1)]
pub fn commit_sliding_pool_2(device:&mut Device,k:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,offsets:&HbmTensor<i32,Chip,m![B=2]>,pool_k:&mut HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>,pool_v:&mut HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>){commit_sliding_pool::<2>(device,k,v,offsets,pool_k,pool_v);}

#[device(chip=1)]
pub fn commit_sliding_pool_3(device:&mut Device,k:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,offsets:&HbmTensor<i32,Chip,m![B=3]>,pool_k:&mut HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>,pool_v:&mut HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>){commit_sliding_pool::<3>(device,k,v,offsets,pool_k,pool_v);}

#[device(chip=1)]
pub fn commit_sliding_pool_4(device:&mut Device,k:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,offsets:&HbmTensor<i32,Chip,m![B=4]>,pool_k:&mut HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>,pool_v:&mut HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>){commit_sliding_pool::<4>(device,k,v,offsets,pool_k,pool_v);}

#[device(chip=1)]
pub fn commit_sliding_pool_5(device:&mut Device,k:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,offsets:&HbmTensor<i32,Chip,m![B=5]>,pool_k:&mut HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>,pool_v:&mut HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>){commit_sliding_pool::<5>(device,k,v,offsets,pool_k,pool_v);}

#[device(chip=1)]
pub fn commit_sliding_pool_6(device:&mut Device,k:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,offsets:&HbmTensor<i32,Chip,m![B=6]>,pool_k:&mut HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>,pool_v:&mut HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>){commit_sliding_pool::<6>(device,k,v,offsets,pool_k,pool_v);}

#[device(chip=1)]
pub fn commit_sliding_pool_7(device:&mut Device,k:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,offsets:&HbmTensor<i32,Chip,m![B=7]>,pool_k:&mut HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>,pool_v:&mut HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>){commit_sliding_pool::<7>(device,k,v,offsets,pool_k,pool_v);}

#[device(chip=1)]
pub fn commit_sliding_pool_8(device:&mut Device,k:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,offsets:&HbmTensor<i32,Chip,m![B=8]>,pool_k:&mut HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>,pool_v:&mut HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>){commit_sliding_pool::<8>(device,k,v,offsets,pool_k,pool_v);}

#[device(chip=1)]
pub fn commit_sliding_pool_9(device:&mut Device,k:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,offsets:&HbmTensor<i32,Chip,m![B=9]>,pool_k:&mut HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>,pool_v:&mut HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>){commit_sliding_pool::<9>(device,k,v,offsets,pool_k,pool_v);}

#[device(chip=1)]
pub fn commit_sliding_pool_10(device:&mut Device,k:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,offsets:&HbmTensor<i32,Chip,m![B=10]>,pool_k:&mut HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>,pool_v:&mut HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>){commit_sliding_pool::<10>(device,k,v,offsets,pool_k,pool_v);}

#[device(chip=1)]
pub fn commit_sliding_pool_11(device:&mut Device,k:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,offsets:&HbmTensor<i32,Chip,m![B=11]>,pool_k:&mut HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>,pool_v:&mut HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>){commit_sliding_pool::<11>(device,k,v,offsets,pool_k,pool_v);}

#[device(chip=1)]
pub fn commit_sliding_pool_12(device:&mut Device,k:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,offsets:&HbmTensor<i32,Chip,m![B=12]>,pool_k:&mut HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>,pool_v:&mut HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>){commit_sliding_pool::<12>(device,k,v,offsets,pool_k,pool_v);}

#[device(chip=1)]
pub fn commit_sliding_pool_13(device:&mut Device,k:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,offsets:&HbmTensor<i32,Chip,m![B=13]>,pool_k:&mut HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>,pool_v:&mut HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>){commit_sliding_pool::<13>(device,k,v,offsets,pool_k,pool_v);}

#[device(chip=1)]
pub fn commit_sliding_pool_14(device:&mut Device,k:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,offsets:&HbmTensor<i32,Chip,m![B=14]>,pool_k:&mut HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>,pool_v:&mut HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>){commit_sliding_pool::<14>(device,k,v,offsets,pool_k,pool_v);}

#[device(chip=1)]
pub fn commit_sliding_pool_15(device:&mut Device,k:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,offsets:&HbmTensor<i32,Chip,m![B=15]>,pool_k:&mut HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>,pool_v:&mut HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>){commit_sliding_pool::<15>(device,k,v,offsets,pool_k,pool_v);}

#[device(chip=1)]
pub fn commit_sliding_pool_16(device:&mut Device,k:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,offsets:&HbmTensor<i32,Chip,m![B=16]>,pool_k:&mut HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>,pool_v:&mut HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>){commit_sliding_pool::<16>(device,k,v,offsets,pool_k,pool_v);}

// ABI-compatible symbol; independent Rc32 routing uses live B16 segments.
#[device(chip=1)]
pub fn commit_sliding_kv_r64(device:&mut Device,k:&HbmTensor<bf16,Chip,m![R64,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![R64,Ns,Ds]>,offsets:&HbmTensor<i32,Chip,m![R64]>,k_cache:&mut HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>,v_cache:&mut HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>){
 type Cluster=m![R64/32]; type Rows=m![R64%32,Ns];
 let k:DmTensor<bf16,Chip,Cluster,Rows,m![Ds]>=k.to_dm(&mut device.tdma);
 let v:DmTensor<bf16,Chip,Cluster,Rows,m![Ds]>=v.to_dm(&mut device.tdma);
 let k:DmTensor<f8e4m3,Chip,Cluster,Rows,m![Ds]>=device.main.begin(k.view())
   .fetch::<m![Ds/16],m![Ds%16]>().fetch_cast::<f32>()
   .collect::<m![Ds/8],m![Ds%8]>().cast::<f8e4m3,m![Ds%8#32]>()
   .commit_trim::<m![Ds%8]>().commit();
 let v:DmTensor<f8e4m3,Chip,Cluster,Rows,m![Ds]>=device.main.begin(v.view())
   .fetch::<m![Ds/16],m![Ds%16]>().fetch_cast::<f32>()
   .collect::<m![Ds/8],m![Ds%8]>().cast::<f8e4m3,m![Ds%8#32]>()
   .commit_trim::<m![Ds%8]>().commit();
 k_cache.view_mut().scatter::<m![Rc64,Ts],m![Ns,Ds]>().by_byte_offsets(offsets.view()).from_dm(&mut device.tdma,k);
 v_cache.view_mut().scatter::<m![Rc64,Ts],m![Ns,Ds]>().by_byte_offsets(offsets.view()).from_dm(&mut device.tdma,v);
}
