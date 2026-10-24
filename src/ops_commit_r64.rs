//! Direct R64 commits require 64 live independent requests and distinct request slots.
//! Decode commits its current K/V before attention. B16 prefix wrappers also serve
//! prompt chunks, whose sliding commits must follow attention over the immutable old ring.
use crate::{Chip, axes::{R64, B, Ns, Ds, Df, Ts, Tf}};
use furiosa_opt_std::prelude::*;
use crate::{ops_pool_views::Rc64,ops_request_attention::Fc};
type Cluster=m![R64/32];
#[device(chip=1)]
pub fn commit_sliding_kv_r64(device:&mut Device,k:&HbmTensor<bf16,Chip,m![R64,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![R64,Ns,Ds]>,offsets:&HbmTensor<i32,Chip,m![R64]>,k_cache:&mut HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>,v_cache:&mut HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>){
 type Rows=m![R64%32,Ns];
 let k:DmTensor<bf16,Chip,Cluster,Rows,m![Ds]>=k.to_dm(&mut device.tdma);
 let v:DmTensor<bf16,Chip,Cluster,Rows,m![Ds]>=v.to_dm(&mut device.tdma);
 k_cache.view_mut().scatter::<m![Rc64,Ts],m![Ns,Ds]>().by_byte_offsets(offsets.view()).from_dm(&mut device.tdma,k);
 v_cache.view_mut().scatter::<m![Rc64,Ts],m![Ns,Ds]>().by_byte_offsets(offsets.view()).from_dm(&mut device.tdma,v);
}
#[device(chip=1)]
pub fn commit_full_kv_r64(device:&mut Device,k:&HbmTensor<bf16,Chip,m![R64,Df]>,v:&HbmTensor<bf16,Chip,m![R64,Df]>,offsets:&HbmTensor<i32,Chip,m![R64]>,k_cache:&mut HbmTensor<bf16,Chip,m![Fc,Tf,Df]>,v_cache:&mut HbmTensor<bf16,Chip,m![Fc,Tf,Df]>){
 type Rows=m![R64%32,Df/64];
 let k:DmTensor<bf16,Chip,Cluster,Rows,m![Df%64]>=k.to_dm(&mut device.tdma);
 let v:DmTensor<bf16,Chip,Cluster,Rows,m![Df%64]>=v.to_dm(&mut device.tdma);
 k_cache.view_mut().scatter::<m![Fc,Tf],m![Df]>().by_byte_offsets(offsets.view()).from_dm(&mut device.tdma,k);
 v_cache.view_mut().scatter::<m![Fc,Tf],m![Df]>().by_byte_offsets(offsets.view()).from_dm(&mut device.tdma,v);
}


fn commit_sliding_pool<const VALID: usize>(
    device: &mut Device,
    k: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![B, Ns, Ds]>,
    byte_offsets: &HbmTensor<i32, Chip, m![B = { VALID }]>,
    pool_k: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    pool_v: &mut HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
) {
    let k: DmTensor<bf16, Chip, crate::device::layout::Cluster, crate::device::layout::Slice, m![B = { VALID }, Ns, Ds]> = k
        .view()
        .tile::<m![B], VALID, m![B={VALID} # 16,Ns,Ds]>(0)
        .to_dm(&mut device.tdma);
    let v: DmTensor<bf16, Chip, crate::device::layout::Cluster, crate::device::layout::Slice, m![B = { VALID }, Ns, Ds]> = v
        .view()
        .tile::<m![B], VALID, m![B={VALID} # 16,Ns,Ds]>(0)
        .to_dm(&mut device.tdma);
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
pub fn commit_sliding_pool_1(device:&mut Device,k:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,offsets:&HbmTensor<i32,Chip,m![B=1]>,pool_k:&mut HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>,pool_v:&mut HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>){commit_sliding_pool::<1>(device,k,v,offsets,pool_k,pool_v);}

#[device(chip=1)]
pub fn commit_sliding_pool_2(device:&mut Device,k:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,offsets:&HbmTensor<i32,Chip,m![B=2]>,pool_k:&mut HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>,pool_v:&mut HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>){commit_sliding_pool::<2>(device,k,v,offsets,pool_k,pool_v);}

#[device(chip=1)]
pub fn commit_sliding_pool_3(device:&mut Device,k:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,offsets:&HbmTensor<i32,Chip,m![B=3]>,pool_k:&mut HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>,pool_v:&mut HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>){commit_sliding_pool::<3>(device,k,v,offsets,pool_k,pool_v);}

#[device(chip=1)]
pub fn commit_sliding_pool_4(device:&mut Device,k:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,offsets:&HbmTensor<i32,Chip,m![B=4]>,pool_k:&mut HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>,pool_v:&mut HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>){commit_sliding_pool::<4>(device,k,v,offsets,pool_k,pool_v);}

#[device(chip=1)]
pub fn commit_sliding_pool_5(device:&mut Device,k:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,offsets:&HbmTensor<i32,Chip,m![B=5]>,pool_k:&mut HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>,pool_v:&mut HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>){commit_sliding_pool::<5>(device,k,v,offsets,pool_k,pool_v);}

#[device(chip=1)]
pub fn commit_sliding_pool_6(device:&mut Device,k:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,offsets:&HbmTensor<i32,Chip,m![B=6]>,pool_k:&mut HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>,pool_v:&mut HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>){commit_sliding_pool::<6>(device,k,v,offsets,pool_k,pool_v);}

#[device(chip=1)]
pub fn commit_sliding_pool_7(device:&mut Device,k:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,offsets:&HbmTensor<i32,Chip,m![B=7]>,pool_k:&mut HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>,pool_v:&mut HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>){commit_sliding_pool::<7>(device,k,v,offsets,pool_k,pool_v);}

#[device(chip=1)]
pub fn commit_sliding_pool_8(device:&mut Device,k:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,offsets:&HbmTensor<i32,Chip,m![B=8]>,pool_k:&mut HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>,pool_v:&mut HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>){commit_sliding_pool::<8>(device,k,v,offsets,pool_k,pool_v);}

#[device(chip=1)]
pub fn commit_sliding_pool_9(device:&mut Device,k:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,offsets:&HbmTensor<i32,Chip,m![B=9]>,pool_k:&mut HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>,pool_v:&mut HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>){commit_sliding_pool::<9>(device,k,v,offsets,pool_k,pool_v);}

#[device(chip=1)]
pub fn commit_sliding_pool_10(device:&mut Device,k:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,offsets:&HbmTensor<i32,Chip,m![B=10]>,pool_k:&mut HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>,pool_v:&mut HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>){commit_sliding_pool::<10>(device,k,v,offsets,pool_k,pool_v);}

#[device(chip=1)]
pub fn commit_sliding_pool_11(device:&mut Device,k:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,offsets:&HbmTensor<i32,Chip,m![B=11]>,pool_k:&mut HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>,pool_v:&mut HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>){commit_sliding_pool::<11>(device,k,v,offsets,pool_k,pool_v);}

#[device(chip=1)]
pub fn commit_sliding_pool_12(device:&mut Device,k:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,offsets:&HbmTensor<i32,Chip,m![B=12]>,pool_k:&mut HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>,pool_v:&mut HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>){commit_sliding_pool::<12>(device,k,v,offsets,pool_k,pool_v);}

#[device(chip=1)]
pub fn commit_sliding_pool_13(device:&mut Device,k:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,offsets:&HbmTensor<i32,Chip,m![B=13]>,pool_k:&mut HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>,pool_v:&mut HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>){commit_sliding_pool::<13>(device,k,v,offsets,pool_k,pool_v);}

#[device(chip=1)]
pub fn commit_sliding_pool_14(device:&mut Device,k:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,offsets:&HbmTensor<i32,Chip,m![B=14]>,pool_k:&mut HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>,pool_v:&mut HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>){commit_sliding_pool::<14>(device,k,v,offsets,pool_k,pool_v);}

#[device(chip=1)]
pub fn commit_sliding_pool_15(device:&mut Device,k:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,offsets:&HbmTensor<i32,Chip,m![B=15]>,pool_k:&mut HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>,pool_v:&mut HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>){commit_sliding_pool::<15>(device,k,v,offsets,pool_k,pool_v);}

#[device(chip=1)]
pub fn commit_sliding_pool_16(device:&mut Device,k:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![B,Ns,Ds]>,offsets:&HbmTensor<i32,Chip,m![B=16]>,pool_k:&mut HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>,pool_v:&mut HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>){commit_sliding_pool::<16>(device,k,v,offsets,pool_k,pool_v);}

fn load_tile<Payload: M, PayloadSlices:M, Elements:M>(
    device: &mut Device,
    source: &HbmTensor<bf16, Chip, m![R64, { Payload }]>,
    tile_index: &HbmTensor<i32, Chip, m![1]>,
    output: &mut HbmTensor<bf16, Chip, m![B, { Payload }]>,
) {
    let index: DmTensor<i32, Chip, crate::device::layout::Cluster, crate::device::layout::Slice, m![1 # 8]> = tile_index.to_dm(&mut device.tdma);
    let index: SpmTensor<i32, Chip, crate::device::layout::Cluster, m![1 # 4], m![1]> = index.to_spm(&mut device.tdma);
    let tile: DmTensor<bf16, Chip, crate::device::layout::Cluster, m![R64%16,{PayloadSlices}], Elements> = source
        .gather::<m![R64 / 16], m![R64 % 16, { Payload }]>()
        .by_positions::<m![1]>(&index)
        .to_dm(&mut device.tdma);
    let tile: DmTensor<bf16, Chip, crate::device::layout::Cluster, m![B,{PayloadSlices}], Elements> = unsafe { tile.reshape() };
    tile.view().to_hbm_view(&mut device.tdma, output.view_mut());
}


#[device(chip=1)]
pub fn load_sliding_kv_tile(device:&mut Device,k:&HbmTensor<bf16,Chip,m![R64,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![R64,Ns,Ds]>,tile_index:&HbmTensor<i32,Chip,m![1]>,out_k:&mut HbmTensor<bf16,Chip,m![B,Ns,Ds]>,out_v:&mut HbmTensor<bf16,Chip,m![B,Ns,Ds]>){load_tile::<m![Ns,Ds],m![Ns,Ds/128],m![Ds%128]>(device,k,tile_index,out_k);load_tile::<m![Ns,Ds],m![Ns,Ds/128],m![Ds%128]>(device,v,tile_index,out_v);}

#[device(chip=1)]
pub fn load_full_kv_tile(device:&mut Device,k:&HbmTensor<bf16,Chip,m![R64,Df]>,v:&HbmTensor<bf16,Chip,m![R64,Df]>,tile_index:&HbmTensor<i32,Chip,m![1]>,out_k:&mut HbmTensor<bf16,Chip,m![B,Df]>,out_v:&mut HbmTensor<bf16,Chip,m![B,Df]>){load_tile::<m![Df],m![Df/32],m![Df%32]>(device,k,tile_index,out_k);load_tile::<m![Df],m![Df/32],m![Df%32]>(device,v,tile_index,out_v);}

/// Diagnostic exact row gather; offsets address absolute bytes in the whole pool.
#[device(chip=1)]
pub fn gather_sliding_rows_r64(device:&mut Device,k:&HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>,v:&HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>,offsets:&HbmTensor<i32,Chip,m![R64]>,out_k:&mut HbmTensor<bf16,Chip,m![R64,Ns,Ds]>,out_v:&mut HbmTensor<bf16,Chip,m![R64,Ns,Ds]>){
 let k:DmTensor<bf16,Chip,Cluster,m![R64%32,Ns],m![Ds]>=k.gather::<m![Rc64,Ts],m![Ns,Ds]>().by_byte_offsets(offsets).to_dm(&mut device.tdma);let v:DmTensor<bf16,Chip,Cluster,m![R64%32,Ns],m![Ds]>=v.gather::<m![Rc64,Ts],m![Ns,Ds]>().by_byte_offsets(offsets).to_dm(&mut device.tdma);k.view().to_hbm_view(&mut device.tdma,out_k.view_mut());v.view().to_hbm_view(&mut device.tdma,out_v.view_mut());}

/// Diagnostic exact row gather; offsets address absolute bytes in the whole pool.
#[device(chip=1)]
pub fn gather_full_rows_r64(device:&mut Device,k:&HbmTensor<bf16,Chip,m![Fc,Tf,Df]>,v:&HbmTensor<bf16,Chip,m![Fc,Tf,Df]>,offsets:&HbmTensor<i32,Chip,m![R64]>,out_k:&mut HbmTensor<bf16,Chip,m![R64,Df]>,out_v:&mut HbmTensor<bf16,Chip,m![R64,Df]>){
 let k:DmTensor<bf16,Chip,Cluster,m![R64%32,Df/64],m![Df%64]>=k.gather::<m![Fc,Tf],m![Df]>().by_byte_offsets(offsets).to_dm(&mut device.tdma);let v:DmTensor<bf16,Chip,Cluster,m![R64%32,Df/64],m![Df%64]>=v.gather::<m![Fc,Tf],m![Df]>().by_byte_offsets(offsets).to_dm(&mut device.tdma);k.view().to_hbm_view(&mut device.tdma,out_k.view_mut());v.view().to_hbm_view(&mut device.tdma,out_v.view_mut());}
