//! Raw-byte storage-codec diagnostic; no serving dispatch.
use crate::{axes::*, Chip};
use furiosa_opt_std::prelude::*;

fn decode<C: M, S: M, T: M>(device: &mut Device,
    x: &DmTensor<f8e4m3, Chip, C, S, m![{T}, Ds]>,
) -> DmTensor<bf16, Chip, C, S, m![{T}, Ds]> {
    device.main.begin(x.view())
        .fetch::<m![{T}, Ds/32], m![Ds%32]>().fetch_table_lookup::<bf16>()
        .collect::<m![{T}, Ds/16], m![Ds%16]>().commit_trim::<m![Ds%16]>().commit()
}

fn encode<C: M, S: M, T: M>(device: &mut Device,
    x: &DmTensor<bf16, Chip, C, S, m![{T}, Ds]>,
) -> DmTensor<f8e4m3, Chip, C, S, m![{T}, Ds]> {
    device.main.begin(x.view())
        .fetch::<m![{T}, Ds/16], m![Ds%16]>().fetch_cast::<f32>()
        .collect::<m![{T}, Ds/8], m![Ds%8]>()
        .cast::<f8e4m3, m![Ds%8#32]>().commit_trim::<m![Ds%8]>().commit()
}

#[device(chip=1)]
pub fn b16(device: &mut Device,
    input: &HbmTensor<f8e4m3, Chip, m![B,Ns,Ds]>,
    decoded: &mut HbmTensor<bf16, Chip, m![B,Ns,Ds]>,
    recoded: &mut HbmTensor<f8e4m3, Chip, m![B,Ns,Ds]>,
) {
    let x: DmTensor<f8e4m3, Chip, m![1#2], m![B,Ns,1#2], m![Ds]> = input.to_dm(&mut device.tdma);
    let y = decode::<_,_,m![]>(device, &x);
    let z = encode::<_,_,m![]>(device, &y);
    y.view().to_hbm_view(&mut device.tdma, decoded.view_mut());
    z.view().to_hbm_view(&mut device.tdma, recoded.view_mut());
}

#[device(chip=1)]
pub fn r64(device: &mut Device,
    input: &HbmTensor<f8e4m3, Chip, m![R64,Ns,Ds]>,
    decoded: &mut HbmTensor<bf16, Chip, m![R64,Ns,Ds]>,
    recoded: &mut HbmTensor<f8e4m3, Chip, m![R64,Ns,Ds]>,
) {
    let x: DmTensor<f8e4m3, Chip, m![R64/32], m![R64/2%16,Ns,1#2], m![R64%2,Ds]> = input.to_dm(&mut device.tdma);
    let y = decode::<_,_,m![R64%2]>(device, &x);
    let z = encode::<_,_,m![R64%2]>(device, &y);
    y.view().to_hbm_view(&mut device.tdma, decoded.view_mut());
    z.view().to_hbm_view(&mut device.tdma, recoded.view_mut());
}

#[device(chip=1)]
pub fn p128(device: &mut Device,
    input: &HbmTensor<f8e4m3, Chip, m![P128,Ns,Ds]>,
    decoded: &mut HbmTensor<bf16, Chip, m![P128,Ns,Ds]>,
    recoded: &mut HbmTensor<f8e4m3, Chip, m![P128,Ns,Ds]>,
) {
    let x: DmTensor<f8e4m3, Chip, m![P128/64], m![P128/4%16,Ns,1#2], m![P128%4,Ds]> = input.to_dm(&mut device.tdma);
    let y = decode::<_,_,m![P128%4]>(device, &x);
    let z = encode::<_,_,m![P128%4]>(device, &y);
    y.view().to_hbm_view(&mut device.tdma, decoded.view_mut());
    z.view().to_hbm_view(&mut device.tdma, recoded.view_mut());
}

#[device(chip=1)]
pub fn read_poison(device: &mut Device,
    input: &HbmTensor<f8e4m3, Chip, m![B,Ns,Ds]>,
    output: &mut HbmTensor<bf16, Chip, m![B,Ns,Ds]>,
) {
    let x: DmTensor<f8e4m3, Chip, m![1#2], m![B,Ns,1#2], m![Ds]> = input.to_dm(&mut device.tdma);
    decode::<_,_,m![]>(device, &x).view().to_hbm_view(&mut device.tdma, output.view_mut());
}
