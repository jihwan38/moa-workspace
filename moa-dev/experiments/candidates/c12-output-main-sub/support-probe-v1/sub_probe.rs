use furiosa_opt_std::prelude::*;
use crate::Chip;
axes![Clusters=2, Groups=16, Columns=16, Rows=60];

#[device(chip=1)]
pub fn main_reduce(device: &mut Device, input: &HbmTensor<f32, Chip, m![Clusters, Groups, Columns, Rows, 1 # 8]>) -> HbmTensor<f32, Chip, m![Clusters, Groups, Rows, 1 # 8]> {
    let x: DmTensor<f32, Chip, m![Clusters], m![Groups, Columns], m![Rows, 1 # 8]> = input.to_dm(&mut device.tdma);
    let y: DmTensor<f32, Chip, m![Clusters], m![Groups, 1 # 16], m![Rows, 1 # 8]> = device.main.begin(x.view())
        .fetch::<m![Rows], m![1 # 8]>().collect::<m![Rows], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![Groups, 1 # 16], m![Rows]>(InterSliceReduceOpF32::Add)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    y.to_hbm(&mut device.tdma)
}

#[device(chip=1)]
pub fn sub_reduce(device: &mut Device, input: &HbmTensor<f32, Chip, m![Clusters, Groups, Columns, Rows, 1 # 8]>) -> HbmTensor<f32, Chip, m![Clusters, Groups, Rows, 1 # 8]> {
    let x: DmTensor<f32, Chip, m![Clusters], m![Groups, Columns], m![Rows, 1 # 8]> = input.to_dm(&mut device.tdma);
    let y: DmTensor<f32, Chip, m![Clusters], m![Groups, 1 # 16], m![Rows, 1 # 8]> = device.sub.begin(x.view())
        .fetch::<m![Rows], m![1 # 8]>().collect::<m![Rows], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![Groups, 1 # 16], m![Rows]>(InterSliceReduceOpF32::Add)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    y.to_hbm(&mut device.tdma)
}

#[device(chip=1)]
pub fn sub_transpose(device: &mut Device, input: &HbmTensor<f32, Chip, m![Clusters, Groups, Columns, Rows, 1 # 8]>) -> HbmTensor<f32, Chip, m![Clusters, Groups, Columns, Rows]> {
    let x: DmTensor<f32, Chip, m![Clusters], m![Groups, Columns], m![Rows, 1 # 8]> = input.to_dm(&mut device.tdma);
    let y: DmTensor<f32, Chip, m![Clusters], m![Groups, Columns], m![Rows]> = device.sub.begin(x.view())
        .fetch::<m![Rows], m![1 # 8]>().collect::<m![Rows], m![1 # 8]>()
        .transpose::<m![Rows / 2], m![Rows % 2 # 8]>()
        .commit_trim::<m![Rows % 2]>().commit();
    y.to_hbm(&mut device.tdma)
}
