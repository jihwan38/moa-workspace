//! Observe the current four-slice F32 Add contract, without changing serving.
use furiosa_opt_std::prelude::*;
use crate::Chip;
axes![N64=64, K4=4];
pub type Inputs=m![N64,K4,1#8];
pub type Outputs=m![N64,1#8];
type Cluster=m![1#2];
type Parts=m![N64,K4];
type Rows=m![N64,1#4];
#[device(chip=1)]
pub fn sum_four(device:&mut Device,input:&HbmTensor<f32,Chip,Inputs>,output:&mut HbmTensor<f32,Chip,Outputs>) {
    let input:DmTensor<f32,Chip,Cluster,Parts,m![1#8]>=input.to_dm(&mut device.tdma);
    let sum:DmTensor<f32,Chip,Cluster,Rows,m![1#8]>=device.main.begin(input.view())
        .fetch::<m![1],m![1#8]>().collect::<m![1],m![1#8]>()
        .vector_init().vector_inter_slice_reduce::<Rows,m![1]>(InterSliceReduceOpF32::Add)
        .vector_final().commit_trim::<m![1#8]>().commit();
    sum.view().to_hbm_view(&mut device.tdma,output.view_mut());
}
