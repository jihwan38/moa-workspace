//! Prompt-only fixed-scale E4 pool boundaries. Current chunks remain rounded BF16.
use crate::{Chip, axes::Ds};
use furiosa_opt_std::prelude::*;

pub(crate) fn decode_rows<C: M, S: M, T: M>(device: &mut Device,
    code: &DmTensor<f8e4m3, Chip, C, S, m![{T}, Ds]>)
    -> DmTensor<bf16, Chip, C, S, m![{T}, Ds]> {
    device.main.begin(code.view())
        .fetch::<m![{T}, Ds/32], m![Ds%32]>().fetch_table_lookup::<bf16>()
        .collect::<m![{T}, Ds/16], m![Ds%16]>()
        .commit_trim::<m![Ds%16]>().commit()
}

pub(crate) fn encode_rows<C: M, S: M, T: M>(device: &mut Device,
    rounded: &DmTensor<bf16, Chip, C, S, m![{T}, Ds]>)
    -> DmTensor<f8e4m3, Chip, C, S, m![{T}, Ds]> {
    device.main.begin(rounded.view())
        .fetch::<m![{T}, Ds/16], m![Ds%16]>().fetch_cast::<f32>()
        .collect::<m![{T}, Ds/8], m![Ds%8]>()
        .cast::<f8e4m3, m![Ds%8#32]>()
        .commit_trim::<m![Ds%8]>().commit()
}
