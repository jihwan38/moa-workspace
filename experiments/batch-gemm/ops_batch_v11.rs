//! Prompt-row matrix products. All rows use the provided model's actual weights.
//!
//! B16 is executed as two B8 TRF lane tiles: H3840 x 8 x BF16 fits the full
//! TRF, whereas H3840 x 16 does not. The decoded weight tile is shared by both.

use furiosa_opt_std::prelude::*;

use crate::axes::{B, H, Qs};
use crate::Chip;

type Cluster = m![1 # 2];
type QRows = m![Qs / 16];

/// Raw sliding-query projection, matching the baseline's BF16 dot boundary
/// followed by per-output-channel scaling. Input must already be RMS-normalized.
#[device(chip = 1)]
pub fn project_query_batch(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![B, H]>,
    weight: &HbmTensor<bf16, Chip, m![Qs, H]>,
    scale: &HbmTensor<bf16, Chip, m![Qs]>,
    output: &mut HbmTensor<bf16, Chip, m![Qs, B]>,
) {
    let input: DmTensor<bf16, Chip, Cluster, QRows, m![B, H]> = input.to_dm(&mut device.tdma);
    let weight: DmTensor<bf16, Chip, Cluster, QRows, m![Qs % 16, H]> = weight.to_dm(&mut device.tdma);
    let mut result: DmTensor<bf16, Chip, Cluster, QRows, m![Qs % 16, B]> = DmTensor::new();
    for chunk in 0..2 {
        let x: TrfTensor<bf16, Chip, Cluster, QRows, m![B = 8], m![H]> = device.sub
            .begin(input.view().tile::<m![B], 8, m![B = 8 # 16, H]>(chunk * 8))
            .fetch::<m![B = 8, H / 16], m![H % 16]>()
            .collect::<m![B = 8, H / 16], m![H % 16]>()
            .to_trf();
        device.main
            .begin(weight.view())
            .fetch::<m![Qs % 16, H / 16], m![H % 16]>()
            .collect::<m![Qs % 16, H / 16], m![H % 16]>()
            .contract_outer::<m![Qs % 16, H / 32], m![H % 32], _, _, _>(&x)
            .contract_packet::<m![1]>()
            .contract_time::<m![Qs % 16]>()
            .contract_lane::<m![Qs % 16], m![B = 8]>(LaneMode::Interleaved)
            .cast::<bf16, m![B = 8 # 16]>()
            .commit_trim::<m![B = 8]>()
.commit_view(result.view_mut().tile::<m![B], 8, m![Qs % 16, B = 8 #{!} 16]>(chunk * 8));
    }
    result.view().to_hbm_view(&mut device.tdma, output.view_mut());
}
