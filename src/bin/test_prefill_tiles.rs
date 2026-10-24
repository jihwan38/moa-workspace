use furiosa_opt_gemma4::{Chip, axes::*, ops_prefill_tiles as op};
use furiosa_opt_std::prelude::*;
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut device = Device::new(op::load_hidden_tile.topology())?;
    let values: Vec<_> = (0..P128::SIZE * H::SIZE)
        .map(|i| bf16::from_f32(((i * 19 % 127) as f32) - 63.0))
        .collect();
    let input: HbmTensor<bf16, Chip, m![P128, H]> =
        HostTensor::<bf16, m![P128, H]>::from_vec(values.clone())
            .to_hbm(&mut device.pdma)
            .await?;
    let mut tile: HbmTensor<bf16, Chip, m![B, H]> = HostTensor::<bf16, m![B, H]>::zero()
        .to_hbm(&mut device.pdma)
        .await?;
    let mut output: HbmTensor<bf16, Chip, m![P128, H]> = HostTensor::<bf16, m![P128, H]>::zero()
        .to_hbm(&mut device.pdma)
        .await?;
    for index in 0..P128::SIZE / B::SIZE {
        let index = HostTensor::<i32, m![1]>::from_vec([index as i32])
            .to_hbm(&mut device.pdma)
            .await?;
        launch(
            op::load_hidden_tile,
            (&mut device, &input, &index, &mut tile),
        )
        .await?;
        launch(
            op::store_hidden_tile,
            (&mut device, &tile, &index, &mut output),
        )
        .await?;
    }
    let actual = output
        .to_host::<m![P128, H]>(&mut device.pdma)
        .await?
        .into_vec();
    assert_eq!(values, actual);
    println!(
        "P128/B16 hidden roundtrip: {} exact BF16 values",
        actual.len()
    );
    let values: Vec<_> = (0..P128::SIZE * Qf::SIZE)
        .map(|i| bf16::from_f32(((i * 37 % 251) as f32) - 125.0))
        .collect();
    let input: HbmTensor<bf16, Chip, m![P128, Qf]> =
        HostTensor::<bf16, m![P128, Qf]>::from_vec(values.clone())
            .to_hbm(&mut device.pdma)
            .await?;
    let mut tile: HbmTensor<bf16, Chip, m![B, Qf]> = HostTensor::<bf16, m![B, Qf]>::zero()
        .to_hbm(&mut device.pdma)
        .await?;
    let mut output: HbmTensor<bf16, Chip, m![P128, Qf]> = HostTensor::<bf16, m![P128, Qf]>::zero()
        .to_hbm(&mut device.pdma)
        .await?;
    for index in 0..P128::SIZE / B::SIZE {
        let index = HostTensor::<i32, m![1]>::from_vec([index as i32])
            .to_hbm(&mut device.pdma)
            .await?;
        launch(
            op::load_query_full_tile,
            (&mut device, &input, &index, &mut tile),
        )
        .await?;
        launch(
            op::store_query_full_tile,
            (&mut device, &tile, &index, &mut output),
        )
        .await?;
    }
    let actual = output
        .to_host::<m![P128, Qf]>(&mut device.pdma)
        .await?
        .into_vec();
    assert_eq!(values, actual);
    println!(
        "P128/B16 full-query roundtrip: {} exact BF16 values",
        actual.len()
    );
    Ok(())
}
