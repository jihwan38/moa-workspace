use furiosa_opt_gemma4::{Chip, axes::*, ops_prompt_pair_io_rows as op};
use furiosa_opt_std::prelude::*;
async fn upload<Mp: M>(d: &mut Device, bytes: Vec<u8>) -> HbmTensor<bf16, Chip, Mp> {
    HostTensor::<bf16, Mp>::from_buf(bytes)
        .to_hbm(&mut d.pdma)
        .await
        .unwrap()
}
async fn read<Mp: M>(d: &mut Device, x: &HbmTensor<bf16, Chip, Mp>) -> Vec<u8> {
    let x: HostTensor<bf16, Mp> = x.to_host(&mut d.pdma).await.unwrap();
    x.into_vec()
        .iter()
        .flat_map(|x| x.to_bits().to_le_bytes())
        .collect()
}
#[tokio::main]
async fn main() {
    let mut d = Device::new(op::pack_hidden_pair.topology()).unwrap();
    let a: Vec<u8> = (0..P128::SIZE * H::SIZE)
        .flat_map(|i| ((i as u16).wrapping_mul(73)).to_le_bytes())
        .collect();
    let b: Vec<u8> = (0..P128::SIZE * H::SIZE)
        .flat_map(|i| ((i as u16).wrapping_mul(131) ^ 0x9271).to_le_bytes())
        .collect();
    let ah = upload::<m![P128, H]>(&mut d, a.clone()).await;
    let bh = upload::<m![P128, H]>(&mut d, b.clone()).await;
    let mut merged = upload::<m![P256, H]>(&mut d, vec![0x71; P256::SIZE * H::SIZE * 2]).await;
    let mut ao = upload::<m![P128, H]>(&mut d, vec![0; P128::SIZE * H::SIZE * 2]).await;
    let mut bo = upload::<m![P128, H]>(&mut d, vec![0; P128::SIZE * H::SIZE * 2]).await;
    let zero = HostTensor::<i32, m![1]>::from_vec([0])
        .to_hbm(&mut d.pdma)
        .await
        .unwrap();
    let second = HostTensor::<i32, m![1]>::from_vec([(P128::SIZE * H::SIZE * 2) as i32])
        .to_hbm(&mut d.pdma)
        .await
        .unwrap();
    for swap in [false, true] {
        let (first, last, expected) = if swap {
            (&bh, &ah, [b.clone(), a.clone()].concat())
        } else {
            (&ah, &bh, [a.clone(), b.clone()].concat())
        };
        launch(
            op::pack_hidden_pair,
            (&mut d, first, last, &zero, &second, &mut merged),
        )
        .await
        .unwrap();
        assert_eq!(read(&mut d, &merged).await, expected);
        launch(
            op::unpack_hidden_pair,
            (&mut d, &merged, &zero, &second, &mut ao, &mut bo),
        )
        .await
        .unwrap();
        assert_eq!(read(&mut d, &ao).await, expected[..a.len()]);
        assert_eq!(read(&mut d, &bo).await, expected[a.len()..]);
        println!(
            "PASS swap={swap} all{} BF16 values exact including half boundary",
            P256::SIZE * H::SIZE
        );
    }
    let now = std::time::Instant::now();
    for _ in 0..20 {
        launch(
            op::pack_hidden_pair,
            (&mut d, &ah, &bh, &zero, &second, &mut merged),
        )
        .await
        .unwrap();
    }
    println!("pack_us={:.3}", now.elapsed().as_secs_f64() * 1e6 / 20.0);
    let now = std::time::Instant::now();
    for _ in 0..20 {
        launch(
            op::unpack_hidden_pair,
            (&mut d, &merged, &zero, &second, &mut ao, &mut bo),
        )
        .await
        .unwrap();
    }
    println!("unpack_us={:.3}", now.elapsed().as_secs_f64() * 1e6 / 20.0);
    for control in [true, false, false, true] {
        let now = std::time::Instant::now();
        for _ in 0..20 {
            if control {
                launch(
                    furiosa_opt_gemma4::ops_prompt_pair_io::pack_hidden_pair,
                    (&mut d, &ah, &bh, &zero, &second, &mut merged),
                )
                .await
                .unwrap();
            } else {
                launch(
                    op::pack_hidden_pair,
                    (&mut d, &ah, &bh, &zero, &second, &mut merged),
                )
                .await
                .unwrap();
            }
        }
        println!(
            "ABBA control={control} pack_us={:.3}",
            now.elapsed().as_secs_f64() * 1e6 / 20.0
        );
        let now = std::time::Instant::now();
        for _ in 0..20 {
            if control {
                launch(
                    furiosa_opt_gemma4::ops_prompt_pair_io::unpack_hidden_pair,
                    (&mut d, &merged, &zero, &second, &mut ao, &mut bo),
                )
                .await
                .unwrap();
            } else {
                launch(
                    op::unpack_hidden_pair,
                    (&mut d, &merged, &zero, &second, &mut ao, &mut bo),
                )
                .await
                .unwrap();
            }
        }
        println!(
            "ABBA control={control} unpack_us={:.3}",
            now.elapsed().as_secs_f64() * 1e6 / 20.0
        );
    }
}
