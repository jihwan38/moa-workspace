//! Every raw BF16 bit across all four quarters and changed request ordering.
use furiosa_opt_gemma4::{Chip, axes::*, ops_prompt_group_io as op};
use furiosa_opt_std::prelude::*;
async fn read<E: M>(d: &mut Device, x: &HbmTensor<bf16, Chip, E>) -> Vec<u8> {
    let h: HostTensor<bf16, E> = x.to_host(&mut d.pdma).await.unwrap();
    h.into_inner().into_buf()
}
#[tokio::main]
async fn main() {
    let mut d = Device::new(op::pack_hidden_group.topology()).unwrap();
    let patterns: Vec<Vec<u8>> = (0..4)
        .map(|r| {
            (0..P128::SIZE * H::SIZE)
                .flat_map(|i| {
                    (i as u16)
                        .wrapping_mul(73 + r * 32)
                        .wrapping_add(0x1234 * r)
                        .to_le_bytes()
                })
                .collect()
        })
        .collect();
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut offsets = Vec::new();
    for (r, p) in patterns.iter().enumerate() {
        inputs.push(
            HostTensor::<bf16, m![P128, H]>::from_buf(p.clone())
                .to_hbm(&mut d.pdma)
                .await
                .unwrap(),
        );
        outputs.push(
            HostTensor::<bf16, m![P128, H]>::zero()
                .to_hbm(&mut d.pdma)
                .await
                .unwrap(),
        );
        offsets.push(
            HostTensor::<i32, m![1]>::from_vec([(r * P128::SIZE * H::SIZE * 2) as i32])
                .to_hbm(&mut d.pdma)
                .await
                .unwrap(),
        );
    }
    let mut joined = HostTensor::<bf16, m![P512, H]>::zero()
        .to_hbm(&mut d.pdma)
        .await
        .unwrap();
    for order in [[0, 1, 2, 3], [3, 1, 0, 2]] {
        let expected: Vec<_> = order.iter().flat_map(|&r| patterns[r].clone()).collect();
        launch(
            op::pack_hidden_group,
            (
                &mut d,
                &inputs[order[0]],
                &inputs[order[1]],
                &inputs[order[2]],
                &inputs[order[3]],
                &offsets[0],
                &offsets[1],
                &offsets[2],
                &offsets[3],
                &mut joined,
            ),
        )
        .await
        .unwrap();
        assert_eq!(read(&mut d, &joined).await, expected);
        let [a, b, c, e] = outputs.as_mut_slice() else {
            unreachable!()
        };
        launch(
            op::unpack_hidden_group,
            (
                &mut d,
                &joined,
                &offsets[0],
                &offsets[1],
                &offsets[2],
                &offsets[3],
                a,
                b,
                c,
                e,
            ),
        )
        .await
        .unwrap();
        for r in 0..4 {
            assert_eq!(read(&mut d, &outputs[r]).await, patterns[order[r]]);
        }
        println!(
            "PASS order={order:?} all{} raw BF16 values",
            P512::SIZE * H::SIZE
        );
    }
    let now = std::time::Instant::now();
    for _ in 0..20 {
        launch(
            op::pack_hidden_group,
            (
                &mut d,
                &inputs[0],
                &inputs[1],
                &inputs[2],
                &inputs[3],
                &offsets[0],
                &offsets[1],
                &offsets[2],
                &offsets[3],
                &mut joined,
            ),
        )
        .await
        .unwrap();
    }
    println!("pack4_us={:.3}", now.elapsed().as_secs_f64() * 1e6 / 20.0);
    let now = std::time::Instant::now();
    for _ in 0..20 {
        let [a, b, c, e] = outputs.as_mut_slice() else {
            unreachable!()
        };
        launch(
            op::unpack_hidden_group,
            (
                &mut d,
                &joined,
                &offsets[0],
                &offsets[1],
                &offsets[2],
                &offsets[3],
                a,
                b,
                c,
                e,
            ),
        )
        .await
        .unwrap();
    }
    println!("unpack4_us={:.3}", now.elapsed().as_secs_f64() * 1e6 / 20.0);
}
