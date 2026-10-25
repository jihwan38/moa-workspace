//! Independent raw-bit oracle for explicit whole-pool offset read/write binding.
use furiosa_opt_gemma4::ops_pool_views::Rc64;
use furiosa_opt_gemma4::{Chip, axes::*, ops_pool_offsets as op, ops_request_attention::Fc};
use furiosa_opt_std::prelude::*;
fn pattern(index: usize, salt: usize) -> bf16 {
    bf16::from_bits((0x3e00 + (index.wrapping_mul(73).wrapping_add(salt * 19) % 1024)) as u16)
}
fn exact(label: &str, actual: &[bf16], expected: &[bf16]) {
    assert_eq!(actual.len(), expected.len());
    let mismatches = actual
        .iter()
        .zip(expected)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    println!("{label} mismatches={mismatches}/{}", actual.len());
    assert_eq!(
        mismatches, 0,
        "view read/write changed raw values or adjacent slots"
    );
}
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut device = Device::new(op::read_ring.topology())?;
    let ring_elements = Ts::SIZE * Ns::SIZE * Ds::SIZE;
    let mut expected: Vec<_> = (0..Rc64::SIZE * ring_elements)
        .map(|i| pattern(i, i / ring_elements))
        .collect();
    let mut rings: HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]> =
        HostTensor::<bf16, m![Rc64, Ts, Ns, Ds]>::from_vec(expected.clone())
            .to_hbm(&mut device.pdma)
            .await?;
    let mut output: HbmTensor<bf16, Chip, m![Ts, Ns, Ds]> =
        HostTensor::<bf16, m![Ts, Ns, Ds]>::zero()
            .to_hbm(&mut device.pdma)
            .await?;
    for slot in [0usize, 15, Rc64::SIZE - 1] {
        // Base is relative to the complete pool; the bound buffer never shifts.
        let offset: HbmTensor<i32, Chip, m![1]> =
            HostTensor::<i32, m![1]>::from_vec([(slot * ring_elements * 2) as i32])
                .to_hbm(&mut device.pdma)
                .await?;
        launch(op::read_ring, (&mut device, &rings, &offset, &mut output)).await?;
        exact(
            &format!("ring={slot} initial"),
            &output
                .to_host::<m![Ts, Ns, Ds]>(&mut device.pdma)
                .await?
                .into_vec(),
            &expected[slot * ring_elements..(slot + 1) * ring_elements],
        );
        let replacement: Vec<_> = (0..ring_elements)
            .map(|i| pattern(i, 1000 + slot))
            .collect();
        let input: HbmTensor<bf16, Chip, m![Ts, Ns, Ds]> =
            HostTensor::<bf16, m![Ts, Ns, Ds]>::from_vec(replacement.clone())
                .to_hbm(&mut device.pdma)
                .await?;
        launch(op::write_ring, (&mut device, &input, &offset, &mut rings)).await?;
        expected[slot * ring_elements..(slot + 1) * ring_elements].copy_from_slice(&replacement);
    }
    exact(
        "all physical rings including untouched neighbors",
        &rings
            .to_host::<m![Rc64, Ts, Ns, Ds]>(&mut device.pdma)
            .await?
            .into_vec(),
        &expected,
    );
    drop(rings);
    drop(expected);
    drop(output);
    let page_elements = Tf::SIZE * Df::SIZE;
    let mut expected: Vec<_> = (0..Fc::SIZE * page_elements)
        .map(|i| pattern(i, i / page_elements))
        .collect();
    let mut pages: HbmTensor<bf16, Chip, m![Fc, Tf, Df]> =
        HostTensor::<bf16, m![Fc, Tf, Df]>::from_vec(expected.clone())
            .to_hbm(&mut device.pdma)
            .await?;
    let mut output: HbmTensor<bf16, Chip, m![Tf, Df]> = HostTensor::<bf16, m![Tf, Df]>::zero()
        .to_hbm(&mut device.pdma)
        .await?;
    for page in [0usize, 127, 255] {
        let offset: HbmTensor<i32, Chip, m![1]> =
            HostTensor::<i32, m![1]>::from_vec([(page * page_elements * 2) as i32])
                .to_hbm(&mut device.pdma)
                .await?;
        launch(op::read_page, (&mut device, &pages, &offset, &mut output)).await?;
        exact(
            &format!("page={page} initial"),
            &output
                .to_host::<m![Tf, Df]>(&mut device.pdma)
                .await?
                .into_vec(),
            &expected[page * page_elements..(page + 1) * page_elements],
        );
        let replacement: Vec<_> = (0..page_elements)
            .map(|i| pattern(i, 2000 + page))
            .collect();
        let input: HbmTensor<bf16, Chip, m![Tf, Df]> =
            HostTensor::<bf16, m![Tf, Df]>::from_vec(replacement.clone())
                .to_hbm(&mut device.pdma)
                .await?;
        launch(op::write_page, (&mut device, &input, &offset, &mut pages)).await?;
        expected[page * page_elements..(page + 1) * page_elements].copy_from_slice(&replacement);
    }
    exact(
        "all256pages including untouched neighbors",
        &pages
            .to_host::<m![Fc, Tf, Df]>(&mut device.pdma)
            .await?
            .into_vec(),
        &expected,
    );
    Ok(())
}
