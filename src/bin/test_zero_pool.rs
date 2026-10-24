//! Complete raw-bit zero check and the actual canonical 80/16 allocation count.
use furiosa_opt_gemma4::{
    Chip, axes::*, ops_pool_views::Rc64, ops_request_attention::Fc, ops_zero_pool as op,
};
use furiosa_opt_std::prelude::*;
async fn check<Mp: M>(d: &mut Device, x: &HbmTensor<bf16, Chip, Mp>) {
    let h: HostTensor<bf16, Mp> = x.to_host(&mut d.pdma).await.unwrap();
    let v = h.into_vec();
    let wrong = v.iter().filter(|x| x.to_bits() != 0).count();
    println!("checked {} BF16 values, nonzero raw bits={wrong}", v.len());
    assert_eq!(wrong, 0);
}
#[tokio::main]
async fn main() {
    let mut d = Device::new(op::zero_sliding_pool.topology()).unwrap();
    let now = std::time::Instant::now();
    let s = launch(op::zero_sliding_pool, &mut d).await.unwrap();
    println!("first_sliding_ms={:.3}", now.elapsed().as_secs_f64() * 1e3);
    check::<m![Rc64, Ts, Ns, Ds]>(&mut d, &s).await;
    let now = std::time::Instant::now();
    let f = launch(op::zero_full_pool, &mut d).await.unwrap();
    println!("first_full_ms={:.3}", now.elapsed().as_secs_f64() * 1e3);
    check::<m![Fc, Tf, Df]>(&mut d, &f).await;
    drop((s, f));
    let now = std::time::Instant::now();
    let mut sliding = Vec::new();
    let mut full = Vec::new();
    for _ in 0..80 {
        sliding.push(launch(op::zero_sliding_pool, &mut d).await.unwrap());
    }
    for _ in 0..16 {
        full.push(launch(op::zero_full_pool, &mut d).await.unwrap());
    }
    println!(
        "canonical_22GiB_alloc_fill_ms={:.3}",
        now.elapsed().as_secs_f64() * 1e3
    );
    check::<m![Rc64, Ts, Ns, Ds]>(&mut d, sliding.last().unwrap()).await;
    check::<m![Fc, Tf, Df]>(&mut d, full.last().unwrap()).await;
    // Returned tensors from repeated launches must own distinct allocations.
    let ring_elements = Ts::SIZE * Ns::SIZE * Ds::SIZE;
    let ring_marker =
        HostTensor::<bf16, m![Ts, Ns, Ds]>::from_vec(vec![bf16::from_f32(1.0); ring_elements])
            .to_hbm(&mut d.pdma)
            .await
            .unwrap();
    let ring_offset =
        HostTensor::<i32, m![1]>::from_vec([((Rc64::SIZE - 1) * ring_elements * 2) as i32])
            .to_hbm(&mut d.pdma)
            .await
            .unwrap();
    launch(
        furiosa_opt_gemma4::ops_pool_offsets::write_ring,
        (&mut d, &ring_marker, &ring_offset, &mut sliding[0]),
    )
    .await
    .unwrap();
    let mut ring_read = HostTensor::<bf16, m![Ts, Ns, Ds]>::zero()
        .to_hbm(&mut d.pdma)
        .await
        .unwrap();
    launch(
        furiosa_opt_gemma4::ops_pool_offsets::read_ring,
        (&mut d, &sliding[0], &ring_offset, &mut ring_read),
    )
    .await
    .unwrap();
    let values: HostTensor<bf16, m![Ts, Ns, Ds]> = ring_read.to_host(&mut d.pdma).await.unwrap();
    assert!(
        values
            .into_vec()
            .iter()
            .all(|v| v.to_bits() == bf16::from_f32(1.0).to_bits())
    );
    check::<m![Rc64, Ts, Ns, Ds]>(&mut d, sliding.last().unwrap()).await;
    let page_elements = Tf::SIZE * Df::SIZE;
    let page_marker =
        HostTensor::<bf16, m![Tf, Df]>::from_vec(vec![bf16::from_f32(2.0); page_elements])
            .to_hbm(&mut d.pdma)
            .await
            .unwrap();
    let page_offset =
        HostTensor::<i32, m![1]>::from_vec([((Fc::SIZE - 1) * page_elements * 2) as i32])
            .to_hbm(&mut d.pdma)
            .await
            .unwrap();
    launch(
        furiosa_opt_gemma4::ops_pool_offsets::write_page,
        (&mut d, &page_marker, &page_offset, &mut full[0]),
    )
    .await
    .unwrap();
    let mut page_read = HostTensor::<bf16, m![Tf, Df]>::zero()
        .to_hbm(&mut d.pdma)
        .await
        .unwrap();
    launch(
        furiosa_opt_gemma4::ops_pool_offsets::read_page,
        (&mut d, &full[0], &page_offset, &mut page_read),
    )
    .await
    .unwrap();
    let values: HostTensor<bf16, m![Tf, Df]> = page_read.to_host(&mut d.pdma).await.unwrap();
    assert!(
        values
            .into_vec()
            .iter()
            .all(|v| v.to_bits() == bf16::from_f32(2.0).to_bits())
    );
    check::<m![Fc, Tf, Df]>(&mut d, full.last().unwrap()).await;
    println!(
        "PASS returned pools remain live with all22GiB allocated; distinct mutable allocations"
    );
}
