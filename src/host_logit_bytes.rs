//! Exact host decoding of packed BF16 logits; no SDK per-element materialization.
use crate::axes::{R64,W};
use furiosa_opt_std::prelude::AxisName;
#[inline(always)]
pub fn bf16_bits_to_f32(bits:u16)->f32 {
    // Match half::bf16::to_f32: preserve sign/payload and quiet signaling NaNs.
    let bits=if bits&0x7fff>0x7f80 {bits|0x0040}else{bits};
    f32::from_bits((bits as u32)<<16)
}
pub fn decode_bf16_le(bytes:&[u8])->Vec<f32> {
    assert_eq!(bytes.len()%2,0);
    bytes.chunks_exact(2).map(|b|bf16_bits_to_f32(u16::from_le_bytes([b[0],b[1]]))).collect()
}
/// Restore live request rows from BlockedLogits physical wire order.
pub fn decode_blocked_r64(bytes:&[u8],live:usize,workers:usize)->Vec<Vec<f32>> {
    assert_eq!(bytes.len(),R64::SIZE*W::SIZE*2);
    assert!(live<=R64::SIZE);
    if live==0{return Vec::new();}
    let workers=workers.clamp(1,8).min(live);
    std::thread::scope(|scope| {
        let mut jobs=Vec::new();
        for worker in 0..workers {
            let first=worker*live/workers;let end=(worker+1)*live/workers;
            jobs.push(scope.spawn(move || {
                let mut rows:Vec<Vec<f32>>=(first..end).map(|_|Vec::with_capacity(W::SIZE)).collect();
                for block in 0..W::SIZE/32 {
                    for (local,out) in rows.iter_mut().enumerate() {
                        let base=(block*R64::SIZE+first+local)*64;
                        out.extend(bytes[base..base+64].chunks_exact(2).map(|b|bf16_bits_to_f32(u16::from_le_bytes([b[0],b[1]]))));
                    }
                }
                rows
            }));
        }
        jobs.into_iter().flat_map(|j|j.join().expect("logit conversion worker panicked")).collect()
    })
}
