//! Exact device selection for greedy R64 rows; ambiguous IEEE cases use CPU logits.
use std::collections::HashSet;
use furiosa_opt_std::prelude::*;
use crate::{Chip, axes::{R64,W}, ops_greedy_r64::{BanWords,GreedyResult}, ops_r64_io_blocked::BlockedLogits};
type Result<T> = std::result::Result<T,Box<dyn std::error::Error>>;

pub struct GreedySelector {
    negative_ids: HbmTensor<i32,Chip,m![W]>,
    bit_positions: HbmTensor<i32,Chip,m![W%32]>,
    result: HbmTensor<i32,Chip,GreedyResult>,
    fallback: super::model::Logits,
}
fn pack_bans(bans: &[Option<&HashSet<usize>>]) -> Vec<u8> {
    assert!(bans.len()<=R64::SIZE);
    let mut bytes=vec![0u8;W::SIZE/32*R64::SIZE*4];
    for (row,banned) in bans.iter().enumerate() {
        if let Some(banned)=banned {
            for &token in *banned {
                if token<W::SIZE {
                    let word=(token/32)*R64::SIZE+row;
                    bytes[word*4+(token%32)/8] |= 1 << (token%8);
                }
            }
        }
    }
    bytes
}
fn selected(fields: &[i32]) -> Result<Option<usize>> {
    if fields.len()!=8 {return Err("invalid greedy result width".into());}
    let all_banned=fields[0]==-65536;
    let maximum=fields[if all_banned {4}else{0}];
    // Device BF16 casts flush subnormals. Only a zero maximum can be ambiguous;
    // normal maxima outrank the flushed values. Nonfinite total_cmp stays on CPU.
    if fields[3]>=0x7f80 || maximum==0 || maximum== -1 {return Ok(None);}
    let negative=fields[if all_banned {2}else{1}];
    let token=negative.checked_neg().ok_or("greedy token overflow")?;
    if !(0..W::SIZE as i32).contains(&token) {return Err("greedy token out of range".into());}
    Ok(Some(token as usize))
}
impl GreedySelector {
    pub async fn new(device:&mut Device)->Result<Self> {
        Ok(Self {
            negative_ids:HostTensor::<i32,m![W]>::from_vec((0..W::SIZE).map(|id|-(id as i32)).collect::<Vec<_>>()).to_hbm(&mut device.pdma).await?,
            bit_positions:HostTensor::<i32,m![W%32]>::from_vec((0..32).collect::<Vec<i32>>()).to_hbm(&mut device.pdma).await?,
            result:HostTensor::<i32,GreedyResult>::zero().to_hbm(&mut device.pdma).await?,
            fallback:HostTensor::<bf16,m![W]>::zero().to_hbm(&mut device.pdma).await?,
        })
    }
    /// A None policy means the caller does not request a selected token for that
    /// row. A None result with a policy means use the original CPU sampler.
    pub async fn select(&mut self,device:&mut Device,logits:&HbmTensor<bf16,Chip,BlockedLogits>,bans:&[Option<&HashSet<usize>>])->Result<Vec<Option<usize>>> {
        if bans.is_empty() || bans.len()>R64::SIZE {return Err("invalid greedy live width".into());}
        if bans.iter().all(Option::is_none) {return Ok(vec![None;bans.len()]);}
        let words=HostTensor::<i32,BanWords>::from_buf(pack_bans(bans)).to_hbm(&mut device.pdma).await?;
        launch(crate::ops_greedy_r64::greedy_argmax_r64,(device,logits,&words,&self.negative_ids,&self.bit_positions,&mut self.result)).await?;
        let host=self.result.to_host::<GreedyResult>(&mut device.pdma).await?.into_vec();
        bans.iter().enumerate().map(|(row,banned)|if banned.is_some(){selected(&host[row*8..row*8+8])}else{Ok(None)}).collect()
    }
    pub async fn read_fallback(&mut self,device:&mut Device,logits:&HbmTensor<bf16,Chip,BlockedLogits>,row:usize)->Result<Vec<f32>> {
        if row>=R64::SIZE {return Err("fallback row out of range".into());}
        let offset=HostTensor::<i32,m![1]>::from_vec([(row*64) as i32]).to_hbm(&mut device.pdma).await?;
        launch(crate::ops_r64_io_blocked::extract_logits_row,(device,logits,&offset,&mut self.fallback)).await?;
        super::generate::read_logits(device,&self.fallback).await
    }
}
#[cfg(test)]
mod tests {
 use super::*;
 #[test] fn ban_wire_order_preserves_row_and_word_boundaries() {
  let a=[0,7,8,31,32,W::SIZE-1,W::SIZE].into_iter().collect();let b=[1,8192].into_iter().collect();
  let bytes=pack_bans(&[Some(&a),None,Some(&b)]);
  for row in 0..R64::SIZE {for token in 0..W::SIZE {
   let actual=bytes[((token/32)*R64::SIZE+row)*4+(token%32)/8]&(1<<(token%8))!=0;
   let expected=match row {0=>a.contains(&token),2=>b.contains(&token),_=>false};assert_eq!(actual,expected);
  }}
 }
 #[test] fn result_policy_preserves_normal_ties_and_all_banned_fallback() {
  assert_eq!(selected(&[100,-31,-99,100,100,0,0,0]).unwrap(),Some(31));
  assert_eq!(selected(&[-65536,0,-99,100,100,0,0,0]).unwrap(),Some(99));
  for maximum in [-1,0] {
   assert_eq!(selected(&[maximum,0,0,100,100,0,0,0]).unwrap(),None);
   assert_eq!(selected(&[-65536,0,0,100,maximum,0,0,0]).unwrap(),None);
  }
  assert_eq!(selected(&[100,0,0,0x7f80,100,0,0,0]).unwrap(),None);
  assert!(selected(&[100,i32::MIN,0,100,100,0,0,0]).is_err());
 }
}
