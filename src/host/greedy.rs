//! Exact device selection for retained B16/R64 rows; ambiguous IEEE cases use CPU logits.
use std::collections::HashSet;
use furiosa_opt_std::prelude::*;
use crate::{Chip, axes::{B,R64,W}, ops_greedy_r64::{BanWords,GreedyResult}, ops_r64_io_blocked::BlockedLogits};
type Result<T> = std::result::Result<T,Box<dyn std::error::Error>>;

pub struct GreedySelector {
    negative_ids: HbmTensor<i32,Chip,m![W]>,
    bit_positions: HbmTensor<i32,Chip,m![W%32]>,
    result: HbmTensor<i32,Chip,GreedyResult>,
    result_b16: HbmTensor<i32,Chip,crate::ops_greedy_b16::GreedyResult>,
    fallback: super::model::Logits,
}
fn pack_bans<const ROWS: usize>(bans: &[Option<&HashSet<usize>>]) -> Vec<u8> {
    assert!(bans.len()<=ROWS);
    let mut bytes=vec![0u8;W::SIZE/32*ROWS*4];
    for (row,banned) in bans.iter().enumerate() {
        if let Some(banned)=banned {
            for &token in *banned {
                if token<W::SIZE {
                    let word=(token/32)*ROWS+row;
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
            result_b16:HostTensor::<i32,crate::ops_greedy_b16::GreedyResult>::zero().to_hbm(&mut device.pdma).await?,
            fallback:HostTensor::<bf16,m![W]>::zero().to_hbm(&mut device.pdma).await?,
        })
    }
    /// A None policy means the caller does not request a selected token for that
    /// row. A None result with a policy means use the original CPU sampler.
    pub async fn select(&mut self,device:&mut Device,logits:&HbmTensor<bf16,Chip,BlockedLogits>,bans:&[Option<&HashSet<usize>>])->Result<Vec<Option<usize>>> {
        if bans.is_empty() || bans.len()>R64::SIZE {return Err("invalid greedy live width".into());}
        if bans.iter().all(Option::is_none) {return Ok(vec![None;bans.len()]);}
        let words=HostTensor::<i32,BanWords>::from_buf(pack_bans::<{R64::SIZE}>(bans)).to_hbm(&mut device.pdma).await?;
        launch(crate::ops_greedy_r64::greedy_argmax_r64,(device,logits,&words,&self.negative_ids,&self.bit_positions,&mut self.result)).await?;
        let host=self.result.to_host::<GreedyResult>(&mut device.pdma).await?.into_vec();
        bans.iter().enumerate().map(|(row,banned)|if banned.is_some(){selected(&host[row*8..row*8+8])}else{Ok(None)}).collect()
    }
    /// Canonical B16 logits remain borrowed until all requested tokens/fallback
    /// rows are consumed. None policies include stochastic and unused rows.
    pub async fn select_b16(&mut self,device:&mut Device,logits:&super::model::Batch16Logits,bans:&[Option<&HashSet<usize>>])->Result<Vec<Option<usize>>> {
        if bans.is_empty() || bans.len()>B::SIZE {return Err("invalid B16 greedy live width".into());}
        if bans.iter().all(Option::is_none) {return Ok(vec![None;bans.len()]);}
        let words=HostTensor::<i32,crate::ops_greedy_b16::BanWords>::from_buf(pack_bans::<{B::SIZE}>(bans)).to_hbm(&mut device.pdma).await?;
        launch(crate::ops_greedy_b16::greedy_argmax_b16,(device,logits,&words,&self.negative_ids,&self.bit_positions,&mut self.result_b16)).await?;
        let host=self.result_b16.to_host::<crate::ops_greedy_b16::GreedyResult>(&mut device.pdma).await?.into_vec();
        bans.iter().enumerate().map(|(row,banned)|if banned.is_some(){selected(&host[row*8..row*8+8])}else{Ok(None)}).collect()
    }
    /// Extract only a live stochastic/exceptional row; preserve original BF16
    /// values and the existing CPU conversion/sampler, including all-banned policy.
    pub async fn read_fallback_b16(&mut self,device:&mut Device,logits:&super::model::Batch16Logits,row:usize)->Result<Vec<f32>> {
        if row>=B::SIZE {return Err("B16 fallback row out of range".into());}
        let offset=HostTensor::<i32,m![1]>::from_vec([(row*W::SIZE*2) as i32]).to_hbm(&mut device.pdma).await?;
        launch(crate::ops_batch_io::extract_logits_row,(device,logits,&offset,&mut self.fallback)).await?;
        super::generate::read_logits(device,&self.fallback).await
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
  let bytes=pack_bans::<{R64::SIZE}>(&[Some(&a),None,Some(&b)]);
  for row in 0..R64::SIZE {for token in 0..W::SIZE {
   let actual=bytes[((token/32)*R64::SIZE+row)*4+(token%32)/8]&(1<<(token%8))!=0;
   let expected=match row {0=>a.contains(&token),2=>b.contains(&token),_=>false};assert_eq!(actual,expected);
  }}
 }
 #[test] fn b16_bans_keep_cluster_boundary_rows_and_ignore_out_of_range_ids() {
  let left:HashSet<_>=[0,31,32,8191,8192,W::SIZE-1,W::SIZE,usize::MAX].into_iter().collect();
  let right:HashSet<_>=[1,17,63,64].into_iter().collect();
  let mut policies=vec![None;B::SIZE];policies[7]=Some(&left);policies[8]=Some(&right);policies[15]=Some(&left);
  let bytes=pack_bans::<{B::SIZE}>(&policies);
  for row in 0..B::SIZE {for token in 0..W::SIZE {
   let got=bytes[((token/32)*B::SIZE+row)*4+(token%32)/8]&(1<<(token%8))!=0;
   assert_eq!(got,policies[row].is_some_and(|b|b.contains(&token)),"row={row} token={token}");
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
