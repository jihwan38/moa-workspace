//! Shared physically blocked activation HBM layouts for P512 prompt projections.
//! Each cluster owns256 token rows. A64-row/K960 tile is61,440 contiguous FP8 bytes.
use furiosa_opt_std::prelude::*;
use crate::axes::{P512,H,L};
pub type P512H960=m![P512/256,H/960,P512%256,H%960];
pub type P512L960=m![P512/256,L/960,P512%256,L%960];
/// One64-row/L240 output tile is30,720 contiguous BF16 bytes.
pub type P512L240=m![P512/256,L/240,P512%256,L%240];
