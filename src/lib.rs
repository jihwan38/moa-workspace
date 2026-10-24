//! Gemma-4-12B-it on the Furiosa NPU: text, vision, and audio, targeting the Tensor
//! Contraction Processor via the `furiosa-opt-std` mapping-expression DSL. See the
//! crate's README for the device/host split and directory layout.

#![expect(clippy::type_complexity)]
#![feature(register_tool)]
#![register_tool(furiosa_opt)]

use furiosa_opt_std::prelude::*;

pub mod api;
pub mod axes;
pub mod host;
pub mod ops;
pub mod ops_batch_io;
pub mod ops_batch;
pub mod ops_batch_broadcast;
pub mod ops_batch_clusters;
pub mod ops_native8;
pub mod ops_native8_wide;
pub mod ops_batch_predecoded;
pub mod ops_batch_wide;
pub mod ops_batch_kpart;
pub mod ops_native_fp8;
pub mod ops_batch_down960;
pub mod ops_native_fp8_b16;
pub mod ops_native_fp8_b16_blocked;
pub mod ops_native_fp8_blocked;
pub mod ops_geglu_direct;
pub mod ops_batch_kpart16;
pub mod ops_audio;
pub mod ops_prefill;
pub mod ops_prefill_tiles;
pub mod ops_attention;
pub mod ops_decode;
pub mod ops_head_batch;
pub mod ops_norm_batch;
pub mod ops_request_attention;
pub mod ops_sliding_batch;
pub mod ops_vision;

pub(crate) mod device;
pub(crate) mod device_e154;

pub type Chip = m![1];

pub const LAYERS: usize = 48;
pub const EMBED_SCALE: f32 = 61.967_734;
pub const LOGIT_SOFTCAP: f32 = 30.0;
pub const EPS: f32 = 1e-6;

pub mod ops_request_attention_clusters;

pub mod ops_request_attention_rings;

pub mod ops_request_full_clusters;

pub mod ops_norm_wide;

pub mod ops_quant_native8;

pub mod ops_batch_prepack8;
pub mod ops_sliding_prompt_clusters;

pub mod ops_head_wide_tile;


pub mod ops_prefill_tiles_reference;

pub mod ops_native_fp8_partition;
pub mod ops_sliding_wide;


pub mod ops_head_wide;
pub mod ops_full_wide;
pub mod ops_commit_wide;

pub mod ops_r64_io;

pub mod ops_pool_views;

pub mod ops_cache_views;

pub mod ops_sliding_prompt_views;

pub mod ops_attention_views;

pub mod ops_prefill_cache_views;

pub mod ops_pool_offsets;

pub mod ops_cache_pools;

pub mod ops_sliding_prompt_pool;

pub mod ops_attention_pool;

pub mod ops_commit_r64;

pub mod ops_request_sliding_b16_rc64;

pub mod ops_norm_r64;

pub mod ops_native_fp8_r64;

pub mod ops_geglu_direct_r64;

pub mod ops_batch_r64;

pub mod ops_request_sliding_r64;

pub mod ops_request_full_r64;

pub mod ops_head_r64;

pub mod ops_sliding_wide_pool;

pub mod ops_full_wide_pool;

pub mod ops_commit_wide_pool;
pub mod ops_r64_io_bandoutput;

pub mod ops_prompt_pair_io;

pub mod ops_native_fp8_p256;
pub mod ops_geglu_direct_p256;
pub mod ops_zero_pool;

pub mod ops_prompt_pair_io_rows;

pub mod ops_prepack_alloc;
