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
pub mod ops_batch;
pub mod ops_batch_predecoded;
pub mod ops_batch_wide;
pub mod ops_batch_kpart;
pub mod ops_batch_down960;
pub mod ops_batch_native4;
pub mod ops_batch_prepack8;
pub mod ops_native_fp8;
pub mod ops_native_fp8_b16;
pub mod ops_native_fp8_b16_blocked;
pub mod ops_native_fp8_blocked;
pub mod ops_geglu_direct;
pub mod ops_batch_r64;
pub mod ops_output_r64_direct;
pub mod ops_output_r64_direct16;
pub mod ops_output_r64_parallel;
pub mod ops_output_r64_parallel_gather;
pub mod ops_output_p128_parallel_gather;
pub mod ops_qkv_r64_blocked;
pub mod ops_qkv_r64_trf32;
pub mod ops_qkv_p128_trf32;
pub mod ops_geglu_direct_r64;
pub mod ops_native_fp8_r64;
pub mod ops_native_fp8_r64_local_control;
pub mod ops_native_fp8_r64_split;
pub mod ops_native_fp8_r64_split_explicit;
pub mod ops_native_fp8_r64_split_switch;
pub mod ops_native_fp8_r64_split_switch_direct;
pub mod ops_native_fp8_r64_split_partition;
pub mod ops_native_fp8_r64_staged;
pub mod ops_native_fp8_p256;
pub mod ops_native_fp8_p512;
pub mod ops_native_fp8_p512_cluster;
pub mod ops_native_fp8_p512_fused;
pub mod ops_native_fp8_trf64;
pub mod ops_geglu_direct_p256;
pub mod ops_batch_kpart16;
pub mod ops_batch_kpart_bf16;
pub mod ops_batch_clusters;
pub mod ops_batch_cluster_bf16;
pub mod ops_audio;
pub mod ops_vision;

pub(crate) mod device;

pub type Chip = m![1];

pub const LAYERS: usize = 48;
pub const EMBED_SCALE: f32 = 61.967_734;
pub const LOGIT_SOFTCAP: f32 = 30.0;
pub const EPS: f32 = 1e-6;
