//! Frozen E154 scalar helpers; batched kernels use the upstream device namespace.
pub(crate) use crate::device::{full, layout};
pub(crate) mod shared;
pub(crate) mod sliding;
