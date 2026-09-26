#![feature(register_tool)]
#![register_tool(furiosa_opt)]
use furiosa_opt_std::prelude::*;
pub type Chip = m![1];
pub mod axes;
pub mod device {
    pub mod layout;
    pub mod mlp;
}
pub use device::mlp::validation_probe as stream_probe;
