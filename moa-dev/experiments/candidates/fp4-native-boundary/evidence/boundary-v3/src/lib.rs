#![feature(register_tool)]
#![register_tool(furiosa_opt)]
use furiosa_opt_std::prelude::*;
pub type Chip = m![1];
pub mod axes;
pub mod probe;
