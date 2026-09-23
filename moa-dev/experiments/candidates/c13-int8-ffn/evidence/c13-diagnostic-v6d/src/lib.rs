#![feature(register_tool)]
#![register_tool(furiosa_opt)]
use furiosa_opt_std::prelude::*;
pub type Chip = m![1];
pub mod axes {
    use furiosa_opt_std::prelude::*;
    axes![Rows=4, K=3840, Term=2, H=3840, L=15360, IntTerm=2, IntBlockElt=16, QuantCopies=4];
}
pub mod integer_probe;
