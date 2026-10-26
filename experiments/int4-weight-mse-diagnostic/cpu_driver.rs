#[path="../../src/bin/support/int4_weight_mse.rs"] mod mse;
use std::io::{self,Read};
fn main(){
    mse::prepare();
    println!("DOMAIN {}",mse::provided_magnitudes().iter().map(u16::to_string).collect::<Vec<_>>().join(","));
    let mut input=String::new();io::stdin().read_to_string(&mut input).unwrap();let mut t=input.split_whitespace();
    let count:usize=t.next().unwrap().parse().unwrap();
    for i in 0..count{
        let scale=f32::from_bits(t.next().unwrap().parse().unwrap());
        let values:Vec<u16>=(0..mse::GROUP).map(|_|t.next().unwrap().parse().unwrap()).collect();
        let baseline:Vec<i8>=(0..mse::GROUP).map(|_|t.next().unwrap().parse().unwrap()).collect();
        let fit=mse::fit(&values,&baseline,scale).unwrap();
        let hex=mse::pack(&fit.codes).unwrap().iter().map(|b|format!("{b:02x}")).collect::<String>();
        println!("{i} {} {} {} {} {} {hex}",fit.selected,fit.scale.to_bits(),fit.baseline_sse,fit.selected_sse,fit.source_ss);
    }assert!(t.next().is_none());
}
