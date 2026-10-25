#[path="support/npu_profile.rs"] mod npu_profile;
#[tokio::main]
async fn main()->Result<(),Box<dyn std::error::Error>>{
 let profile=npu_profile::Collector::default();let _guard=tracing::subscriber::set_default(profile.clone());
 furiosa_opt_gemma4::host::b16_f4_partialscale_diagnostic::run(&|v|profile.enable(v)).await
}
