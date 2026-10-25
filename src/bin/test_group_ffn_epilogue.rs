#[path="support/npu_profile.rs"] mod npu_profile;
#[tokio::main]
async fn main()->Result<(),Box<dyn std::error::Error>> {
    let collector=npu_profile::Collector::default();
    tracing::subscriber::set_global_default(collector.clone())?;
    furiosa_opt_gemma4::host::group_ffn_epilogue_diagnostic::run(|on|collector.enable(on)).await
}
