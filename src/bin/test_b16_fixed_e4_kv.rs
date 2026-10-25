#[path = "support/npu_profile.rs"]
mod npu_profile;
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let profile = npu_profile::Collector::default();
    tracing::subscriber::set_global_default(profile.clone())?;
    furiosa_opt_gemma4::host::b16_fixed_e4_kv_diagnostic::run(|on| profile.enable(on)).await
}
