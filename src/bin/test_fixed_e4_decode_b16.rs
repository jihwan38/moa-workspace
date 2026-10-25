#[path = "support/npu_profile.rs"]
mod npu_profile;
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let profile = npu_profile::Collector::default();
    tracing::subscriber::set_global_default(profile.clone())?;
    furiosa_opt_gemma4::host::fixed_e4_decode_b16_diagnostic::run(|on| profile.enable(on)).await
}
