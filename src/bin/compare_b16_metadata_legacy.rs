#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    furiosa_opt_gemma4::host::b16_metadata_diagnostic::run_legacy().await
}
