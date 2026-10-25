//! Opt-in per-operation device cycles. Disabled during normal correctness/timing.
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
#[derive(Clone, Default)]
pub struct Collector {
    enabled: Arc<AtomicBool>,
    next: Arc<AtomicU64>,
}
impl Collector {
    pub fn enable(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }
}
#[derive(Default)]
struct Fields(serde_json::Map<String, serde_json::Value>);
impl tracing::field::Visit for Fields {
    fn record_u64(&mut self, f: &tracing::field::Field, v: u64) {
        self.0.insert(f.name().into(), v.into());
    }
    fn record_str(&mut self, f: &tracing::field::Field, v: &str) {
        self.0.insert(f.name().into(), v.into());
    }
    fn record_debug(&mut self, f: &tracing::field::Field, v: &dyn std::fmt::Debug) {
        self.0.insert(f.name().into(), format!("{v:?}").into());
    }
}
impl tracing::Subscriber for Collector {
    fn register_callsite(
        &self,
        m: &'static tracing::Metadata<'static>,
    ) -> tracing::subscriber::Interest {
        if m.target() == "span::npu" {
            tracing::subscriber::Interest::sometimes()
        } else {
            tracing::subscriber::Interest::never()
        }
    }
    fn enabled(&self, m: &tracing::Metadata<'_>) -> bool {
        m.target() == "span::npu" && self.enabled.load(Ordering::Relaxed)
    }
    fn new_span(&self, a: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        if self.enabled(a.metadata()) {
            let mut f = Fields::default();
            a.record(&mut f);
            println!("NPU_SPAN {}", serde_json::Value::Object(f.0));
        }
        tracing::span::Id::from_u64(self.next.fetch_add(1, Ordering::Relaxed) + 1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, _: &tracing::Event<'_>) {}
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}
