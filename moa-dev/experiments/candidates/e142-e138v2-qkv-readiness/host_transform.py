"""Original E104 diagnostic-host transform, unchanged; no kernel arithmetic edits."""
def host_source(source):
    replacements = {
        "#[derive(Clone, Copy)]\nstruct Span {": "#[derive(Clone)]\nstruct Span {\n    name: String,",
        "    cluster: Option<u64>,\n}": "    cluster: Option<u64>,\n    name: Option<String>,\n}",
        "    fn record_debug(&mut self, _field: &tracing::field::Field, _value: &dyn std::fmt::Debug) {}":
        '''    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "name" { self.name = Some(value.to_owned()); }
    }
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "name" { self.name = Some(format!("{value:?}").trim_matches('"').to_owned()); }
    }''',
        "push(Span { cluster, begin, end });": "push(Span { name: extractor.name.unwrap_or_default(), cluster, begin, end });",
        "    let mut failures = Vec::new();\n    for test in TESTS {":
        '''    let mut failures = Vec::new();
    let mut selected = 0usize;
    for test in TESTS {
        if std::env::var("DIAG_KERNEL").is_ok_and(|name| name != test.name) { continue; }
        selected += 1;''',
        "                collector.window_cycles()":
        '''                for span in collector.spans.lock().unwrap().iter() {
                    println!("DIAG_SPAN {}", serde_json::json!({
                        "kernel": test.name, "run": run, "name": span.name,
                        "cluster": span.cluster, "begin": span.begin, "end": span.end
                    }));
                }
                collector.window_cycles()''',
        'println!("all {} tests passed", TESTS.len());': 'println!("DIAG_CHECK_COMPLETE selected={selected} runs={RUNS} failures=0");',
    }
    for old, new in replacements.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    return source
