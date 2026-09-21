use super::*;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "moa-inspector-private-fixture-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::SeqCst)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn weight(&self, header: &Value, data: &[u8]) -> Vec<u8> {
        let raw = serde_json::to_vec(header).unwrap();
        let mut full = (raw.len() as u64).to_le_bytes().to_vec();
        full.extend(raw);
        full.extend(data);
        fs::write(self.0.join("model.safetensors"), &full).unwrap();
        full
    }
    fn metadata(&self) {
        for name in METADATA_FILES {
            fs::write(self.0.join(name), b"private fixture contents").unwrap();
        }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn tiny_contract(dtype: &str, strict: bool) -> Contract {
    let value = json!({"schema":"MOA_MODEL_LOADER_HEADER_CONTRACT_V1", "source_sha256":{},
        "tensor_count":1, "required_storage_bytes":4, "required_tensors":[{
            "name":"private_tensor_name", "dtype":dtype, "shape":[2], "storage_bytes":4,
            "loader_dtype_check":if strict {"strict"} else {"semantic_only"}, "scalar_shape_unchecked":false}]});
    parse_contract(&serde_json::to_vec(&value).unwrap()).unwrap()
}
fn tiny_header() -> Value {
    json!({"private_tensor_name":{"dtype":"BF16", "shape":[2], "data_offsets":[0,4]}})
}

#[test]
fn sha256_known_vectors_and_full_digest_including_header() {
    assert_eq!(
        digest(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        digest(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    let f = Fixture::new();
    f.metadata();
    let bytes = f.weight(&tiny_header(), &[1, 2, 3, 4]);
    let (report, exit) = build_report(Some(f.0.clone()), false, &tiny_contract("BF16", true));
    assert_eq!(exit, 0);
    assert_eq!(report["status"], "complete");
    assert_eq!(report["files"].as_array().unwrap().len(), 5);
    assert_eq!(report["files"][0]["full_file"]["sha256"], digest(&bytes));
    assert_eq!(
        report["files"][0]["header"]["sha256"],
        digest(&serde_json::to_vec(&tiny_header()).unwrap())
    );
    assert_eq!(
        report["files"][0]["header_contract"]["loader_contract_pass"],
        true
    );
    assert_eq!(report["files"][0]["file_unchanged_by_stat"], true);
    for entry in report["files"].as_array().unwrap().iter().skip(1) {
        assert_eq!(
            entry["full_file"]["sha256"],
            digest(b"private fixture contents")
        );
    }
    let output = report.to_string();
    assert!(!output.contains("private_tensor_name"));
    assert!(!output.contains("private fixture contents"));
    assert!(!output.contains(f.0.to_str().unwrap()));
}

#[test]
fn selector_order_and_direct_root_semantics() {
    let f = Fixture::new();
    let a = f.0.join("Gemma4-12B-it");
    let b = f.0.join("gemma-4-12b-it-NVFP4");
    fs::create_dir(&a).unwrap();
    fs::create_dir(&b).unwrap();
    fs::write(b.join("model.safetensors"), b"").unwrap();
    assert_eq!(select_directory(&f.0).unwrap().0, "gemma-4-12b-it-NVFP4");
    fs::write(a.join("model.safetensors"), b"").unwrap();
    assert_eq!(select_directory(&f.0).unwrap().0, "Gemma4-12B-it");
    fs::write(f.0.join("model.safetensors"), b"").unwrap();
    assert_eq!(select_directory(&f.0).unwrap().0, "direct");
    assert_eq!(
        Path::new("").join("model.safetensors"),
        Path::new("model.safetensors")
    );
}

#[test]
fn header_only_skips_weight_hash_but_hashes_metadata_and_reports_missing() {
    let f = Fixture::new();
    f.weight(&tiny_header(), &[0; 4]);
    let contract = tiny_contract("BF16", true);
    let (partial, exit) = build_report(Some(f.0.clone()), true, &contract);
    assert_eq!(exit, 0);
    assert_eq!(partial["status"], "partial");
    assert_eq!(partial["files"][0]["status"], "header_only");
    assert_eq!(partial["files"][0]["full_file"]["status"], "skipped");
    assert_eq!(partial["files"][0]["full_file"]["sha256"], Value::Null);
    assert_eq!(partial["files"][0]["full_file"]["bytes_hashed"], 0);
    assert_eq!(partial["files"][1]["status"], "missing");
    f.metadata();
    let (full, exit) = build_report(Some(f.0.clone()), true, &contract);
    assert_eq!(exit, 0);
    assert_eq!(full["status"], "header_only");
    assert_eq!(full["files"][1]["full_file"]["status"], "complete");
}

#[test]
fn deadlines_never_publish_partial_digest_and_short_read_is_error() {
    let mut bytes = io::Cursor::new(vec![1u8; 10]);
    let timed = hash_file(&mut bytes, 10, Some(Duration::ZERO)).unwrap();
    assert_eq!(timed["status"], "incomplete");
    assert_eq!(timed["sha256"], Value::Null);
    assert_eq!(timed["sha256_scope"], Value::Null);
    assert_eq!(timed["bytes_hashed"], 0);
    assert_eq!(
        hash_file(&mut bytes, 11, None).unwrap_err().code,
        "file_short_read"
    );
    struct Slow(io::Cursor<Vec<u8>>);
    impl Read for Slow {
        fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
            std::thread::sleep(Duration::from_millis(200));
            self.0.read(b)
        }
    }
    impl Seek for Slow {
        fn seek(&mut self, p: SeekFrom) -> io::Result<u64> {
            self.0.seek(p)
        }
    }
    let post = hash_file(
        &mut Slow(io::Cursor::new(vec![3])),
        1,
        Some(Duration::from_millis(100)),
    )
    .unwrap();
    assert_eq!(post["status"], "incomplete");
    assert_eq!(post["bytes_hashed"], 1);
    assert_eq!(post["sha256"], Value::Null); // Even after the last chunk, budget expiry withholds digest.
}

#[test]
fn modification_or_path_replacement_invalidates_all_identity_claims() {
    for replacement in [false, true] {
        let f = Fixture::new();
        let bytes = f.weight(&tiny_header(), &[0; 4]);
        let path = f.0.join("model.safetensors");
        let checked = fingerprint_with_hook(
            &f.0,
            "model.safetensors",
            true,
            false,
            &tiny_contract("BF16", true),
            || {
                if replacement {
                    let other = f.0.join("replacement");
                    fs::write(&other, &bytes).unwrap();
                    fs::rename(other, &path).unwrap();
                } else {
                    let file = fs::OpenOptions::new().write(true).open(&path).unwrap();
                    file.set_len(bytes.len() as u64 + 1).unwrap();
                }
            },
        );
        assert_eq!(checked["status"], "error");
        assert_eq!(checked["file_unchanged_by_stat"], false);
        assert_eq!(checked["header"]["sha256"], Value::Null);
        assert_eq!(checked["full_file"]["sha256"], Value::Null);
        assert_eq!(checked["header_contract"]["status"], "invalidated");
        assert_eq!(
            checked["header_contract"]["loader_contract_pass"],
            Value::Null
        );
    }
}

#[test]
fn malformed_header_bound_duplicates_nonstandard_numbers_and_metadata() {
    let f = Fixture::new();
    let contract = tiny_contract("BF16", true);
    for (prefix, tail, code) in [
        (
            0u64,
            b"".as_slice(),
            "safetensors_header_length_out_of_bounds",
        ),
        (
            MAX_HEADER_BYTES + 1,
            b"",
            "safetensors_header_length_out_of_bounds",
        ),
        (100, b"{}", "safetensors_header_exceeds_file"),
    ] {
        let mut raw = prefix.to_le_bytes().to_vec();
        raw.extend(tail);
        fs::write(f.0.join("model.safetensors"), raw).unwrap();
        let result = fingerprint_file(&f.0, "model.safetensors", true, true, &contract);
        assert_eq!(result["status"], "error");
        assert_eq!(result["error"]["code"], code);
    }
    for raw in [
        b"{\"x\":1,\"x\":2}".as_slice(),
        b"{\"x\":{\"y\":1,\"y\":2}}",
        b"{\"x\":NaN}",
        b"{\"x\":Infinity}",
    ] {
        assert!(parse_json(raw, "bad").is_err());
    }
    let mut h = tiny_header();
    h["__metadata__"] = json!({"x":42});
    assert_eq!(
        parse_tensors(&h, 4).unwrap_err().code,
        "safetensors_metadata_value_not_string"
    );
    h["__metadata__"] = Value::Null;
    assert!(parse_tensors(&h, 4).is_ok());
}

#[test]
fn full_extent_includes_extras_and_detects_gaps_overlaps_bad_shapes_and_dtype() {
    let mut h = tiny_header();
    h["extra"] = json!({"dtype":"U8","shape":[1],"data_offsets":[4,5]});
    let all = parse_tensors(&h, 5).unwrap();
    assert_eq!(
        check_contract(&all, &tiny_contract("BF16", true))["extra_tensor_count"],
        1
    );
    h["extra"]["data_offsets"] = json!([5, 6]);
    assert_eq!(
        parse_tensors(&h, 6).unwrap_err().code,
        "safetensors_offset_gap_or_overlap"
    );
    h["extra"]["data_offsets"] = json!([3, 4]);
    assert_eq!(
        parse_tensors(&h, 4).unwrap_err().code,
        "safetensors_offset_gap_or_overlap"
    );
    assert_eq!(
        parse_tensors(&tiny_header(), 5).unwrap_err().code,
        "safetensors_trailing_or_missing_data"
    );
    let mut h = tiny_header();
    h["private_tensor_name"]["shape"] = json!([3]);
    assert_eq!(
        parse_tensors(&h, 4).unwrap_err().code,
        "safetensors_shape_dtype_byte_count_mismatch"
    );
    h["private_tensor_name"]["shape"] = json!([u64::MAX, 2]);
    assert_eq!(
        parse_tensors(&h, 4).unwrap_err().code,
        "safetensors_shape_size_overflow"
    );
    h["private_tensor_name"]["dtype"] = json!("F4");
    assert_eq!(
        parse_tensors(&h, 4).unwrap_err().code,
        "safetensors_unsupported_dtype"
    );
}

#[test]
fn actual_loader_dtype_and_semantic_dtype_are_separate() {
    let tensors = parse_tensors(&tiny_header(), 4).unwrap();
    let semantic = check_contract(&tensors, &tiny_contract("F16", false));
    assert_eq!(semantic["loader_contract_pass"], true);
    assert_eq!(semantic["semantic_dtype_pass"], false);
    assert_eq!(semantic["mismatches"]["semantic_dtype"], 1);
    let strict = check_contract(&tensors, &tiny_contract("F16", true));
    assert_eq!(strict["loader_contract_pass"], false);
    let missing = check_contract(&BTreeMap::new(), &tiny_contract("BF16", true));
    assert_eq!(missing["mismatches"]["missing"], 1);
    assert_eq!(missing["failed_contract_indices"], json!([0]));
}

#[test]
fn scalar_f32_shape_unchecked_but_storage_and_type_required() {
    let raw = json!({"schema":"MOA_MODEL_LOADER_HEADER_CONTRACT_V1","source_sha256":{},
        "tensor_count":1,"required_storage_bytes":4,"required_tensors":[{
            "name":"scale","dtype":"F32","shape":null,"storage_bytes":4,
            "loader_dtype_check":"strict","scalar_shape_unchecked":true}]});
    let contract = parse_contract(&serde_json::to_vec(&raw).unwrap()).unwrap();
    for shape in [json!([]), json!([1]), json!([1, 1, 1])] {
        let header = json!({"scale":{"dtype":"F32","shape":shape,"data_offsets":[0,4]}});
        let tensors = parse_tensors(&header, 4).unwrap();
        assert_eq!(
            check_contract(&tensors, &contract)["loader_contract_pass"],
            true
        );
    }
}

#[test]
fn embedded_contract_and_missing_environment_errors() {
    let c = parse_contract(CONTRACT_BYTES).unwrap();
    assert_eq!(c.required.len(), 1149);
    assert_eq!(c.required_bytes, 9_304_785_568);
    assert_eq!(c.required.iter().filter(|r| r.strict).count(), 472);
    assert_eq!(c.required.iter().filter(|r| r.shape.is_none()).count(), 144);
    assert_eq!(
        c.sha256,
        "8e2b9346ff1f4ee3d58581326c85bb89f2dca5c97cb6c5e2b1fd9c3260754055"
    );
    let (report, exit) = build_report(None, true, &c);
    assert_eq!(exit, 1);
    assert_eq!(report["error"]["code"], "rngd_model_dir_unset");
}

#[test]
fn all_1149_embedded_requirements_validate_from_synthetic_header_only() {
    let contract = parse_contract(CONTRACT_BYTES).unwrap();
    let mut header = Map::new();
    let mut offset = 0u64;
    for r in &contract.required {
        header.insert(
            r.name.clone(),
            json!({"dtype":r.dtype, "shape":r.shape.clone().unwrap_or_default(),
            "data_offsets":[offset, offset+r.bytes]}),
        );
        offset += r.bytes;
    }
    // Only descriptors exist in memory: no 9.3-GB fixture or tensor data allocated.
    let tensors = parse_tensors(&Value::Object(header), offset).unwrap();
    let report = check_contract(&tensors, &contract);
    assert_eq!(report["status"], "pass");
    assert_eq!(report["required_tensors_found"], 1149);
    assert_eq!(report["extra_tensor_count"], 0);
}
