//! CPU-only, read-only asset/header inspection. No Furiosa SDK or NPU calls.
//! Errors deliberately contain fixed codes, never paths or model contents.
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::{self, File, Metadata};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const CHUNK_BYTES: usize = 8 * 1024 * 1024;
const MAX_HEADER_BYTES: u64 = 16 * 1024 * 1024;
const WEIGHT_HASH_SECONDS: u64 = 40;
const CONTRACT_BYTES: &[u8] = include_bytes!("../loader-header-contract.json");
const METADATA_FILES: [&str; 4] = [
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "chat_template.json",
];

#[derive(Debug)]
struct Failure {
    code: &'static str,
    errno: Option<i32>,
}
impl Failure {
    fn code(code: &'static str) -> Self {
        Self { code, errno: None }
    }
    fn json(&self) -> Value {
        if self.code == "os_error" {
            json!({"code": self.code, "errno": self.errno})
        } else {
            json!({"code": self.code})
        }
    }
}
impl From<io::Error> for Failure {
    fn from(error: io::Error) -> Self {
        Self {
            code: "os_error",
            errno: error.raw_os_error(),
        }
    }
}
type Result<T> = std::result::Result<T, Failure>;
fn require(condition: bool, code: &'static str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(Failure::code(code))
    }
}
fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    bytes
        .iter()
        .flat_map(|b| {
            [
                DIGITS[(b >> 4) as usize] as char,
                DIGITS[(b & 15) as usize] as char,
            ]
        })
        .collect()
}
fn digest(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

// serde_json::Value otherwise silently keeps only the last duplicate key.
// This visitor rejects duplicates at every nesting level without echoing keys.
struct UniqueValue(Value);
impl<'de> Deserialize<'de> for UniqueValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct UniqueVisitor;
        impl<'de> Visitor<'de> for UniqueVisitor {
            type Value = UniqueValue;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("JSON value")
            }
            fn visit_bool<E: de::Error>(self, value: bool) -> std::result::Result<Self::Value, E> {
                Ok(UniqueValue(value.into()))
            }
            fn visit_i64<E: de::Error>(self, value: i64) -> std::result::Result<Self::Value, E> {
                Ok(UniqueValue(value.into()))
            }
            fn visit_u64<E: de::Error>(self, value: u64) -> std::result::Result<Self::Value, E> {
                Ok(UniqueValue(value.into()))
            }
            fn visit_f64<E: de::Error>(self, value: f64) -> std::result::Result<Self::Value, E> {
                serde_json::Number::from_f64(value)
                    .map(|v| UniqueValue(Value::Number(v)))
                    .ok_or_else(|| E::custom("nonfinite JSON number"))
            }
            fn visit_str<E: de::Error>(self, value: &str) -> std::result::Result<Self::Value, E> {
                Ok(UniqueValue(value.into()))
            }
            fn visit_string<E: de::Error>(
                self,
                value: String,
            ) -> std::result::Result<Self::Value, E> {
                Ok(UniqueValue(value.into()))
            }
            fn visit_unit<E: de::Error>(self) -> std::result::Result<Self::Value, E> {
                Ok(UniqueValue(Value::Null))
            }
            fn visit_none<E: de::Error>(self) -> std::result::Result<Self::Value, E> {
                Ok(UniqueValue(Value::Null))
            }
            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(UniqueValue(value)) = seq.next_element()? {
                    values.push(value);
                }
                Ok(UniqueValue(values.into()))
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut values = Map::new();
                while let Some(key) = map.next_key::<String>()? {
                    if values.contains_key(&key) {
                        return Err(de::Error::custom("duplicate JSON key"));
                    }
                    let UniqueValue(value) = map.next_value()?;
                    values.insert(key, value);
                }
                Ok(UniqueValue(Value::Object(values)))
            }
        }
        deserializer.deserialize_any(UniqueVisitor)
    }
}
fn parse_json(bytes: &[u8], code: &'static str) -> Result<Value> {
    serde_json::from_slice::<UniqueValue>(bytes)
        .map(|v| v.0)
        .map_err(|_| Failure::code(code))
}

#[derive(Debug, PartialEq, Eq)]
struct Identity {
    dev: u64,
    ino: u64,
    mode: u32,
    size: u64,
    mtime: (i64, i64),
    ctime: (i64, i64),
}
fn identity(m: &Metadata) -> Identity {
    Identity {
        dev: m.dev(),
        ino: m.ino(),
        mode: m.mode(),
        size: m.len(),
        mtime: (m.mtime(), m.mtime_nsec()),
        ctime: (m.ctime(), m.ctime_nsec()),
    }
}
fn public_stat(m: &Metadata) -> Value {
    json!({"size_bytes": m.len(), "mtime_ns": i128::from(m.mtime()) * 1_000_000_000 + i128::from(m.mtime_nsec())})
}
fn select_directory(root: &Path) -> Result<(&'static str, PathBuf)> {
    for (selector, path) in [
        ("direct", root.to_path_buf()),
        ("Gemma4-12B-it", root.join("Gemma4-12B-it")),
        ("gemma-4-12b-it-NVFP4", root.join("gemma-4-12b-it-NVFP4")),
    ] {
        if path.join("model.safetensors").is_file() {
            return Ok((selector, path));
        }
    }
    Err(Failure::code("no_model_safetensors_in_three_candidates"))
}

fn dtype_size(dtype: &str) -> Option<u64> {
    // Exactly safetensors 0.4.5 Dtype, used by the frozen model loader.
    match dtype {
        "BOOL" | "U8" | "I8" | "F8_E5M2" | "F8_E4M3" => Some(1),
        "I16" | "U16" | "F16" | "BF16" => Some(2),
        "I32" | "U32" | "F32" => Some(4),
        "I64" | "U64" | "F64" => Some(8),
        _ => None,
    }
}
#[derive(Debug)]
struct Tensor {
    dtype: String,
    shape: Vec<u64>,
    begin: u64,
    end: u64,
}
fn uint_array(value: &Value, code: &'static str) -> Result<Vec<u64>> {
    value
        .as_array()
        .ok_or_else(|| Failure::code(code))?
        .iter()
        .map(|v| v.as_u64().ok_or_else(|| Failure::code(code)))
        .collect()
}
fn parse_tensors(header: &Value, data_bytes: u64) -> Result<BTreeMap<String, Tensor>> {
    let header = header
        .as_object()
        .ok_or_else(|| Failure::code("safetensors_header_not_object"))?;
    let mut tensors = BTreeMap::new();
    for (name, descriptor) in header {
        if name == "__metadata__" {
            if descriptor.is_null() {
                continue;
            } // safetensors 0.4.5 Option<HashMap<String,String>>.
            let metadata = descriptor
                .as_object()
                .ok_or_else(|| Failure::code("safetensors_metadata_not_object"))?;
            require(
                metadata.values().all(Value::is_string),
                "safetensors_metadata_value_not_string",
            )?;
            continue;
        }
        let d = descriptor
            .as_object()
            .ok_or_else(|| Failure::code("safetensors_tensor_descriptor_not_object"))?;
        let dtype = d
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| Failure::code("safetensors_invalid_dtype"))?;
        let size =
            dtype_size(dtype).ok_or_else(|| Failure::code("safetensors_unsupported_dtype"))?;
        let shape = uint_array(
            d.get("shape").unwrap_or(&Value::Null),
            "safetensors_invalid_shape",
        )?;
        let offsets = uint_array(
            d.get("data_offsets").unwrap_or(&Value::Null),
            "safetensors_invalid_offsets",
        )?;
        require(offsets.len() == 2, "safetensors_invalid_offsets")?;
        let elements = shape
            .iter()
            .try_fold(1u64, |a, b| a.checked_mul(*b))
            .ok_or_else(|| Failure::code("safetensors_shape_size_overflow"))?;
        let bytes = elements
            .checked_mul(size)
            .ok_or_else(|| Failure::code("safetensors_shape_size_overflow"))?;
        require(
            offsets[1] >= offsets[0] && offsets[1] <= data_bytes,
            "safetensors_offset_out_of_bounds",
        )?;
        require(
            offsets[1] - offsets[0] == bytes,
            "safetensors_shape_dtype_byte_count_mismatch",
        )?;
        tensors.insert(
            name.clone(),
            Tensor {
                dtype: dtype.to_owned(),
                shape,
                begin: offsets[0],
                end: offsets[1],
            },
        );
    }
    let mut ordered: Vec<_> = tensors.values().collect();
    ordered.sort_by_key(|t| (t.begin, t.end));
    let mut end = 0;
    for tensor in ordered {
        require(tensor.begin == end, "safetensors_offset_gap_or_overlap")?;
        end = tensor.end;
    }
    require(end == data_bytes, "safetensors_trailing_or_missing_data")?;
    Ok(tensors)
}

struct Required {
    name: String,
    dtype: String,
    shape: Option<Vec<u64>>,
    bytes: u64,
    strict: bool,
}
struct Contract {
    required: Vec<Required>,
    sha256: String,
    source_sha256: Value,
    required_bytes: u64,
}
fn parse_contract(raw: &[u8]) -> Result<Contract> {
    let contract = parse_json(raw, "invalid_embedded_contract_json")?;
    require(
        contract["schema"] == "MOA_MODEL_LOADER_HEADER_CONTRACT_V1",
        "invalid_embedded_contract_schema",
    )?;
    let list = contract["required_tensors"]
        .as_array()
        .ok_or_else(|| Failure::code("invalid_embedded_contract_tensors"))?;
    let mut seen = BTreeSet::new();
    let mut required = Vec::new();
    let mut required_bytes = 0u64;
    for item in list {
        let name = item["name"]
            .as_str()
            .ok_or_else(|| Failure::code("invalid_embedded_contract_name"))?;
        require(
            seen.insert(name.to_owned()),
            "duplicate_embedded_contract_name",
        )?;
        let dtype = item["dtype"]
            .as_str()
            .ok_or_else(|| Failure::code("invalid_embedded_contract_dtype"))?;
        require(
            dtype_size(dtype).is_some(),
            "invalid_embedded_contract_dtype",
        )?;
        let bytes = item["storage_bytes"]
            .as_u64()
            .ok_or_else(|| Failure::code("invalid_embedded_contract_bytes"))?;
        let shape = if item["shape"].is_null() {
            None
        } else {
            Some(uint_array(
                &item["shape"],
                "invalid_embedded_contract_shape",
            )?)
        };
        let strict = match item["loader_dtype_check"].as_str() {
            Some("strict") => true,
            Some("semantic_only") => false,
            _ => return Err(Failure::code("invalid_embedded_contract_dtype_check")),
        };
        require(
            shape.is_some()
                || (strict
                    && dtype == "F32"
                    && bytes == 4
                    && item["scalar_shape_unchecked"] == true),
            "invalid_embedded_contract_unchecked_shape",
        )?;
        required_bytes = required_bytes
            .checked_add(bytes)
            .ok_or_else(|| Failure::code("embedded_contract_size_overflow"))?;
        required.push(Required {
            name: name.to_owned(),
            dtype: dtype.to_owned(),
            shape,
            bytes,
            strict,
        });
    }
    require(
        contract["tensor_count"].as_u64() == Some(required.len() as u64),
        "embedded_contract_count_mismatch",
    )?;
    require(
        contract["required_storage_bytes"].as_u64() == Some(required_bytes),
        "embedded_contract_size_mismatch",
    )?;
    // This is a locally generated, compiled-in contract, never asset-supplied data.
    let source_sha256 = contract["source_sha256"].clone();
    require(
        source_sha256.is_object(),
        "invalid_embedded_contract_source_hashes",
    )?;
    Ok(Contract {
        required,
        sha256: digest(raw),
        source_sha256,
        required_bytes,
    })
}
fn check_contract(tensors: &BTreeMap<String, Tensor>, contract: &Contract) -> Value {
    let (mut missing, mut shape, mut bytes, mut strict_dtype, mut semantic_dtype) = (0, 0, 0, 0, 0);
    let mut failed = Vec::new();
    let mut found = 0;
    for (index, expected) in contract.required.iter().enumerate() {
        let Some(actual) = tensors.get(&expected.name) else {
            missing += 1;
            failed.push(index);
            continue;
        };
        found += 1;
        let mut bad = false;
        if expected.shape.as_ref().is_some_and(|s| *s != actual.shape) {
            shape += 1;
            bad = true;
        }
        if actual.end - actual.begin != expected.bytes {
            bytes += 1;
            bad = true;
        }
        if actual.dtype != expected.dtype {
            semantic_dtype += 1;
            if expected.strict {
                strict_dtype += 1;
            }
            bad = true;
        }
        if bad {
            failed.push(index);
        }
    }
    let loader_pass = missing + shape + bytes + strict_dtype == 0;
    let semantic_pass = loader_pass && semantic_dtype == 0;
    json!({
        "status": if loader_pass && semantic_pass { "pass" } else { "fail" },
        "contract_sha256": contract.sha256, "source_sha256": contract.source_sha256,
        "required_tensor_count": contract.required.len(), "required_storage_bytes": contract.required_bytes,
        "required_tensors_found": found, "extra_tensor_count": tensors.len() - found,
        "loader_contract_pass": loader_pass, "semantic_dtype_pass": semantic_pass,
        "mismatches": {"missing": missing, "shape": shape, "storage_bytes": bytes,
            "loader_enforced_dtype": strict_dtype, "semantic_dtype": semantic_dtype},
        "failed_contract_indices": failed,
        "scope": "Header-only necessary conditions; not proof of load_model success. No tensor values, global-scale finite/nonzero or reciprocal values, metadata contents, device allocation, or NPU behavior inspected."
    })
}
fn read_header(stream: &mut File, file_size: u64, contract: &Contract) -> Result<(Value, Value)> {
    require(file_size >= 8, "short_safetensors_length_prefix")?;
    let mut prefix = [0u8; 8];
    stream.read_exact(&mut prefix)?;
    let len = u64::from_le_bytes(prefix);
    require(
        len > 0 && len <= MAX_HEADER_BYTES,
        "safetensors_header_length_out_of_bounds",
    )?;
    require(len <= file_size - 8, "safetensors_header_exceeds_file")?;
    let mut raw = vec![0u8; len as usize];
    stream.read_exact(&mut raw)?;
    let value = parse_json(&raw, "invalid_safetensors_json_header")?;
    let tensors = parse_tensors(&value, file_size - 8 - len)?;
    let checked = check_contract(&tensors, contract);
    Ok((
        json!({"status":"complete", "length_bytes":len, "sha256":digest(&raw),
        "sha256_scope":"raw_json_header_bytes_excluding_8_byte_length_prefix", "tensor_count":tensors.len(),
        "validation":"Duplicate-free JSON; safetensors-0.4.5 descriptor dtype/shape bytes, contiguous offsets, complete file data extent and string-map-or-null metadata; stricter 16-MiB header cap"}),
        checked,
    ))
}

fn hash_file<R: Read + Seek>(stream: &mut R, size: u64, budget: Option<Duration>) -> Result<Value> {
    let started = Instant::now();
    stream.seek(SeekFrom::Start(0))?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; CHUNK_BYTES];
    let mut count = 0u64;
    let mut complete = true;
    while count < size {
        if budget.is_some_and(|b| started.elapsed() >= b) {
            complete = false;
            break;
        }
        let len = stream.read(&mut buffer[..((size - count).min(CHUNK_BYTES as u64) as usize)])?;
        require(len > 0, "file_short_read")?;
        hash.update(&buffer[..len]);
        count += len as u64;
        if budget.is_some_and(|b| started.elapsed() >= b) {
            complete = false;
            break;
        }
    }
    let mut result = json!({"status":if complete {"complete"} else {"incomplete"},
        "sha256":if complete {Some(hex(&hash.finalize()))} else {None},
        "sha256_scope":if complete {Some("entire_file")} else {None},
        "bytes_hashed":count, "elapsed_seconds":started.elapsed().as_secs_f64()});
    if let Some(budget) = budget {
        result["budget_seconds"] = json!(budget.as_secs_f64());
        result["budget_enforcement"] =
            json!("monotonic_checks_around_each_chunk; blocking OS reads are not interrupted");
    }
    if !complete {
        result["reason"] = json!("time_budget_exceeded_no_digest_published");
    }
    Ok(result)
}

fn fingerprint_file(
    directory: &Path,
    basename: &str,
    weight: bool,
    header_only: bool,
    contract: &Contract,
) -> Value {
    fingerprint_with_hook(directory, basename, weight, header_only, contract, || {})
}
fn fingerprint_with_hook<F: FnOnce()>(
    directory: &Path,
    basename: &str,
    weight: bool,
    header_only: bool,
    contract: &Contract,
    after_read: F,
) -> Value {
    let path = directory.join(basename);
    let mut result = json!({"basename": basename});
    let before = match fs::metadata(&path) {
        Ok(value) => value,
        Err(error) => {
            result["status"] = json!(if matches!(error.raw_os_error(), Some(2 | 20)) {
                "missing"
            } else {
                "error"
            });
            result["error"] = Failure::from(error).json();
            return result;
        }
    };
    result["stat_before"] = public_stat(&before);
    let (mut opened, mut after_fd) = (None, None);
    let inspected = (|| -> Result<()> {
        require(before.is_file(), "not_regular_file")?;
        let mut stream = File::open(&path)?;
        opened = Some(stream.metadata()?);
        let reading = (|| -> Result<()> {
            require(
                identity(opened.as_ref().unwrap()) == identity(&before),
                "file_changed_before_open",
            )?;
            if weight {
                let (header, checked) = read_header(&mut stream, before.len(), contract)?;
                result["header"] = header;
                result["header_contract"] = checked;
            }
            result["full_file"] = if weight && header_only {
                json!({"status":"skipped", "sha256":null, "sha256_scope":null, "bytes_hashed":0,
                    "reason":"header_only_requested; no tensor data bytes read"})
            } else {
                hash_file(
                    &mut stream,
                    before.len(),
                    weight.then_some(Duration::from_secs(WEIGHT_HASH_SECONDS)),
                )?
            };
            Ok(())
        })();
        after_read();
        after_fd = Some(stream.metadata()?);
        reading
    })();
    let after_path = fs::metadata(&path);
    if let Ok(ref value) = after_path {
        result["stat_after"] = public_stat(value);
    }
    let stable = opened
        .as_ref()
        .zip(after_fd.as_ref())
        .zip(after_path.as_ref().ok())
        .is_some_and(|((a, b), c)| [a, b, c].iter().all(|m| identity(m) == identity(&before)));
    result["file_unchanged_by_stat"] = json!(stable);
    let error = inspected
        .err()
        .or_else(|| after_path.err().map(Failure::from))
        .or_else(|| (!stable).then(|| Failure::code("file_changed_during_read")));
    if let Some(error) = error {
        result["status"] = json!("error");
        result["error"] = error.json();
        for field in ["header", "full_file"] {
            if let Some(value) = result.get_mut(field) {
                value["status"] = json!("invalidated");
                value["sha256"] = Value::Null;
                value["sha256_scope"] = Value::Null;
            }
        }
        if let Some(value) = result.get_mut("header_contract") {
            value["status"] = json!("invalidated");
            value["loader_contract_pass"] = Value::Null;
            value["semantic_dtype_pass"] = Value::Null;
        }
    } else if weight && result["header_contract"]["status"] != "pass" {
        result["status"] = json!("error");
        result["error"] = json!({"code":"header_contract_mismatch"});
    } else {
        result["status"] = if weight && header_only {
            json!("header_only")
        } else {
            result["full_file"]["status"].clone()
        };
    }
    result
}

fn build_report(root: Option<PathBuf>, header_only: bool, contract: &Contract) -> (Value, i32) {
    let mut report = json!({"schema":"MOA_MODEL_ASSET_INSPECTOR_RUST_V1", "selector":null,
        "chunk_bytes":CHUNK_BYTES, "max_header_bytes":MAX_HEADER_BYTES,
        "weight_hash_budget_seconds":WEIGHT_HASH_SECONDS, "header_only":header_only,
        "files":[], "embedded_contract_sha256":contract.sha256,
        "scope":"Read-only asset identity and necessary loader header conditions; no model contents or paths printed, no NPU execution, no model-value validity or inference validation."});
    let work = (|| -> Result<i32> {
        let root = root.ok_or_else(|| Failure::code("rngd_model_dir_unset"))?;
        let (selector, directory) = select_directory(&root)?;
        report["selector"] = json!(selector);
        // Preserve four small file fingerprints before using the weight hash budget.
        let metadata: Vec<_> = METADATA_FILES
            .iter()
            .map(|name| fingerprint_file(&directory, name, false, header_only, contract))
            .collect();
        let weight = fingerprint_file(&directory, "model.safetensors", true, header_only, contract);
        let weight_error = matches!(weight["status"].as_str(), Some("error" | "missing"));
        let mut files = vec![weight];
        files.extend(metadata);
        let error = weight_error || files.iter().any(|f| f["status"] == "error");
        let partial = files
            .iter()
            .any(|f| !matches!(f["status"].as_str(), Some("complete" | "header_only")));
        report["status"] = json!(if error {
            "error"
        } else if partial {
            "partial"
        } else if header_only {
            "header_only"
        } else {
            "complete"
        });
        report["files"] = json!(files);
        Ok(if error { 1 } else { 0 })
    })();
    match work {
        Ok(exit) => (report, exit),
        Err(error) => {
            report["status"] = json!("error");
            report["error"] = error.json();
            (report, 1)
        }
    }
}
fn main() {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let valid = args.is_empty() || (args.len() == 1 && args[0] == "--header-only");
    let (report, exit) = if !valid {
        (
            json!({"schema":"MOA_MODEL_ASSET_INSPECTOR_RUST_V1", "status":"error",
            "error":{"code":"invalid_arguments"}, "usage":"moa-model-asset-inspector [--header-only]"}),
            2,
        )
    } else {
        match parse_contract(CONTRACT_BYTES) {
            Ok(contract) => build_report(
                std::env::var_os("RNGD_MODEL_DIR").map(PathBuf::from),
                !args.is_empty(),
                &contract,
            ),
            Err(error) => (
                json!({"schema":"MOA_MODEL_ASSET_INSPECTOR_RUST_V1", "status":"error", "error":error.json()}),
                2,
            ),
        }
    };
    let mut stdout = io::stdout().lock();
    if serde_json::to_writer(&mut stdout, &report).is_err() || stdout.write_all(b"\n").is_err() {
        std::process::exit(2);
    }
    std::process::exit(exit);
}

#[cfg(test)]
mod tests;
