//! Lossless JSON tree deltas. This changes storage representation, not cognition.
use serde_json::{json, Value};
use std::ffi::{c_char, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;

fn equivalent(a: &Value, b: &Value) -> bool {
    match (a,b) {
        (Value::Number(x),Value::Number(y)) if x.is_f64() || y.is_f64() =>
            x.is_f64() && y.is_f64() && x.as_f64().unwrap().to_bits()==y.as_f64().unwrap().to_bits(),
        (Value::Array(x),Value::Array(y)) => x.len()==y.len() && x.iter().zip(y).all(|(a,b)|equivalent(a,b)),
        (Value::Object(x),Value::Object(y)) => x.len()==y.len() && x.iter().all(|(k,v)| y.get(k).map(|w|equivalent(v,w)).unwrap_or(false)),
        _ => a==b,
    }
}

fn walk(old: &Value, new: &Value, path: &mut Vec<Value>, sets: &mut Vec<Value>, deletes: &mut Vec<Value>, appends: &mut Vec<Value>) {
    match (old, new) {
        (Value::Object(a), Value::Object(b)) => {
            for key in a.keys() {
                if !b.contains_key(key) {
                    path.push(Value::String(key.clone()));
                    deletes.push(Value::Array(path.clone()));
                    path.pop();
                }
            }
            for (key, value) in b {
                path.push(Value::String(key.clone()));
                match a.get(key) {
                    Some(previous) => walk(previous, value, path, sets, deletes, appends),
                    None => sets.push(json!([path, value])),
                }
                path.pop();
            }
        }
        (Value::Array(a), Value::Array(b)) if b.len() > a.len() && a.iter().zip(b).all(|(x,y)|equivalent(x,y)) => {
            appends.push(json!([path, &b[a.len()..]]));
        }
        _ if !equivalent(old,new) => sets.push(json!([path, new])),
        _ => (),
    }
}

pub fn diff(old: &Value, new: &Value) -> Value {
    let (mut sets, mut deletes, mut appends) = (Vec::new(), Vec::new(), Vec::new());
    walk(old, new, &mut Vec::new(), &mut sets, &mut deletes, &mut appends);
    json!({"set": sets, "delete": deletes, "append": appends})
}

/// Inputs must be valid readable buffers of their declared lengths. Returned
/// allocation belongs to this library and must be released by regesh_memory_free.
#[no_mangle]
pub unsafe extern "C" fn regesh_memory_diff(old: *const u8, old_len: usize, new: *const u8, new_len: usize) -> *mut c_char {
    if old.is_null() || new.is_null() { return std::ptr::null_mut(); }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let a: Value = serde_json::from_slice(slice::from_raw_parts(old, old_len))?;
        let b: Value = serde_json::from_slice(slice::from_raw_parts(new, new_len))?;
        Ok::<Value, serde_json::Error>(diff(&a, &b))
    }));
    let value = match result {
        Ok(Ok(value)) => value,
        Ok(Err(error)) => json!({"error": format!("invalid JSON: {error}")}),
        Err(_) => json!({"error": "native delta kernel panicked"}),
    };
    CString::new(value.to_string()).expect("JSON contains no literal NUL").into_raw()
}

/// Release a pointer returned by regesh_memory_diff exactly once.
#[no_mangle]
pub unsafe extern "C" fn regesh_memory_free(ptr: *mut c_char) {
    if !ptr.is_null() { drop(CString::from_raw(ptr)); }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn additions_deletions_and_append() {
        assert_eq!(diff(&json!({"a": [1], "gone": 4}), &json!({"a": [1, 2], "b": null})),
                   json!({"set": [[["b"], null]], "delete": [["gone"]], "append": [[["a"], [2]]]}));
    }
    #[test]
    fn numeric_representation_is_preserved() {
        let a: Value=serde_json::from_str("{\"a\":1,\"b\":-0.0}").unwrap();
        let b: Value=serde_json::from_str("{\"a\":1.0,\"b\":0.0}").unwrap();
        // JSON integer/float distinctions and signed zero affect Python canonical hashes.
        assert_eq!(diff(&a,&b)["set"].as_array().unwrap().len(), 2);
    }
}
