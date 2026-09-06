use std::{env, fs, path::PathBuf};

const CLI_SOURCE: &str = "examples/synthesize.rs";
const GENERATED_CLI: &str = "qwen3tts_cli.rs";

fn main() {
    println!("cargo:rerun-if-changed={CLI_SOURCE}");

    let source = fs::read_to_string(CLI_SOURCE)
        .unwrap_or_else(|error| panic!("failed to read {CLI_SOURCE}: {error}"));
    let generated = normalize_crate_docs(&source);
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo must provide OUT_DIR"));

    fs::write(out_dir.join(GENERATED_CLI), generated)
        .unwrap_or_else(|error| panic!("failed to generate CLI source: {error}"));
}

fn normalize_crate_docs(source: &str) -> String {
    let mut generated = String::with_capacity(source.len());

    for line in source.split_inclusive('\n') {
        if let Some(rest) = line.strip_prefix("//!") {
            generated.push_str("//");
            generated.push_str(rest);
        } else {
            generated.push_str(line);
        }
    }

    generated
}
