//! Installable `qwen3tts-rs` command-line entry point.
//!
//! The implementation stays shared with the historical `synthesize` example
//! so both invocation styles expose the same arguments and behavior.

mod legacy_cli {
    include!("../../examples/synthesize.rs");

    pub(super) fn run() {
        main();
    }
}

fn main() {
    legacy_cli::run();
}
