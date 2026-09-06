//! Installable `qwen3tts-rs` command-line entry point.
//!
//! `build.rs` creates an includable copy of the historical `synthesize`
//! example so both invocation styles expose the same arguments and behavior.

mod legacy_cli {
    include!(concat!(env!("OUT_DIR"), "/qwen3tts_cli.rs"));

    pub(super) fn run() {
        main();
    }
}

fn main() {
    legacy_cli::run();
}
