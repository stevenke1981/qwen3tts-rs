# Local Scripts

This directory keeps ad-hoc Windows run wrappers that were previously in the
repository root.

These scripts are useful for reproducing local smoke runs, but they are not the
release build entrypoint. Prefer `tools/package_release.ps1` for packaging and
the documented `cargo run --example synthesize ...` commands for portable
usage examples.
