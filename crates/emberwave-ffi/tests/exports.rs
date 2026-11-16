//! The shared library exports the functions of the C header and nothing else.
//!
//! The header, `include/emberwave.h`, is the list: a function that it declares
//! has to be exported, and a name that is exported without being declared there
//! fails the test. Rust's standard library, wgpu and the dependencies' leftovers
//! stay inside the library.
//!
//! One exception, on macOS: four names per Objective-C class that a dependency
//! declares with objc2's `define_class!` (raw-window-metal's
//! `RawWindowMetalLayer` is the one there is), `___CLASS_`,
//! `___DROP_FLAG_OFFSET_`, `___IVAR_OFFSET_` and `___REGISTER_CLASS_` followed
//! by the class's name. rustc exports every `#[no_mangle]` item of the
//! dependencies, passes the linker a list of them with `-exported_symbols_list`,
//! and ld64 takes no second list that removes names from it (`-unexported_*`
//! cannot be used with `-exported_*`), so they stay in the export table; no
//! client uses them (decision D73). Any other name that is not the header's
//! fails the test, on macOS and on Linux alike.
//!
//! The test builds the library, as `cargo build -p emberwave-ffi` does, and reads
//! its export table with `nm`, on macOS and Linux. Windows has no `nm`, and there
//! rustc exports only the `#[no_mangle]` functions.

#![cfg(any(target_os = "macos", target_os = "linux"))]

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::process::Command;

/// The names of the functions that the header declares: every `emberwave_`
/// identifier followed by an opening parenthesis on a line outside comments.
fn header_functions() -> BTreeSet<String> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("include/emberwave.h");
    let header = std::fs::read_to_string(path).expect("include/emberwave.h is readable");
    let mut names = BTreeSet::new();
    for line in header.lines() {
        let line = line.trim_start();
        if line.starts_with("//") || line.starts_with("/*") || line.starts_with('*') {
            continue;
        }
        let mut rest = line;
        while let Some(start) = rest.find("emberwave_") {
            let tail = &rest[start..];
            let end = tail
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .unwrap_or(tail.len());
            if tail[end..].starts_with('(') {
                names.insert(tail[..end].to_string());
            }
            rest = &tail[end..];
        }
    }
    names
}

/// The shared library, built anew, at the place cargo reports for it: `cargo
/// test` builds the library for the tests as an rlib only, so a shared library
/// left from an earlier build would be checked otherwise, and where cargo puts
/// the files depends on its version and its settings.
fn library() -> PathBuf {
    let profile = {
        let exe = std::env::current_exe().expect("the test's path is known");
        // The profile's directory is named after it: debug or release.
        exe.ancestors()
            .any(|dir| dir.file_name().is_some_and(|name| name == "release"))
    };
    let mut build = Command::new(env!("CARGO"));
    build.args([
        "build",
        "-p",
        "emberwave-ffi",
        "--lib",
        "--locked",
        "--message-format=json",
    ]);
    if profile {
        build.arg("--release");
    }
    let output = build.output().expect("cargo runs");
    assert!(
        output.status.success(),
        "cargo build -p emberwave-ffi failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    // The line of the library's artifact lists its files: the shared library,
    // and the rlib or the like beside it.
    let extension = if cfg!(target_os = "macos") {
        ".dylib"
    } else {
        ".so"
    };
    String::from_utf8(output.stdout)
        .expect("cargo prints UTF-8")
        .lines()
        .filter(|line| {
            line.contains(r#""reason":"compiler-artifact""#) && line.contains("emberwave_ffi")
        })
        .flat_map(|line| line.split('"'))
        .find(|piece| piece.ends_with(extension) && piece.contains("libemberwave_ffi"))
        .map(PathBuf::from)
        .expect("cargo reports the shared library")
}

/// The names the library exports, as the C programs that link it see them.
fn exported(library: &PathBuf) -> BTreeSet<String> {
    let mut nm = Command::new("nm");
    if cfg!(target_os = "macos") {
        // External symbols that are defined here.
        nm.arg("-gU");
    } else {
        nm.args(["-D", "--defined-only"]);
    }
    let output = nm.arg(library).output().expect("nm runs");
    assert!(
        output.status.success(),
        "nm failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .expect("nm prints UTF-8")
        .lines()
        .filter_map(|line| line.split_whitespace().last())
        // Mach-O symbols have a leading underscore. A line that is only an
        // address or a header has no name worth keeping.
        .map(|name| {
            name.strip_prefix('_')
                .filter(|_| cfg!(target_os = "macos"))
                .unwrap_or(name)
        })
        .map(str::to_string)
        .collect()
}

/// The Objective-C class registration names that objc2 leaves in the table of a
/// macOS library, as nm prints them after the leading underscore is removed.
fn objc_class_leftover(name: &str) -> bool {
    cfg!(target_os = "macos")
        && [
            "__CLASS_",
            "__DROP_FLAG_OFFSET_",
            "__IVAR_OFFSET_",
            "__REGISTER_CLASS_",
        ]
        .iter()
        .any(|prefix| name.starts_with(prefix))
}

#[test]
fn the_library_exports_the_header_s_functions_and_nothing_else() {
    let declared = header_functions();
    assert!(
        declared.len() > 50,
        "the header's functions were not found: {declared:?}"
    );
    let library = library();
    let exported = exported(&library);

    let hidden: Vec<_> = declared.difference(&exported).collect();
    let extra: Vec<_> = exported
        .difference(&declared)
        .filter(|name| !objc_class_leftover(name))
        .collect();
    assert!(
        hidden.is_empty(),
        "declared by the header but not exported by {}: {hidden:?}",
        library.display()
    );
    assert!(
        extra.is_empty(),
        "exported by {} but not declared by the header: {extra:?}",
        library.display()
    );
}
