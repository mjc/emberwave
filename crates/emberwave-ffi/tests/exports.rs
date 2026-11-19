//! The shared library exports the functions of the C header and nothing else, and
//! names itself by the crate's major version.
//!
//! The header, `include/emberwave.h`, is the list: a function that it declares
//! has to be exported, and a name that is exported without being declared there
//! fails the test. Rust's standard library, wgpu and the dependencies' leftovers
//! stay inside the library.
//!
//! One exception, on macOS: the four names of the Objective-C class that a
//! dependency, raw-window-metal, declares with objc2's `define_class!`:
//! `___CLASS_RawWindowMetalLayer`, `___DROP_FLAG_OFFSET_RawWindowMetalLayer`,
//! `___IVAR_OFFSET_RawWindowMetalLayer` and `___REGISTER_CLASS_RawWindowMetalLayer`.
//! rustc exports every `#[export_name]` and `#[no_mangle]` item of the
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

/// Where cargo puts what this test run builds: the target directory, the
/// target triple if there is one (`cargo test --target`), and the profile's
/// directory. They are read from the test's own path, which is below
/// `TARGET/[TRIPLE/]PROFILE/`, so that the build below goes to the same place
/// whatever `--target-dir`, `CARGO_TARGET_DIR`, `--target` and `--profile`
/// the test was started with.
struct Layout {
    target_dir: PathBuf,
    triple: Option<String>,
    profile_dir: String,
}

fn layout() -> Layout {
    let exe = std::env::current_exe().expect("the test's path is known");
    // The outermost directory that cargo marked as a target directory: with
    // CACHEDIR.TAG where cargo made the directory, and with .rustc_info.json,
    // which it writes into a directory that was there already (an empty
    // CARGO_TARGET_DIR from mktemp, say).
    let target_dir = exe
        .ancestors()
        .filter(|dir| dir.join("CACHEDIR.TAG").exists() || dir.join(".rustc_info.json").exists())
        .last()
        .expect("the test runs below a target directory")
        .to_path_buf();
    let rest: Vec<String> = exe
        .strip_prefix(&target_dir)
        .expect("the test is below the target directory")
        .components()
        .map(|part| part.as_os_str().to_string_lossy().into_owned())
        .collect();
    // TARGET/PROFILE/... or TARGET/TRIPLE/PROFILE/...: a triple has dashes,
    // which no profile's directory has.
    let (triple, profile_dir) = if rest[0].contains('-') {
        (Some(rest[0].clone()), rest[1].clone())
    } else {
        (None, rest[0].clone())
    };
    Layout {
        target_dir,
        triple,
        profile_dir,
    }
}

/// The shared library, built anew, at the place cargo reports for it: `cargo
/// test` compiles the library for the tests but does not put the shared
/// library at the top of the profile's directory, where a C program looks for
/// it, and a shared library left there from an earlier build would be checked
/// otherwise.
fn library() -> PathBuf {
    let layout = layout();
    let mut build = Command::new(env!("CARGO"));
    build.args([
        "build",
        "-p",
        "emberwave-ffi",
        "--lib",
        "--locked",
        "--message-format=json",
        "--target-dir",
    ]);
    build.arg(&layout.target_dir);
    if let Some(triple) = &layout.triple {
        build.args(["--target", triple]);
    }
    // The dev and release profiles have directories of their own, others are
    // named after the profile.
    match layout.profile_dir.as_str() {
        "debug" => {}
        "release" => {
            build.arg("--release");
        }
        other => {
            build.args(["--profile", other]);
        }
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

/// The four names of the Objective-C class that raw-window-metal declares,
/// `RawWindowMetalLayer`, as nm prints them after the leading underscore is
/// removed. A class that another dependency declares, or a change of these
/// names, fails the test until someone has looked at it.
fn objc_class_leftover(name: &str) -> bool {
    cfg!(target_os = "macos")
        && [
            "__CLASS_RawWindowMetalLayer",
            "__DROP_FLAG_OFFSET_RawWindowMetalLayer",
            "__IVAR_OFFSET_RawWindowMetalLayer",
            "__REGISTER_CLASS_RawWindowMetalLayer",
        ]
        .contains(&name)
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

/// The name the library gives itself, and on macOS its compatibility and current
/// versions, as the platform's tools print them: `readelf -d` for the SONAME on
/// Linux, `otool -l` for the install name and the versions on macOS.
fn own_name(library: &PathBuf) -> (String, Option<(String, String)>) {
    if cfg!(target_os = "macos") {
        let output = Command::new("otool")
            .arg("-l")
            .arg(library)
            .output()
            .expect("otool runs");
        assert!(output.status.success(), "otool failed");
        let text = String::from_utf8(output.stdout).expect("otool prints UTF-8");
        // The LC_ID_DYLIB command, up to the next one.
        let block = text
            .split("cmd LC_ID_DYLIB")
            .nth(1)
            .expect("the library has an install name")
            .split("Load command")
            .next()
            .expect("the command ends");
        let field = |label: &str, after: &str| {
            block
                .lines()
                .map(str::trim)
                .find_map(|line| line.strip_prefix(label))
                .map(|rest| {
                    rest.trim()
                        .split(after)
                        .next()
                        .expect("a value")
                        .trim()
                        .to_string()
                })
                .unwrap_or_else(|| panic!("no {label} in {block}"))
        };
        let name = field("name ", " (offset");
        (
            name,
            Some((
                field("compatibility version ", " "),
                field("current version ", " "),
            )),
        )
    } else {
        // readelf's labels are translated; the C locale keeps the ones parsed below.
        let output = Command::new("readelf")
            .env("LC_ALL", "C")
            .arg("-d")
            .arg(library)
            .output()
            .expect("readelf runs");
        assert!(output.status.success(), "readelf failed");
        let text = String::from_utf8(output.stdout).expect("readelf prints UTF-8");
        let name = text
            .lines()
            .find_map(|line| line.split("Library soname: [").nth(1))
            .and_then(|rest| rest.split(']').next())
            .expect("the library has a SONAME")
            .to_string();
        (name, None)
    }
}

#[test]
fn the_library_names_itself_by_its_major_version() {
    let major = env!("CARGO_PKG_VERSION_MAJOR");
    let version = format!(
        "{major}.{}.{}",
        env!("CARGO_PKG_VERSION_MINOR"),
        env!("CARGO_PKG_VERSION_PATCH")
    );
    let (name, versions) = own_name(&library());
    if cfg!(target_os = "macos") {
        assert_eq!(name, format!("@rpath/libemberwave_ffi.{major}.dylib"));
        let (compatibility, current) = versions.expect("macOS libraries have versions");
        assert_eq!(compatibility, format!("{major}.0.0"));
        assert_eq!(current, version);
    } else {
        assert_eq!(name, format!("libemberwave_ffi.so.{major}"));
    }
}
