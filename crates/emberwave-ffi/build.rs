//! Names the shared library so that it can move with the application that
//! uses it. By default the macOS library carries the absolute path it was
//! built at, and the Linux one no name at all, which makes programs record
//! the path they were linked against. With these names they look it up by
//! file name: through the run path (`@rpath`) on macOS, and the usual search
//! (run path, `LD_LIBRARY_PATH`, system directories) on Linux.
//!
//! The name carries the crate's major version (decision D75): `libemberwave_ffi.so.0`
//! and `@rpath/libemberwave_ffi.0.dylib` while the major version is 0, `.so.1` and
//! `.1.dylib` from 1.0.0, so that from 1.0 the number means the ABI (D69) and a
//! program linked against one major version never loads another. On macOS the
//! library also has a compatibility version, the major version with zeros, and
//! the crate's version as its current one. Cargo still makes the file
//! `libemberwave_ffi.so` or `.dylib`: `scripts/package_ffi.sh` copies it to the
//! versioned name and leaves the unversioned one as a link. Windows keeps
//! `emberwave_ffi.dll`.

fn main() {
    // Without this, a change to any file of the package would run the script
    // again and rebuild the crate.
    println!("cargo::rerun-if-changed=build.rs");
    let version = |part: &str| {
        std::env::var(format!("CARGO_PKG_VERSION_{part}")).expect("cargo sets the version")
    };
    let (major, minor, patch) = (version("MAJOR"), version("MINOR"), version("PATCH"));
    match std::env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("macos") => {
            println!(
                "cargo::rustc-cdylib-link-arg=-Wl,-install_name,@rpath/libemberwave_ffi.{major}.dylib"
            );
            println!("cargo::rustc-cdylib-link-arg=-Wl,-compatibility_version,{major}.0.0");
            println!("cargo::rustc-cdylib-link-arg=-Wl,-current_version,{major}.{minor}.{patch}");
        }
        Ok("linux") => {
            println!("cargo::rustc-cdylib-link-arg=-Wl,-soname,libemberwave_ffi.so.{major}");
        }
        _ => {}
    }
}
