#!/usr/bin/env bash
# Puts the C library into a `lib` directory as a release archive has it, and
# checks that its name is the one the library gives itself (decision D75).
#
# Cargo makes `libemberwave_ffi.so` or `libemberwave_ffi.dylib`. The library
# names itself by the crate's major version (`crates/emberwave-ffi/build.rs`):
# the SONAME `libemberwave_ffi.so.0` on Linux, the install name
# `@rpath/libemberwave_ffi.0.dylib` on macOS, `.1` from 1.0.0. A program linked
# against it records that name and looks for that file, so the library is copied
# to it and the unversioned name stays beside it as a link, which is what a
# linker is given (`-lemberwave_ffi`). Windows keeps `emberwave_ffi.dll`.
#
# Usage: scripts/package_ffi.sh SRC DEST [--with-static]
#
#   SRC           where cargo put the library: target/release, target/debug
#   DEST          the directory to fill, made if it is not there
#   --with-static also copy the static library (`libemberwave_ffi.a`, or
#                 `emberwave_ffi.lib` on Windows), which a release builds apart
#
# Needs readelf on Linux and otool on macOS for the check.
set -euo pipefail

cd "$(dirname "$0")/.."
SRC=${1:?usage: scripts/package_ffi.sh SRC DEST [--with-static]}
DEST=${2:?usage: scripts/package_ffi.sh SRC DEST [--with-static]}
WITH_STATIC=${3:-}

# The workspace's version, which the crates take: its first number is the major.
version=$(sed -n '/^\[workspace.package\]/,/^\[/{s/^version = "\([0-9]*\)\..*/\1/p;}' Cargo.toml)
major=${version:?no version in Cargo.toml}

mkdir -p "$DEST"
case "$(uname -s)" in
    Linux)
        versioned=libemberwave_ffi.so.$major
        cp "$SRC/libemberwave_ffi.so" "$DEST/$versioned"
        ln -sf "$versioned" "$DEST/libemberwave_ffi.so"
        [ -z "$WITH_STATIC" ] || cp "$SRC/libemberwave_ffi.a" "$DEST/"
        soname=$(readelf -d "$DEST/$versioned" | sed -n 's/.*Library soname: \[\(.*\)\]/\1/p')
        [ "$soname" = "$versioned" ] || {
            echo "the library's SONAME is '$soname', not $versioned" >&2
            exit 1
        }
        ;;
    Darwin)
        versioned=libemberwave_ffi.$major.dylib
        cp "$SRC/libemberwave_ffi.dylib" "$DEST/$versioned"
        ln -sf "$versioned" "$DEST/libemberwave_ffi.dylib"
        [ -z "$WITH_STATIC" ] || cp "$SRC/libemberwave_ffi.a" "$DEST/"
        name=$(otool -D "$DEST/$versioned" | tail -n 1)
        [ "$name" = "@rpath/$versioned" ] || {
            echo "the library's install name is '$name', not @rpath/$versioned" >&2
            exit 1
        }
        ;;
    *)
        # Windows (Git Bash): the names do not change.
        cp "$SRC/emberwave_ffi.dll" "$SRC/emberwave_ffi.dll.lib" "$DEST/"
        [ -z "$WITH_STATIC" ] || cp "$SRC/emberwave_ffi.lib" "$DEST/"
        ;;
esac
echo "$DEST: $(ls "$DEST" | tr '\n' ' ')"
