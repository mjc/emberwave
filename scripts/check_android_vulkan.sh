#!/usr/bin/env bash
# Cross-compiles emberwave-ffi (cdylib + staticlib) and the Android Vulkan R16
# C smoke harness (crates/emberwave-ffi/examples/android/vulkan_r16.c) for
# aarch64-linux-android using the Android NDK, and optionally runs both the
# dynamic and static binaries on an attached Android device over adb.
#
# Usage: scripts/check_android_vulkan.sh [--build-only] [--release] [--serial SERIAL]
#
# Environment:
#   ANDROID_NDK_HOME / ANDROID_NDK_ROOT / ANDROID_NDK_LATEST_HOME  Android NDK root
#   EMBERWAVE_ANDROID_API                                          Android API level (default: 24)
#   EMBERWAVE_ANDROID_SERIAL                                       adb device serial
set -euo pipefail

cd "$(dirname "$0")/.."

BUILD_ONLY=0
PROFILE=debug
CARGO_PROFILE_ARGS=()
SERIAL=${EMBERWAVE_ANDROID_SERIAL:-}

while [ $# -gt 0 ]; do
    case "$1" in
        --build-only)
            BUILD_ONLY=1
            shift
            ;;
        --release)
            PROFILE=release
            CARGO_PROFILE_ARGS=(--release)
            shift
            ;;
        --serial)
            SERIAL=${2:?--serial requires a device serial}
            shift 2
            ;;
        *)
            echo "usage: $0 [--build-only] [--release] [--serial SERIAL]" >&2
            exit 2
            ;;
    esac
done

TARGET=aarch64-linux-android
API_LEVEL=${EMBERWAVE_ANDROID_API:-24}

NDK=${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-${ANDROID_NDK_LATEST_HOME:-}}}
if [ -z "$NDK" ] || [ ! -d "$NDK" ]; then
    echo "Android NDK not found: set ANDROID_NDK_HOME (or ANDROID_NDK_ROOT)" >&2
    exit 1
fi

HOST_TAG=$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m)
TOOLCHAIN="$NDK/toolchains/llvm/prebuilt/$HOST_TAG/bin"
if [ ! -d "$TOOLCHAIN" ]; then
    TOOLCHAIN=$(echo "$NDK"/toolchains/llvm/prebuilt/*/bin | awk '{print $1}')
fi
CLANG="$TOOLCHAIN/aarch64-linux-android${API_LEVEL}-clang"
AR="$TOOLCHAIN/llvm-ar"
READELF="$TOOLCHAIN/llvm-readelf"
[ -x "$CLANG" ] || { echo "NDK clang not found at $CLANG" >&2; exit 1; }
[ -x "$AR" ] || { echo "NDK llvm-ar not found at $AR" >&2; exit 1; }

sysroot=$(rustc --print sysroot)
if [ ! -d "$sysroot/lib/rustlib/$TARGET" ]; then
    echo "Rust standard library for $TARGET missing: run 'rustup target add $TARGET'" >&2
    exit 1
fi

export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$CLANG"
export CC_aarch64_linux_android="$CLANG"
export AR_aarch64_linux_android="$AR"

echo "== Building emberwave-ffi cdylib ($TARGET, $PROFILE)"
cargo build "${CARGO_PROFILE_ARGS[@]}" --locked -p emberwave-ffi --target "$TARGET"

CARGO_OUT="target/$TARGET/$PROFILE"
"$READELF" -d "$CARGO_OUT/libemberwave_ffi.so" | grep -q "Library soname: \[libemberwave_ffi.so\]" || {
    echo "expected DT_SONAME libemberwave_ffi.so on $CARGO_OUT/libemberwave_ffi.so" >&2
    exit 1
}

echo "== Building emberwave-ffi staticlib ($TARGET, $PROFILE)"
STATIC_LOG=$(mktemp)
trap 'rm -f "$STATIC_LOG"' EXIT
cargo rustc "${CARGO_PROFILE_ARGS[@]}" --locked --color never -p emberwave-ffi --lib \
    --target "$TARGET" --crate-type staticlib -- --print native-static-libs 2>&1 | tee "$STATIC_LOG"
STATIC_LIBS=$(sed -n 's/.*native-static-libs: //p' "$STATIC_LOG" | tail -n 1)
[ -n "$STATIC_LIBS" ] || { echo "failed to read native-static-libs from rustc" >&2; exit 1; }

OUT="target/android-smoke"
mkdir -p "$OUT"

echo "== Compiling Android Vulkan R16 C smoke binaries"
"$CLANG" -std=c99 -Wall -Wextra -Werror -pedantic \
    -Icrates/emberwave-ffi/include \
    crates/emberwave-ffi/examples/android/vulkan_r16.c \
    -L"$CARGO_OUT" -lemberwave_ffi -lvulkan \
    -Wl,-rpath,'$ORIGIN' \
    -o "$OUT/emberwave_vulkan_r16_shared"

# shellcheck disable=SC2086
"$CLANG" -std=c99 -Wall -Wextra -Werror -pedantic \
    -Icrates/emberwave-ffi/include \
    crates/emberwave-ffi/examples/android/vulkan_r16.c \
    "$CARGO_OUT/libemberwave_ffi.a" -lvulkan $STATIC_LIBS \
    -o "$OUT/emberwave_vulkan_r16_static"

if [ "$BUILD_ONLY" -eq 1 ]; then
    echo "Build succeeded (--build-only)."
    exit 0
fi

command -v adb >/dev/null 2>&1 || { echo "adb not found in PATH" >&2; exit 1; }
ADB=(adb)
if [ -n "$SERIAL" ]; then
    ADB=(adb -s "$SERIAL")
fi

DEVICE_DIR="/data/local/tmp/emberwave-android-smoke-$$"
cleanup_device() {
    rm -f "$STATIC_LOG"
    "${ADB[@]}" shell "rm -rf '$DEVICE_DIR'" >/dev/null 2>&1 || true
}
trap cleanup_device EXIT

"${ADB[@]}" shell "mkdir -p '$DEVICE_DIR'"
"${ADB[@]}" push \
    "$OUT/emberwave_vulkan_r16_shared" \
    "$OUT/emberwave_vulkan_r16_static" \
    "$CARGO_OUT/libemberwave_ffi.so" \
    testdata/noise_192x136_420.pwv1 \
    testdata/gradient_200x120_420.pwv1 \
    testdata/noise_131x67_444.pwv1 \
    "$DEVICE_DIR/"
"${ADB[@]}" shell "chmod +x '$DEVICE_DIR/emberwave_vulkan_r16_shared' '$DEVICE_DIR/emberwave_vulkan_r16_static'"

for vec in noise_192x136_420.pwv1 gradient_200x120_420.pwv1 noise_131x67_444.pwv1; do
    echo "== Running shared smoke on $vec"
    "${ADB[@]}" shell "LD_LIBRARY_PATH='$DEVICE_DIR' '$DEVICE_DIR/emberwave_vulkan_r16_shared' '$DEVICE_DIR/$vec'"
done

echo "== Running static smoke on noise_192x136_420.pwv1"
"${ADB[@]}" shell "'$DEVICE_DIR/emberwave_vulkan_r16_static' '$DEVICE_DIR/noise_192x136_420.pwv1'"
