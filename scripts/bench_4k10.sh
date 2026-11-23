#!/usr/bin/env bash
# Generates deterministic 4K (3840x2160, 4:2:0) 10-bit HDR10 PWV1 test streams
# across streaming-representative bitrates (50-350 Mbps) and runs the
# per-stage decode benchmark (CPU assembly, upload, QueueTurns, submit,
# dequant, idwt + per-level idwt, r16unorm vs r8unorm plane store).
#
# Usage:
#   scripts/bench_4k10.sh [--frames N] [--warmup-seconds SEC] [--low-power]
#                         [--adapter SUBSTR] [--android-serial SERIAL] [--out-dir DIR]
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
FRAMES=120
GEN_FRAMES=60
WARMUP_SECONDS=
GPU_ARGS=()
ANDROID_SERIAL=${EMBERWAVE_ANDROID_SERIAL:-}
OUT_DIR=$ROOT/target/bench/4k10

while [ $# -gt 0 ]; do
    case "$1" in
        --frames) FRAMES=$2; shift 2 ;;
        --warmup-seconds) WARMUP_SECONDS=$2; shift 2 ;;
        --low-power) GPU_ARGS+=(--low-power); shift ;;
        --adapter) GPU_ARGS+=(--adapter "$2"); shift 2 ;;
        --android-serial) ANDROID_SERIAL=$2; shift 2 ;;
        --out-dir) OUT_DIR=$2; shift 2 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done

if [ -z "$WARMUP_SECONDS" ]; then
    if [ -n "$ANDROID_SERIAL" ]; then
        WARMUP_SECONDS=3.0
    else
        WARMUP_SECONDS=1.0
    fi
fi

STREAMS_DIR=$ROOT/testdata/generated/4k10
mkdir -p "$STREAMS_DIR" "$OUT_DIR"

cargo build --release --locked -p emberwave-cli --manifest-path "$ROOT/Cargo.toml"
CLI=${CARGO_TARGET_DIR:-$ROOT/target}/release/emberwave-cli

gen_if_missing() {
    local name=$1
    local pattern=$2
    local mbps=$3
    local path="$STREAMS_DIR/$name.pwv1"
    if [ ! -f "$path" ]; then
        echo "== generating $name ($pattern @ ${mbps} Mbps, 3840x2160 4:2:0 10-bit HDR10)"
        "$CLI" gen-stream --out "$path" \
            --width 3840 --height 2160 --chroma 420 \
            --pattern "$pattern" --frames "$GEN_FRAMES" --fps 60 \
            --bitrate-mbps "$mbps" --matrix bt2020 --bit-depth 10 --gpu
    fi
}

gen_if_missing idle_4k10_50mbps idle 50
gen_if_missing scene_4k10_50mbps scene 50
gen_if_missing scene_4k10_150mbps scene 150
gen_if_missing scene_4k10_350mbps scene 350
gen_if_missing high_motion_4k10_350mbps high-motion 350

STREAM_LIST=(
    idle_4k10_50mbps
    scene_4k10_50mbps
    scene_4k10_150mbps
    scene_4k10_350mbps
    high_motion_4k10_350mbps
)

if [ -n "$ANDROID_SERIAL" ]; then
    TARGET=aarch64-linux-android
    API_LEVEL=${EMBERWAVE_ANDROID_API:-24}
    NDK=${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-${ANDROID_NDK_LATEST_HOME:-}}}
    [ -n "$NDK" ] || { echo "ANDROID_NDK_HOME/ANDROID_NDK_ROOT required for --android-serial" >&2; exit 1; }
    HOST_TAG=$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m)
    TOOLCHAIN="$NDK/toolchains/llvm/prebuilt/$HOST_TAG/bin"
    CLANG="$TOOLCHAIN/aarch64-linux-android${API_LEVEL}-clang"
    AR="$TOOLCHAIN/llvm-ar"
    export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$CLANG"
    export CC_aarch64_linux_android="$CLANG"
    export AR_aarch64_linux_android="$AR"

    cargo build --release --locked -p emberwave-cli --target "$TARGET" --manifest-path "$ROOT/Cargo.toml"
    ANDROID_CLI=${CARGO_TARGET_DIR:-$ROOT/target}/$TARGET/release/emberwave-cli

    ADB=(adb -s "$ANDROID_SERIAL")
    DEVICE_DIR="/data/local/tmp/emberwave-bench-4k10-$$"
    cleanup() {
        "${ADB[@]}" shell "rm -rf '$DEVICE_DIR'" >/dev/null 2>&1 || true
    }
    trap cleanup EXIT

    "${ADB[@]}" shell "mkdir -p '$DEVICE_DIR'"
    "${ADB[@]}" push "$ANDROID_CLI" "$DEVICE_DIR/emberwave-cli"
    "${ADB[@]}" shell "chmod +x '$DEVICE_DIR/emberwave-cli'"
    for s in "${STREAM_LIST[@]}"; do
        "${ADB[@]}" push "$STREAMS_DIR/$s.pwv1" "$DEVICE_DIR/$s.pwv1"
    done

    for s in "${STREAM_LIST[@]}"; do
        echo "=== Android [$s] r16 (queue + QueueTurns) ==="
        "${ADB[@]}" shell "'$DEVICE_DIR/emberwave-cli' bench '$DEVICE_DIR/$s.pwv1' --frames $FRAMES --warmup-seconds $WARMUP_SECONDS --output r16 --upload queue --queue-turns" | tee "$OUT_DIR/android_${s}_r16_queue.txt"
        "${ADB[@]}" shell "'$DEVICE_DIR/emberwave-cli' bench '$DEVICE_DIR/$s.pwv1' --frames $FRAMES --warmup-seconds $WARMUP_SECONDS --output r16 --upload queue --queue-turns --json" > "$OUT_DIR/android_${s}_r16_queue.json"
        echo "=== Android [$s] r16 (mapped) ==="
        "${ADB[@]}" shell "'$DEVICE_DIR/emberwave-cli' bench '$DEVICE_DIR/$s.pwv1' --frames $FRAMES --warmup-seconds $WARMUP_SECONDS --output r16 --upload mapped --json" > "$OUT_DIR/android_${s}_r16_mapped.json"
        echo "=== Android [$s] r8 (queue + QueueTurns) ==="
        "${ADB[@]}" shell "'$DEVICE_DIR/emberwave-cli' bench '$DEVICE_DIR/$s.pwv1' --frames $FRAMES --warmup-seconds $WARMUP_SECONDS --output r8 --upload queue --queue-turns --json" > "$OUT_DIR/android_${s}_r8_queue.json"
    done
    echo "=== Android per-level idwt [scene_4k10_150mbps] ==="
    "${ADB[@]}" shell "'$DEVICE_DIR/emberwave-cli' bench '$DEVICE_DIR/scene_4k10_150mbps.pwv1' --frames $FRAMES --warmup-seconds $WARMUP_SECONDS --output r16 --upload queue --queue-turns --per-level" | tee "$OUT_DIR/android_scene_4k10_150mbps_r16_per_level.txt"
    "${ADB[@]}" shell "'$DEVICE_DIR/emberwave-cli' bench '$DEVICE_DIR/scene_4k10_150mbps.pwv1' --frames $FRAMES --warmup-seconds $WARMUP_SECONDS --output r16 --upload queue --queue-turns --per-level --json" > "$OUT_DIR/android_scene_4k10_150mbps_r16_per_level.json"
    "${ADB[@]}" shell "'$DEVICE_DIR/emberwave-cli' bench '$DEVICE_DIR/scene_4k10_150mbps.pwv1' --frames $FRAMES --warmup-seconds $WARMUP_SECONDS --output r8 --upload queue --queue-turns --per-level --json" > "$OUT_DIR/android_scene_4k10_150mbps_r8_per_level.json"
else
    for s in "${STREAM_LIST[@]}"; do
        echo "=== Local [$s] r16 (queue + QueueTurns) ==="
        "$CLI" bench "$STREAMS_DIR/$s.pwv1" "${GPU_ARGS[@]}" --frames "$FRAMES" --warmup-seconds "$WARMUP_SECONDS" --output r16 --upload queue --queue-turns | tee "$OUT_DIR/local_${s}_r16_queue.txt"
        "$CLI" bench "$STREAMS_DIR/$s.pwv1" "${GPU_ARGS[@]}" --frames "$FRAMES" --warmup-seconds "$WARMUP_SECONDS" --output r16 --upload queue --queue-turns --json > "$OUT_DIR/local_${s}_r16_queue.json"
        echo "=== Local [$s] r16 (mapped) ==="
        "$CLI" bench "$STREAMS_DIR/$s.pwv1" "${GPU_ARGS[@]}" --frames "$FRAMES" --warmup-seconds "$WARMUP_SECONDS" --output r16 --upload mapped --json > "$OUT_DIR/local_${s}_r16_mapped.json"
        echo "=== Local [$s] r8 (queue + QueueTurns) ==="
        "$CLI" bench "$STREAMS_DIR/$s.pwv1" "${GPU_ARGS[@]}" --frames "$FRAMES" --warmup-seconds "$WARMUP_SECONDS" --output r8 --upload queue --queue-turns --json > "$OUT_DIR/local_${s}_r8_queue.json"
    done
    echo "=== Local per-level idwt [scene_4k10_150mbps] ==="
    "$CLI" bench "$STREAMS_DIR/scene_4k10_150mbps.pwv1" "${GPU_ARGS[@]}" --frames "$FRAMES" --warmup-seconds "$WARMUP_SECONDS" --output r16 --upload queue --queue-turns --per-level | tee "$OUT_DIR/local_scene_4k10_150mbps_r16_per_level.txt"
    "$CLI" bench "$STREAMS_DIR/scene_4k10_150mbps.pwv1" "${GPU_ARGS[@]}" --frames "$FRAMES" --warmup-seconds "$WARMUP_SECONDS" --output r16 --upload queue --queue-turns --per-level --json > "$OUT_DIR/local_scene_4k10_150mbps_r16_per_level.json"
    "$CLI" bench "$STREAMS_DIR/scene_4k10_150mbps.pwv1" "${GPU_ARGS[@]}" --frames "$FRAMES" --warmup-seconds "$WARMUP_SECONDS" --output r8 --upload queue --queue-turns --per-level --json > "$OUT_DIR/local_scene_4k10_150mbps_r8_per_level.json"
fi
