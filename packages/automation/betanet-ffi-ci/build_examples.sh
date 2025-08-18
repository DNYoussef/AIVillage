#!/bin/bash
#
# Build script for Betanet FFI examples
# Demonstrates how to compile C programs against the Betanet FFI library
#

set -e

echo "=== Building Betanet FFI Library and Examples ==="
echo

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FFI_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$FFI_DIR")")"

echo "FFI Directory: $FFI_DIR"
echo "Project Root: $PROJECT_ROOT"
echo

# Build Rust FFI library first
echo "Building Rust FFI library..."
cd "$PROJECT_ROOT"
cargo build --package betanet-ffi --release

# Check if build succeeded
if [ ! -f "target/release/libbetanet_ffi.so" ] && [ ! -f "target/release/libbetanet_ffi.dylib" ] && [ ! -f "target/release/betanet_ffi.dll" ]; then
    echo "Error: FFI library not found after build"
    exit 1
fi

echo "✓ FFI library built successfully"
echo

# Generate C headers
echo "Generating C headers..."
cd "$FFI_DIR"
if [ -f "include/betanet.h" ]; then
    echo "✓ Header file found: include/betanet.h"
else
    echo "Warning: Header file not found, cbindgen may have failed"
fi
echo

# Detect library file extension
if [ -f "$PROJECT_ROOT/target/release/libbetanet_ffi.so" ]; then
    LIB_FILE="$PROJECT_ROOT/target/release/libbetanet_ffi.so"
    LIB_EXT="so"
elif [ -f "$PROJECT_ROOT/target/release/libbetanet_ffi.dylib" ]; then
    LIB_FILE="$PROJECT_ROOT/target/release/libbetanet_ffi.dylib"
    LIB_EXT="dylib"
elif [ -f "$PROJECT_ROOT/target/release/betanet_ffi.dll" ]; then
    LIB_FILE="$PROJECT_ROOT/target/release/betanet_ffi.dll"
    LIB_EXT="dll"
else
    echo "Error: No FFI library file found"
    exit 1
fi

echo "Using library: $LIB_FILE"
echo

# Compile examples
cd "$FFI_DIR"
mkdir -p build

echo "Compiling examples..."

# Compiler flags
CFLAGS="-Wall -Wextra -std=c99 -I./include"
LDFLAGS="-L$PROJECT_ROOT/target/release"

# Link flags vary by platform
case "$LIB_EXT" in
    "so")
        LDLIBS="-lbetanet_ffi -lpthread -ldl -lm"
        ;;
    "dylib")
        LDLIBS="-lbetanet_ffi -lpthread -ldl -lm"
        ;;
    "dll")
        LDLIBS="-lbetanet_ffi -lws2_32 -luserenv -lbcrypt"
        ;;
esac

# Build each example
EXAMPLES=("htx_example" "utls_example" "mixnode_example" "linter_example")

for example in "${EXAMPLES[@]}"; do
    echo "Building $example..."

    if gcc $CFLAGS $LDFLAGS "examples/${example}.c" $LDLIBS -o "build/${example}"; then
        echo "✓ $example compiled successfully"
    else
        echo "✗ Failed to compile $example"
        exit 1
    fi
done

echo
echo "=== Build Complete ==="
echo
echo "Compiled examples:"
ls -la build/
echo
echo "To run examples:"
echo "  export LD_LIBRARY_PATH=$PROJECT_ROOT/target/release:\$LD_LIBRARY_PATH"
echo "  ./build/htx_example"
echo "  ./build/utls_example"
echo "  ./build/mixnode_example"
echo "  ./build/linter_example"
echo
