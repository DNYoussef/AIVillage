# Betanet FFI Makefile
# Cross-platform build system for FFI library and examples

# Default target
.PHONY: all clean build test examples install

# Detect OS
UNAME_S := $(shell uname -s 2>/dev/null || echo Windows)

# Set library extension based on OS
ifeq ($(UNAME_S),Linux)
    LIB_EXT = so
    LIB_PREFIX = lib
    LDLIBS = -lbetanet_ffi -lpthread -ldl -lm
endif
ifeq ($(UNAME_S),Darwin)
    LIB_EXT = dylib
    LIB_PREFIX = lib
    LDLIBS = -lbetanet_ffi -lpthread -ldl -lm
endif
ifeq ($(UNAME_S),Windows)
    LIB_EXT = dll
    LIB_PREFIX =
    LDLIBS = -lbetanet_ffi -lws2_32 -luserenv -lbcrypt
endif

# Paths
PROJECT_ROOT = ../..
TARGET_DIR = $(PROJECT_ROOT)/target/release
LIB_FILE = $(TARGET_DIR)/$(LIB_PREFIX)betanet_ffi.$(LIB_EXT)
HEADER_FILE = include/betanet.h

# Compiler settings
CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -I./include
LDFLAGS = -L$(TARGET_DIR)

# Examples
EXAMPLES = htx_example utls_example mixnode_example linter_example
EXAMPLE_SOURCES = $(addprefix examples/,$(addsuffix .c,$(EXAMPLES)))
EXAMPLE_BINARIES = $(addprefix build/,$(EXAMPLES))

# Default target
all: build examples

# Build the Rust FFI library
build:
	@echo "Building Betanet FFI library..."
	cd $(PROJECT_ROOT) && cargo build --package betanet-ffi --release
	@echo "✓ FFI library built: $(LIB_FILE)"
	@echo "✓ Header file: $(HEADER_FILE)"

# Test the FFI library
test: build
	@echo "Testing Betanet FFI library..."
	cd $(PROJECT_ROOT) && cargo test --package betanet-ffi
	@echo "✓ FFI tests passed"

# Build examples
examples: build $(EXAMPLE_BINARIES)
	@echo "✓ All examples built successfully"

# Create build directory
build:
	@mkdir -p build

# Build individual examples
build/%: examples/%.c build
	@echo "Building $*..."
	$(CC) $(CFLAGS) $(LDFLAGS) $< $(LDLIBS) -o $@

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	cd $(PROJECT_ROOT) && cargo clean
	@echo "✓ Clean complete"

# Install library (Unix-like systems)
install: build
ifeq ($(UNAME_S),Windows)
	@echo "Install target not supported on Windows"
	@exit 1
else
	@echo "Installing Betanet FFI library..."
	sudo cp $(LIB_FILE) /usr/local/lib/
	sudo cp $(HEADER_FILE) /usr/local/include/
	sudo ldconfig
	@echo "✓ Library installed to /usr/local"
endif

# Run examples
run-examples: examples
	@echo "Running examples..."
	@echo "=== HTX Example ==="
	LD_LIBRARY_PATH=$(TARGET_DIR):$$LD_LIBRARY_PATH ./build/htx_example || true
	@echo ""
	@echo "=== uTLS Example ==="
	LD_LIBRARY_PATH=$(TARGET_DIR):$$LD_LIBRARY_PATH ./build/utls_example || true
	@echo ""
	@echo "=== Mixnode Example ==="
	LD_LIBRARY_PATH=$(TARGET_DIR):$$LD_LIBRARY_PATH ./build/mixnode_example || true
	@echo ""
	@echo "=== Linter Example ==="
	LD_LIBRARY_PATH=$(TARGET_DIR):$$LD_LIBRARY_PATH ./build/linter_example || true

# Check library symbols
check-symbols: build
ifeq ($(UNAME_S),Windows)
	@echo "Symbol checking not implemented for Windows"
else
	@echo "Checking exported symbols..."
	@if command -v nm >/dev/null 2>&1; then \
		echo "Using nm:"; \
		nm -D $(LIB_FILE) | grep betanet || echo "No betanet symbols found"; \
	elif command -v objdump >/dev/null 2>&1; then \
		echo "Using objdump:"; \
		objdump -T $(LIB_FILE) | grep betanet || echo "No betanet symbols found"; \
	else \
		echo "No symbol inspection tool available"; \
	fi
endif

# Development target - rebuild and test
dev: clean build test examples
	@echo "✓ Development build complete"

# Release target - build optimized
release:
	@echo "Building release version..."
	cd $(PROJECT_ROOT) && cargo build --package betanet-ffi --release
	$(MAKE) examples
	@echo "✓ Release build complete"

# Help target
help:
	@echo "Betanet FFI Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build library and examples (default)"
	@echo "  build        - Build Rust FFI library"
	@echo "  test         - Run FFI tests"
	@echo "  examples     - Build C examples"
	@echo "  clean        - Clean build artifacts"
	@echo "  install      - Install library system-wide (Unix only)"
	@echo "  run-examples - Build and run all examples"
	@echo "  check-symbols- Check exported symbols"
	@echo "  dev          - Development build (clean + build + test)"
	@echo "  release      - Release build"
	@echo "  help         - Show this help"
	@echo ""
	@echo "Environment:"
	@echo "  OS: $(UNAME_S)"
	@echo "  Library: $(LIB_FILE)"
	@echo "  Header: $(HEADER_FILE)"
