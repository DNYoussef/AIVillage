# Betanet C FFI Library

Safe C bindings for the Betanet secure networking protocol, providing easy integration with C/C++ applications and Python infrastructure.

## Features

- **Async I/O**: Non-blocking network operations with internal async runtime management
- **Multiple Transports**: TCP, QUIC, Noise-XK, and Hybrid KEM support
- **Thread-Safe**: Safe concurrent access with proper synchronization
- **Error Handling**: Comprehensive error codes and thread-local error messages
- **Memory Safe**: Rust ownership system prevents memory leaks and use-after-free

## Building

### Prerequisites

- Rust 1.70+ (install from https://rustup.rs)
- CMake 3.14+ (for C examples)
- C compiler (gcc/clang on Linux/macOS, MSVC on Windows)

### Building the Rust Library

```bash
cd ffi/betanet-c
cargo build --release
```

This generates:
- **Linux**: `target/release/libbetanet_c.so`
- **macOS**: `target/release/libbetanet_c.dylib`
- **Windows**: `target/release/betanet_c.dll`
- **Header**: `include/betanet.h`

### Building C Examples

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

## API Usage

### Initialization

```c
#include <betanet.h>

// Initialize the library
BetanetResult result = betanet_init();
if (result != BETANET_RESULT_SUCCESS) {
    fprintf(stderr, "Failed to initialize: %s\n", betanet_get_last_error());
    return 1;
}
```

### Creating a Client

```c
// Configure the client
BetanetConfig config = {
    .listen_addr = NULL,
    .server_name = "example.com",
    .transport = BETANET_TRANSPORT_TCP,
    .max_connections = 1,
    .connection_timeout_secs = 30,
    .keepalive_interval_secs = 10,
    .enable_compression = 0
};

// Create client
BetanetHtxClient* client = betanet_htx_client_create(&config);
if (!client) {
    fprintf(stderr, "Failed to create client: %s\n", betanet_get_last_error());
    return 1;
}
```

### Async Connection

```c
void on_connected(void* user_data, BetanetConnectionState state) {
    if (state == BETANET_CONNECTION_STATE_CONNECTED) {
        printf("Connected successfully!\n");
    }
}

// Connect asynchronously
result = betanet_htx_client_connect_async(
    client,
    "127.0.0.1:9000",
    on_connected,
    NULL  // user_data
);
```

### Sending Data

```c
void on_send_complete(void* user_data, BetanetResult result, const char* error) {
    if (result == BETANET_RESULT_SUCCESS) {
        printf("Data sent successfully\n");
    } else {
        printf("Send failed: %s\n", error);
    }
}

// Send data asynchronously
const char* message = "Hello, Betanet!";
result = betanet_htx_client_send_async(
    client,
    (const uint8_t*)message,
    strlen(message),
    on_send_complete,
    NULL  // user_data
);
```

### Receiving Data

```c
uint8_t buffer[1024];
uint32_t received;

// Non-blocking receive
result = betanet_htx_client_recv(client, buffer, sizeof(buffer), &received);
if (result == BETANET_RESULT_SUCCESS && received > 0) {
    printf("Received %u bytes\n", received);
}
```

### Cleanup

```c
betanet_htx_client_destroy(client);
```

## Error Handling

The library provides comprehensive error handling:

```c
BetanetResult result = some_betanet_function();
if (result != BETANET_RESULT_SUCCESS) {
    const char* error_msg = betanet_get_last_error();
    fprintf(stderr, "Error %d: %s\n", result, error_msg ? error_msg : "Unknown");
}

// Clear error after handling
betanet_clear_error();
```

### Error Codes

- `BETANET_RESULT_SUCCESS` (0): Operation successful
- `BETANET_RESULT_ERROR` (1): Generic error
- `BETANET_RESULT_INVALID_PARAMETER` (2): Invalid parameter provided
- `BETANET_RESULT_NETWORK_ERROR` (3): Network operation failed
- `BETANET_RESULT_CRYPTO_ERROR` (4): Cryptographic operation failed
- `BETANET_RESULT_TIMEOUT` (5): Operation timed out
- `BETANET_RESULT_NOT_CONNECTED` (6): Not connected
- `BETANET_RESULT_ALREADY_CONNECTED` (7): Already connected
- `BETANET_RESULT_BUFFER_TOO_SMALL` (8): Buffer too small for data

## Python Integration

The library can be used from Python via ctypes:

```python
import ctypes
from ctypes import c_char_p, c_uint, c_void_p, POINTER, byref

# Load the library
lib = ctypes.CDLL('./libbetanet_c.so')  # Linux
# lib = ctypes.CDLL('./libbetanet_c.dylib')  # macOS
# lib = ctypes.WinDLL('./betanet_c.dll')  # Windows

# Define function signatures
lib.betanet_init.restype = ctypes.c_int
lib.betanet_get_version.restype = c_char_p

# Initialize
result = lib.betanet_init()
if result == 0:
    version = lib.betanet_get_version()
    print(f"Betanet version: {version.decode()}")
```

## Thread Safety

All public functions are thread-safe. The library maintains internal synchronization for:
- Connection state management
- Error message storage (thread-local)
- Async task scheduling

## Memory Management

- All allocated objects must be freed using their corresponding destroy functions
- The library handles internal memory management safely
- No manual memory management required for data buffers passed to callbacks

## Examples

See the `examples/` directory for complete examples:
- `c_echo_client.c`: Echo client implementation
- `c_echo_server.c`: Echo server implementation

## Building for Different Platforms

### Linux
```bash
cargo build --release
```

### macOS
```bash
cargo build --release
```

### Windows (MSVC)
```cmd
cargo build --release
```

### Cross-compilation
```bash
# For Android
cargo build --target aarch64-linux-android --release

# For iOS
cargo build --target aarch64-apple-ios --release
```

## SBOM Generation

Generate a Software Bill of Materials:

```bash
./generate_sbom.sh
```

This creates:
- `betanet-c-sbom.json`: JSON format SBOM
- `betanet-c-sbom.spdx`: SPDX format SBOM
- `betanet-c-dependencies.txt`: Dependency tree
- `betanet-c-sbom-report.md`: Human-readable report

## License

See LICENSE file in the repository root.

## Contributing

Please see CONTRIBUTING.md for guidelines.

## Support

For issues and questions, please use the GitHub issue tracker.
