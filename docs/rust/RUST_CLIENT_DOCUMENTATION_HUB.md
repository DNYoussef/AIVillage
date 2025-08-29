# Rust Client Documentation Hub

## Overview

The AIVillage ecosystem includes 30+ production-ready Rust crates that provide high-performance, memory-safe implementations of core networking, cryptographic, and distributed computing functionality. This documentation hub serves as the central guide for deploying, configuring, and using these Rust components.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                 Rust Client Ecosystem                  │
├─────────────────────────────────────────────────────────┤
│  Network Layer                                         │
│  ├── BetaNet (Mixnode networking)                      │
│  ├── BitChat (P2P messaging)                           │
│  ├── DTN (Delay-tolerant networking)                   │
│  └── UTLS (Transport layer security)                   │
├─────────────────────────────────────────────────────────┤
│  Agent Layer                                           │
│  ├── Agent Fabric (Multi-agent coordination)          │
│  ├── Navigator (Pathfinding & routing)                 │
│  └── Twin Vault (Digital twin storage)                 │
├─────────────────────────────────────────────────────────┤
│  Infrastructure Layer                                   │
│  ├── Federated Learning                                │
│  ├── Linting & Quality Tools                          │
│  └── FFI Bridges (C/Python integration)               │
├─────────────────────────────────────────────────────────┤
│  Application Layer                                     │
│  ├── Gateway Clients                                  │
│  ├── Benchmarking Tools                               │
│  └── Testing Frameworks                               │
└─────────────────────────────────────────────────────────┘
```

## Crate Categories and Index

### Network Communication Crates

#### BetaNet Ecosystem
- **`betanet-client`** - Core BetaNet client library
- **`betanet-gateway`** - Gateway services for BetaNet integration
- **`betanet-mixnode`** - Mixnode implementation for anonymity
- **`betanet-htx`** - High-throughput extensions
- **`betanet-utls`** - Transport layer security
- **`betanet-dtn`** - Delay-tolerant networking
- **`betanet-lint`** - Code quality tools for BetaNet
- **`betanet-cla`** - Convergence Layer Adapter

#### BitChat Communication
- **`bitchat-cla`** - BitChat Convergence Layer Adapter
- **`libbetanet`** - Low-level BetaNet library

### Agent Infrastructure

#### Multi-Agent Systems
- **`agent-fabric`** - Distributed agent coordination framework
- **`navigator`** - Pathfinding and routing for agent networks
- **`twin-vault`** - Secure storage for digital twin data

#### Federated Computing
- **`federated`** - Federated learning infrastructure

### Development Tools

#### Code Quality & Linting
- **`betanet-linter`** - BetaNet-specific code analysis
- **`utlsgen`** - UTLS configuration generator

#### FFI Integration
- **`betanet-c`** - C language bindings
- **`betanet-c-unified`** - Unified C interface
- **`betanet-ffi`** - Foreign Function Interface utilities

#### Testing & Benchmarking
- **`bench`** - Performance benchmarking tools
- **`config`** - Configuration management utilities

## Quick Start Guide

### Prerequisites

```bash
# Rust toolchain (1.70+ required)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Additional tools
rustup component add clippy rustfmt
rustup target add wasm32-unknown-unknown  # For WebAssembly builds

# System dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install build-essential pkg-config libssl-dev libudev-dev
```

### Installation and Setup

```bash
# Clone the repository
git clone https://github.com/your-org/AIVillage.git
cd AIVillage

# Navigate to Rust clients directory
cd integrations/clients/rust

# Build all crates
cargo build --release --all

# Run tests
cargo test --all

# Install common development tools
cargo install cargo-watch cargo-audit cargo-outdated
```

## Core Crates Documentation

### BetaNet Client (`betanet-client`)

**Purpose**: Core networking client for BetaNet mixnode-based anonymous communication.

**Location**: `integrations/clients/rust/betanet/betanet-client/`

#### Key Features
- Anonymous message routing through mixnodes
- End-to-end encryption
- Circuit-based communication paths
- Configurable anonymity levels
- Tor-like onion routing

#### Basic Usage

```rust
// Cargo.toml
[dependencies]
betanet-client = { path = "../betanet-client" }
tokio = { version = "1.0", features = ["full"] }

// main.rs
use betanet_client::{BetaNetClient, ClientConfig, CircuitConfig};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize client configuration
    let config = ClientConfig {
        mixnode_endpoints: vec![
            "mix1.betanet.ai:9443".to_string(),
            "mix2.betanet.ai:9443".to_string(),
            "mix3.betanet.ai:9443".to_string(),
        ],
        circuit_length: 3,
        connection_timeout: Duration::from_secs(10),
        retry_attempts: 3,
    };

    // Create client
    let mut client = BetaNetClient::new(config).await?;

    // Connect to network
    client.connect().await?;
    println!("Connected to BetaNet");

    // Create circuit for anonymous communication
    let circuit_config = CircuitConfig {
        hops: 3,
        anonymity_level: "high".to_string(),
    };
    let circuit_id = client.create_circuit(circuit_config).await?;

    // Send anonymous message
    let message = b"Hello BetaNet!";
    let response = client.send_message(circuit_id, message).await?;
    println!("Response: {:?}", response);

    // Clean up
    client.close_circuit(circuit_id).await?;
    client.disconnect().await?;

    Ok(())
}
```

#### Configuration Options

```toml
# betanet-config.toml
[network]
mixnode_endpoints = [
    "mix1.betanet.ai:9443",
    "mix2.betanet.ai:9443",
    "mix3.betanet.ai:9443"
]
circuit_length = 3
connection_timeout_ms = 10000
retry_attempts = 3

[security]
anonymity_level = "high"  # low, medium, high, maximum
encryption_algorithm = "ChaCha20Poly1305"
key_exchange = "X25519"

[performance]
max_concurrent_circuits = 10
circuit_idle_timeout_ms = 300000  # 5 minutes
buffer_size = 4096
compression_enabled = true

[logging]
level = "info"
file = "betanet.log"
max_size_mb = 100
```

### BitChat CLA (`bitchat-cla`)

**Purpose**: Convergence Layer Adapter for BitChat peer-to-peer messaging.

**Location**: `integrations/clients/rust/bitchat-cla/`

#### Key Features
- Direct peer-to-peer messaging
- Offline message storage and forwarding
- Mobile-optimized protocols
- Battery-efficient operation
- Store-and-forward capabilities

#### Basic Usage

```rust
use bitchat_cla::{BitChatCLA, MessageConfig, PeerConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize BitChat CLA
    let peer_config = PeerConfig {
        peer_id: "my_peer_001".to_string(),
        listen_port: 7777,
        discovery_enabled: true,
        store_and_forward: true,
    };

    let mut bitchat = BitChatCLA::new(peer_config).await?;

    // Start peer discovery
    bitchat.start_discovery().await?;
    println!("BitChat peer discovery started");

    // Wait for peers to be discovered
    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
    let peers = bitchat.get_discovered_peers().await?;
    println!("Discovered {} peers", peers.len());

    // Send message to peer
    if let Some(peer) = peers.first() {
        let message_config = MessageConfig {
            recipient: peer.peer_id.clone(),
            content: b"Hello from BitChat!".to_vec(),
            priority: "normal".to_string(),
            ttl_hours: 24,
        };

        let message_id = bitchat.send_message(message_config).await?;
        println!("Message sent with ID: {}", message_id);

        // Check for responses
        if let Some(response) = bitchat.receive_message().await? {
            println!("Received: {:?}", String::from_utf8_lossy(&response.content));
        }
    }

    // Clean shutdown
    bitchat.stop_discovery().await?;
    bitchat.shutdown().await?;

    Ok(())
}
```

### Agent Fabric (`agent-fabric`)

**Purpose**: Distributed multi-agent coordination and management framework.

**Location**: `integrations/clients/rust/agent-fabric/`

#### Key Features
- Distributed agent lifecycle management
- Service discovery and registration
- Load balancing and failover
- Inter-agent communication
- Resource optimization

#### Basic Usage

```rust
use agent_fabric::{AgentFabric, AgentConfig, ServiceDefinition};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize agent fabric
    let fabric_config = AgentConfig {
        node_id: "fabric_node_001".to_string(),
        cluster_endpoints: vec![
            "fabric://10.0.1.100:8880".to_string(),
            "fabric://10.0.1.101:8880".to_string(),
        ],
        max_agents: 50,
        resource_limits: ResourceLimits {
            cpu_cores: 4,
            memory_gb: 8,
        },
    };

    let mut fabric = AgentFabric::new(fabric_config).await?;

    // Start fabric node
    fabric.start().await?;
    println!("Agent fabric node started");

    // Register a service
    let service_def = ServiceDefinition {
        name: "data-processor".to_string(),
        version: "1.0.0".to_string(),
        endpoints: vec!["process_data".to_string()],
        resource_requirements: ResourceRequirements {
            cpu_cores: 1,
            memory_mb: 512,
        },
    };

    fabric.register_service(service_def).await?;

    // Discover available services
    let services = fabric.discover_services("data-*").await?;
    println!("Available services: {:?}", services);

    // Create and deploy agent
    let agent_spec = AgentSpec {
        name: "processor_agent_001".to_string(),
        service: "data-processor".to_string(),
        replicas: 2,
        placement_strategy: "balanced".to_string(),
    };

    let agent_id = fabric.deploy_agent(agent_spec).await?;
    println!("Deployed agent: {}", agent_id);

    // Monitor agent status
    let status = fabric.get_agent_status(&agent_id).await?;
    println!("Agent status: {:?}", status);

    // Send message to agent
    let message = AgentMessage {
        to: agent_id.clone(),
        from: "fabric_node_001".to_string(),
        content: serde_json::json!({
            "command": "process",
            "data": [1, 2, 3, 4, 5]
        }),
    };

    let response = fabric.send_message(message).await?;
    println!("Agent response: {:?}", response);

    // Clean shutdown
    fabric.undeploy_agent(&agent_id).await?;
    fabric.stop().await?;

    Ok(())
}
```

### Twin Vault (`twin-vault`)

**Purpose**: Secure storage and management for digital twin data and models.

**Location**: `integrations/clients/rust/twin-vault/`

#### Key Features
- Encrypted digital twin storage
- Versioned model management
- Secure multi-tenant access
- Backup and synchronization
- Privacy-preserving operations

#### Basic Usage

```rust
use twin_vault::{TwinVault, VaultConfig, DigitalTwin, AccessPolicy};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize vault
    let vault_config = VaultConfig {
        storage_path: "./twin_vault_data".to_string(),
        encryption_key: "your-32-byte-encryption-key-here".to_string(),
        backup_enabled: true,
        compression: true,
    };

    let mut vault = TwinVault::new(vault_config).await?;

    // Create digital twin
    let twin = DigitalTwin {
        id: "user_12345_twin".to_string(),
        owner: "user_12345".to_string(),
        version: "1.0.0".to_string(),
        data: serde_json::json!({
            "preferences": {
                "communication_style": "friendly",
                "interests": ["technology", "science", "music"]
            },
            "behavior_patterns": {
                "active_hours": [8, 22],
                "response_speed": "moderate"
            }
        }),
        privacy_level: "high".to_string(),
        last_updated: chrono::Utc::now(),
    };

    // Store twin
    vault.store_twin(twin).await?;
    println!("Digital twin stored securely");

    // Retrieve twin
    let retrieved_twin = vault.get_twin("user_12345_twin").await?;
    println!("Retrieved twin: {:?}", retrieved_twin.data);

    // Update twin with versioning
    let mut updated_twin = retrieved_twin.unwrap();
    updated_twin.version = "1.1.0".to_string();
    updated_twin.data["behavior_patterns"]["last_interaction"] =
        serde_json::json!(chrono::Utc::now().timestamp());

    vault.update_twin(updated_twin).await?;

    // Set access policy
    let access_policy = AccessPolicy {
        twin_id: "user_12345_twin".to_string(),
        allowed_operations: vec!["read".to_string(), "update".to_string()],
        allowed_users: vec!["user_12345".to_string()],
        expiry: Some(chrono::Utc::now() + chrono::Duration::days(30)),
    };

    vault.set_access_policy(access_policy).await?;

    // List all twins for user
    let user_twins = vault.list_twins_for_user("user_12345").await?;
    println!("User has {} digital twins", user_twins.len());

    // Create backup
    let backup_path = vault.create_backup().await?;
    println!("Backup created: {}", backup_path);

    Ok(())
}
```

### DTN (Delay-Tolerant Networking) (`betanet-dtn`)

**Purpose**: Delay-tolerant networking for intermittent connectivity scenarios.

**Location**: `integrations/clients/rust/betanet-dtn/`

#### Key Features
- Store-and-forward messaging
- Opportunistic networking
- Bundle routing protocols
- Congestion control
- Multi-hop delivery

#### Basic Usage

```rust
use betanet_dtn::{DTNNode, NodeConfig, Bundle, RoutingStrategy};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize DTN node
    let node_config = NodeConfig {
        node_id: "dtn_node_001".to_string(),
        storage_path: "./dtn_storage".to_string(),
        max_storage_gb: 1.0,
        routing_strategy: RoutingStrategy::SprayAndWait { copies: 5 },
        contact_discovery: true,
    };

    let mut dtn_node = DTNNode::new(node_config).await?;

    // Start node
    dtn_node.start().await?;
    println!("DTN node started");

    // Create and send bundle
    let bundle = Bundle {
        source: "dtn_node_001".to_string(),
        destination: "dtn_node_002".to_string(),
        payload: b"Important message that must be delivered".to_vec(),
        ttl: chrono::Duration::hours(24),
        priority: "high".to_string(),
        delivery_receipt_requested: true,
    };

    let bundle_id = dtn_node.send_bundle(bundle).await?;
    println!("Bundle sent: {}", bundle_id);

    // Check bundle status
    let status = dtn_node.get_bundle_status(&bundle_id).await?;
    println!("Bundle status: {:?}", status);

    // Listen for incoming bundles
    while let Some(received_bundle) = dtn_node.receive_bundle().await? {
        println!("Received bundle from: {}", received_bundle.source);
        println!("Payload: {:?}", String::from_utf8_lossy(&received_bundle.payload));

        // Send delivery receipt if requested
        if received_bundle.delivery_receipt_requested {
            dtn_node.send_delivery_receipt(&received_bundle).await?;
        }
    }

    // Clean shutdown
    dtn_node.stop().await?;

    Ok(())
}
```

### Navigator (`navigator`)

**Purpose**: Pathfinding and routing services for distributed agent networks.

**Location**: `integrations/clients/rust/navigator/`

#### Key Features
- Graph-based pathfinding algorithms
- Dynamic route optimization
- Load-aware routing
- Multi-constraint pathfinding
- Network topology management

#### Basic Usage

```rust
use navigator::{Navigator, NetworkGraph, RoutingRequest, PathConstraints};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize navigator
    let mut navigator = Navigator::new().await?;

    // Build network topology
    let mut graph = NetworkGraph::new();

    // Add nodes
    graph.add_node("node_a", NodeMetadata {
        location: (0.0, 0.0),
        capacity: 100,
        current_load: 20,
    });
    graph.add_node("node_b", NodeMetadata {
        location: (10.0, 0.0),
        capacity: 80,
        current_load: 60,
    });
    graph.add_node("node_c", NodeMetadata {
        location: (5.0, 8.0),
        capacity: 120,
        current_load: 30,
    });

    // Add edges with metrics
    graph.add_edge("node_a", "node_b", EdgeMetrics {
        latency_ms: 50,
        bandwidth_mbps: 100,
        reliability: 0.95,
        cost: 1.0,
    });
    graph.add_edge("node_b", "node_c", EdgeMetrics {
        latency_ms: 30,
        bandwidth_mbps: 200,
        reliability: 0.98,
        cost: 1.5,
    });
    graph.add_edge("node_a", "node_c", EdgeMetrics {
        latency_ms: 80,
        bandwidth_mbps: 50,
        reliability: 0.90,
        cost: 2.0,
    });

    // Load topology into navigator
    navigator.update_topology(graph).await?;

    // Find optimal path
    let routing_request = RoutingRequest {
        source: "node_a".to_string(),
        destination: "node_c".to_string(),
        constraints: PathConstraints {
            max_latency_ms: Some(100),
            min_bandwidth_mbps: Some(75),
            min_reliability: Some(0.92),
            max_hops: Some(3),
        },
        optimization_goal: "minimize_latency".to_string(),
    };

    let path = navigator.find_path(routing_request).await?;
    println!("Optimal path: {:?}", path.nodes);
    println!("Total latency: {}ms", path.total_latency);
    println!("Total cost: {}", path.total_cost);

    // Find multiple paths for load balancing
    let paths = navigator.find_k_shortest_paths("node_a", "node_c", 3).await?;
    for (i, path) in paths.iter().enumerate() {
        println!("Path {}: {:?} (cost: {})", i + 1, path.nodes, path.total_cost);
    }

    // Real-time route monitoring
    let route_monitor = navigator.create_route_monitor(path.clone()).await?;

    // Check route health
    loop {
        let health = route_monitor.check_health().await?;
        println!("Route health: {:?}", health);

        if health.status != "healthy" {
            println!("Route degraded, finding alternative...");
            let alternative = navigator.find_alternative_path(&path, &routing_request).await?;
            println!("Alternative path: {:?}", alternative.nodes);
            break;
        }

        tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;
    }

    Ok(())
}
```

## FFI Integration Guide

### C Language Bindings (`betanet-c`)

For integrating Rust functionality into C/C++ applications:

**Location**: `integrations/clients/rust/betanet-ffi/` and `build/core-build/ffi/betanet-c/`

#### Building C Bindings

```bash
# Navigate to FFI crate
cd integrations/clients/rust/betanet-ffi

# Build shared library
cargo build --release --lib

# Generate C header file
cargo install cbindgen
cbindgen --output betanet.h

# Build example C program
gcc -o example example.c -L target/release -lbetanet_ffi -lpthread -ldl
```

#### C API Usage

```c
// betanet_example.c
#include "betanet.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    // Initialize BetaNet client
    BetaNetConfig config = {
        .mixnode_endpoints = {"mix1.betanet.ai:9443", "mix2.betanet.ai:9443"},
        .mixnode_count = 2,
        .circuit_length = 3,
        .timeout_ms = 10000
    };

    BetaNetClient* client = betanet_client_new(&config);
    if (!client) {
        fprintf(stderr, "Failed to create BetaNet client\\n");
        return 1;
    }

    // Connect to network
    if (betanet_client_connect(client) != 0) {
        fprintf(stderr, "Failed to connect to BetaNet\\n");
        betanet_client_free(client);
        return 1;
    }

    printf("Connected to BetaNet successfully\\n");

    // Create circuit
    uint64_t circuit_id;
    if (betanet_client_create_circuit(client, &circuit_id) != 0) {
        fprintf(stderr, "Failed to create circuit\\n");
        betanet_client_disconnect(client);
        betanet_client_free(client);
        return 1;
    }

    printf("Created circuit: %llu\\n", circuit_id);

    // Send message
    const char* message = "Hello from C!";
    uint8_t response[1024];
    size_t response_len = sizeof(response);

    if (betanet_client_send_message(client, circuit_id,
                                   (const uint8_t*)message, strlen(message),
                                   response, &response_len) == 0) {
        printf("Response received: %.*s\\n", (int)response_len, response);
    } else {
        fprintf(stderr, "Failed to send message\\n");
    }

    // Cleanup
    betanet_client_close_circuit(client, circuit_id);
    betanet_client_disconnect(client);
    betanet_client_free(client);

    return 0;
}
```

#### CMake Integration

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.16)
project(BetaNetExample)

set(CMAKE_C_STANDARD 11)

# Find Rust library
find_library(BETANET_LIB NAMES betanet_ffi
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../integrations/clients/rust/betanet-ffi/target/release
    NO_DEFAULT_PATH)

if(NOT BETANET_LIB)
    message(FATAL_ERROR "BetaNet FFI library not found. Build it first with: cargo build --release")
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../integrations/clients/rust/betanet-ffi)

# Create executable
add_executable(betanet_example betanet_example.c)

# Link libraries
target_link_libraries(betanet_example ${BETANET_LIB} pthread dl)

# Custom target to build Rust library first
add_custom_target(build_rust_lib
    COMMAND cargo build --release
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../integrations/clients/rust/betanet-ffi
    COMMENT "Building Rust FFI library"
)

add_dependencies(betanet_example build_rust_lib)
```

### Python Integration

```python
# python_integration.py
import ctypes
import os
from typing import Optional

# Load the Rust library
lib_path = "target/release/libbetanet_ffi.so"  # Linux
# lib_path = "target/release/libbetanet_ffi.dylib"  # macOS
# lib_path = "target/release/betanet_ffi.dll"  # Windows

if not os.path.exists(lib_path):
    raise RuntimeError(f"Rust library not found at {lib_path}. Build it first with 'cargo build --release'")

lib = ctypes.CDLL(lib_path)

# Define C structures in Python
class BetaNetConfig(ctypes.Structure):
    _fields_ = [
        ("mixnode_endpoints", ctypes.POINTER(ctypes.c_char_p)),
        ("mixnode_count", ctypes.c_size_t),
        ("circuit_length", ctypes.c_uint32),
        ("timeout_ms", ctypes.c_uint32),
    ]

# Define function signatures
lib.betanet_client_new.argtypes = [ctypes.POINTER(BetaNetConfig)]
lib.betanet_client_new.restype = ctypes.c_void_p

lib.betanet_client_connect.argtypes = [ctypes.c_void_p]
lib.betanet_client_connect.restype = ctypes.c_int

lib.betanet_client_send_message.argtypes = [
    ctypes.c_void_p,  # client
    ctypes.c_uint64,  # circuit_id
    ctypes.POINTER(ctypes.c_uint8),  # message
    ctypes.c_size_t,  # message_len
    ctypes.POINTER(ctypes.c_uint8),  # response
    ctypes.POINTER(ctypes.c_size_t),  # response_len
]
lib.betanet_client_send_message.restype = ctypes.c_int

lib.betanet_client_free.argtypes = [ctypes.c_void_p]
lib.betanet_client_free.restype = None

class BetaNetClient:
    def __init__(self, mixnode_endpoints: list[str], circuit_length: int = 3, timeout_ms: int = 10000):
        # Convert Python strings to C strings
        self.endpoints = [endpoint.encode('utf-8') for endpoint in mixnode_endpoints]
        endpoint_ptrs = (ctypes.c_char_p * len(self.endpoints))()
        for i, endpoint in enumerate(self.endpoints):
            endpoint_ptrs[i] = endpoint

        # Create configuration
        config = BetaNetConfig(
            mixnode_endpoints=endpoint_ptrs,
            mixnode_count=len(mixnode_endpoints),
            circuit_length=circuit_length,
            timeout_ms=timeout_ms,
        )

        # Create client
        self.client = lib.betanet_client_new(ctypes.byref(config))
        if not self.client:
            raise RuntimeError("Failed to create BetaNet client")

    def connect(self) -> bool:
        result = lib.betanet_client_connect(self.client)
        return result == 0

    def send_message(self, circuit_id: int, message: bytes) -> Optional[bytes]:
        message_buffer = (ctypes.c_uint8 * len(message))()
        for i, byte in enumerate(message):
            message_buffer[i] = byte

        response_buffer = (ctypes.c_uint8 * 4096)()
        response_len = ctypes.c_size_t(4096)

        result = lib.betanet_client_send_message(
            self.client,
            circuit_id,
            message_buffer,
            len(message),
            response_buffer,
            ctypes.byref(response_len)
        )

        if result == 0:
            return bytes(response_buffer[:response_len.value])
        return None

    def __del__(self):
        if hasattr(self, 'client') and self.client:
            lib.betanet_client_free(self.client)

# Usage example
def main():
    client = BetaNetClient([
        "mix1.betanet.ai:9443",
        "mix2.betanet.ai:9443",
        "mix3.betanet.ai:9443"
    ])

    if client.connect():
        print("Connected to BetaNet")

        # In a real implementation, you'd create a circuit first
        circuit_id = 12345  # This would come from circuit creation
        response = client.send_message(circuit_id, b"Hello from Python!")

        if response:
            print(f"Response: {response.decode('utf-8')}")
    else:
        print("Failed to connect to BetaNet")

if __name__ == "__main__":
    main()
```

## Configuration Management

### Global Configuration

Create a unified configuration file for all Rust components:

```toml
# rust_clients_config.toml
[global]
log_level = "info"
data_directory = "./aivillage_data"
temp_directory = "/tmp/aivillage"

[network]
# BetaNet configuration
[network.betanet]
mixnode_endpoints = [
    "mix1.betanet.ai:9443",
    "mix2.betanet.ai:9443",
    "mix3.betanet.ai:9443"
]
circuit_length = 3
connection_timeout_ms = 10000
max_concurrent_circuits = 50

# BitChat configuration
[network.bitchat]
listen_port = 7777
discovery_enabled = true
store_and_forward = true
peer_timeout_ms = 30000
max_stored_messages = 1000

# DTN configuration
[network.dtn]
storage_path = "./dtn_storage"
max_storage_gb = 5.0
routing_strategy = "spray_and_wait"
routing_copies = 5

[agents]
# Agent Fabric configuration
[agents.fabric]
cluster_endpoints = [
    "fabric://10.0.1.100:8880",
    "fabric://10.0.1.101:8880"
]
max_agents = 100
heartbeat_interval_ms = 5000

# Navigator configuration
[agents.navigator]
update_interval_ms = 1000
pathfinding_algorithm = "dijkstra"
max_path_length = 10

[storage]
# Twin Vault configuration
[storage.twin_vault]
storage_path = "./twin_vault_data"
encryption_enabled = true
backup_enabled = true
compression_enabled = true
max_vault_size_gb = 10.0

[security]
encryption_algorithm = "ChaCha20Poly1305"
key_derivation = "Argon2"
secure_random_source = "/dev/urandom"

[performance]
worker_threads = 0  # 0 = auto-detect
max_blocking_threads = 512
thread_stack_size_kb = 2048
```

### Configuration Loading

```rust
// config.rs - Shared configuration management
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct RustClientsConfig {
    pub global: GlobalConfig,
    pub network: NetworkConfig,
    pub agents: AgentConfig,
    pub storage: StorageConfig,
    pub security: SecurityConfig,
    pub performance: PerformanceConfig,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct GlobalConfig {
    pub log_level: String,
    pub data_directory: String,
    pub temp_directory: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct NetworkConfig {
    pub betanet: BetaNetConfig,
    pub bitchat: BitChatConfig,
    pub dtn: DtnConfig,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct BetaNetConfig {
    pub mixnode_endpoints: Vec<String>,
    pub circuit_length: u32,
    pub connection_timeout_ms: u64,
    pub max_concurrent_circuits: usize,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct BitChatConfig {
    pub listen_port: u16,
    pub discovery_enabled: bool,
    pub store_and_forward: bool,
    pub peer_timeout_ms: u64,
    pub max_stored_messages: usize,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct DtnConfig {
    pub storage_path: String,
    pub max_storage_gb: f64,
    pub routing_strategy: String,
    pub routing_copies: usize,
}

impl RustClientsConfig {
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let config: RustClientsConfig = toml::from_str(&content)?;
        Ok(config)
    }

    pub fn load_default() -> Self {
        // Load from default location or environment
        if let Ok(config_path) = std::env::var("RUST_CLIENTS_CONFIG") {
            Self::load_from_file(config_path).unwrap_or_else(|_| Self::default())
        } else if Path::new("rust_clients_config.toml").exists() {
            Self::load_from_file("rust_clients_config.toml").unwrap_or_else(|_| Self::default())
        } else {
            Self::default()
        }
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let content = toml::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }
}

impl Default for RustClientsConfig {
    fn default() -> Self {
        Self {
            global: GlobalConfig {
                log_level: "info".to_string(),
                data_directory: "./aivillage_data".to_string(),
                temp_directory: "/tmp/aivillage".to_string(),
            },
            network: NetworkConfig {
                betanet: BetaNetConfig {
                    mixnode_endpoints: vec![
                        "mix1.betanet.ai:9443".to_string(),
                        "mix2.betanet.ai:9443".to_string(),
                    ],
                    circuit_length: 3,
                    connection_timeout_ms: 10000,
                    max_concurrent_circuits: 50,
                },
                bitchat: BitChatConfig {
                    listen_port: 7777,
                    discovery_enabled: true,
                    store_and_forward: true,
                    peer_timeout_ms: 30000,
                    max_stored_messages: 1000,
                },
                dtn: DtnConfig {
                    storage_path: "./dtn_storage".to_string(),
                    max_storage_gb: 5.0,
                    routing_strategy: "spray_and_wait".to_string(),
                    routing_copies: 5,
                },
            },
            // ... other default configurations
        }
    }
}
```

## Performance Benchmarking

### Using the Bench Crate

**Location**: `integrations/bounties/betanet/tools/bench/`

```rust
// benchmarks.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use betanet_client::BetaNetClient;
use bitchat_cla::BitChatCLA;
use agent_fabric::AgentFabric;

fn benchmark_betanet_connection(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("betanet_connect", |b| {
        b.to_async(&rt).iter(|| async {
            let config = BetaNetClientConfig::default();
            let client = BetaNetClient::new(config).await.unwrap();
            black_box(client.connect().await)
        })
    });
}

fn benchmark_bitchat_messaging(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("bitchat_send_message", |b| {
        b.to_async(&rt).iter(|| async {
            let config = BitChatConfig::default();
            let mut bitchat = BitChatCLA::new(config).await.unwrap();

            let message = MessageConfig {
                recipient: "test_peer".to_string(),
                content: b"benchmark message".to_vec(),
                priority: "normal".to_string(),
                ttl_hours: 1,
            };

            black_box(bitchat.send_message(message).await)
        })
    });
}

fn benchmark_agent_fabric_deployment(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("agent_fabric_deploy", |b| {
        b.to_async(&rt).iter(|| async {
            let config = AgentConfig::default();
            let mut fabric = AgentFabric::new(config).await.unwrap();
            fabric.start().await.unwrap();

            let agent_spec = AgentSpec {
                name: "benchmark_agent".to_string(),
                service: "test_service".to_string(),
                replicas: 1,
                placement_strategy: "local".to_string(),
            };

            black_box(fabric.deploy_agent(agent_spec).await)
        })
    });
}

criterion_group!(benches,
    benchmark_betanet_connection,
    benchmark_bitchat_messaging,
    benchmark_agent_fabric_deployment
);
criterion_main!(benches);
```

### Running Benchmarks

```bash
# Navigate to bench crate
cd integrations/bounties/betanet/tools/bench

# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench betanet_connect

# Generate detailed report
cargo bench -- --output-format html

# Save baseline for comparison
cargo bench -- --save-baseline my_baseline

# Compare with baseline
cargo bench -- --baseline my_baseline
```

## Testing Infrastructure

### Integration Testing

```rust
// tests/integration_tests.rs
use tokio;
use std::time::Duration;

#[tokio::test]
async fn test_betanet_bitchat_integration() {
    // Initialize BetaNet client
    let betanet_config = BetaNetClientConfig {
        mixnode_endpoints: vec!["127.0.0.1:9443".to_string()],
        circuit_length: 2,
        connection_timeout: Duration::from_secs(5),
    };

    let betanet_client = BetaNetClient::new(betanet_config).await.unwrap();

    // Initialize BitChat CLA
    let bitchat_config = BitChatConfig {
        peer_id: "test_peer".to_string(),
        listen_port: 7778,
        discovery_enabled: false,
        store_and_forward: true,
    };

    let bitchat_cla = BitChatCLA::new(bitchat_config).await.unwrap();

    // Test cross-protocol message routing
    let message = "Hello from BetaNet to BitChat!";

    // Send via BetaNet
    let circuit_id = betanet_client.create_circuit().await.unwrap();
    let response = betanet_client.send_message(circuit_id, message.as_bytes()).await.unwrap();

    // Verify message received via BitChat
    let received = bitchat_cla.receive_message().await.unwrap();
    assert_eq!(received.content, message.as_bytes());

    // Clean up
    betanet_client.close_circuit(circuit_id).await.unwrap();
    betanet_client.disconnect().await.unwrap();
    bitchat_cla.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_agent_fabric_twin_vault_integration() {
    // Initialize Agent Fabric
    let fabric_config = AgentConfig {
        node_id: "test_fabric".to_string(),
        max_agents: 10,
        resource_limits: ResourceLimits {
            cpu_cores: 2,
            memory_gb: 4,
        },
    };

    let mut fabric = AgentFabric::new(fabric_config).await.unwrap();
    fabric.start().await.unwrap();

    // Initialize Twin Vault
    let vault_config = VaultConfig {
        storage_path: "./test_vault".to_string(),
        encryption_key: "test-key-32-bytes-long-for-test".to_string(),
        backup_enabled: false,
        compression: false,
    };

    let mut vault = TwinVault::new(vault_config).await.unwrap();

    // Deploy agent that uses Twin Vault
    let agent_spec = AgentSpec {
        name: "vault_agent".to_string(),
        service: "twin_storage".to_string(),
        replicas: 1,
        placement_strategy: "local".to_string(),
    };

    let agent_id = fabric.deploy_agent(agent_spec).await.unwrap();

    // Create digital twin
    let twin = DigitalTwin {
        id: "test_twin".to_string(),
        owner: "test_user".to_string(),
        version: "1.0.0".to_string(),
        data: serde_json::json!({"test": "data"}),
        privacy_level: "medium".to_string(),
        last_updated: chrono::Utc::now(),
    };

    // Store via agent
    let message = AgentMessage {
        to: agent_id.clone(),
        from: "test".to_string(),
        content: serde_json::json!({
            "command": "store_twin",
            "twin": twin
        }),
    };

    let response = fabric.send_message(message).await.unwrap();
    assert!(response.content["success"].as_bool().unwrap());

    // Verify storage directly
    let stored_twin = vault.get_twin("test_twin").await.unwrap().unwrap();
    assert_eq!(stored_twin.id, "test_twin");

    // Clean up
    fabric.undeploy_agent(&agent_id).await.unwrap();
    fabric.stop().await.unwrap();

    // Clean up test data
    std::fs::remove_dir_all("./test_vault").ok();
}
```

### Running Tests

```bash
# Run all tests
cargo test --all

# Run integration tests only
cargo test --test integration_tests

# Run tests with output
cargo test -- --nocapture

# Run tests in parallel
cargo test --jobs 4

# Test specific crate
cargo test -p betanet-client

# Test with code coverage
cargo install cargo-tarpaulin
cargo tarpaulin --all --out Html
```

## Development Workflow

### Code Quality Tools

```bash
# Format all code
cargo fmt --all

# Run clippy lints
cargo clippy --all -- -D warnings

# Security audit
cargo audit

# Check for outdated dependencies
cargo outdated

# Generate documentation
cargo doc --no-deps --open
```

### Custom Linting Rules

Create `.cargo/config.toml`:

```toml
[alias]
check-all = [
    "fmt", "--all", "--check",
    "clippy", "--all", "--", "-D", "warnings",
    "test", "--all",
    "audit"
]

fix-all = [
    "fmt", "--all",
    "clippy", "--all", "--fix", "--allow-dirty",
    "fix", "--all"
]

doc-all = [
    "doc", "--all", "--no-deps", "--open"
]
```

### Continuous Integration

```yaml
# .github/workflows/rust-ci.yml
name: Rust CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  CARGO_TERM_COLOR: always

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        rust: [stable, beta, nightly]

    steps:
    - uses: actions/checkout@v3

    - name: Install Rust toolchain
      uses: actions-rs/toolchain@v1
      with:
        toolchain: ${{ matrix.rust }}
        override: true
        components: rustfmt, clippy

    - name: Cache cargo registry
      uses: actions/cache@v3
      with:
        path: |
          ~/.cargo/registry
          ~/.cargo/git
          target
        key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}

    - name: Format check
      run: cargo fmt --all -- --check

    - name: Clippy check
      run: cargo clippy --all -- -D warnings

    - name: Run tests
      run: cargo test --all --verbose

    - name: Security audit
      run: |
        cargo install cargo-audit
        cargo audit

    - name: Build all crates
      run: cargo build --all --release
```

## Deployment Strategies

### Docker Deployment

```dockerfile
# Dockerfile.rust-clients
FROM rust:1.70 AS builder

# Set working directory
WORKDIR /app

# Copy manifests
COPY Cargo.toml Cargo.lock ./
COPY integrations/clients/rust/ ./integrations/clients/rust/

# Build dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release
RUN rm -f src/main.rs

# Copy source code
COPY . .

# Build application
RUN cargo build --release --all

# Runtime image
FROM debian:bullseye-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl1.1 \
    && rm -rf /var/lib/apt/lists/*

# Copy binaries
COPY --from=builder /app/target/release/betanet-client /usr/local/bin/
COPY --from=builder /app/target/release/bitchat-cla /usr/local/bin/
COPY --from=builder /app/target/release/agent-fabric /usr/local/bin/

# Copy configuration
COPY rust_clients_config.toml /etc/aivillage/

# Create data directory
RUN mkdir -p /var/lib/aivillage

# Set environment
ENV RUST_CLIENTS_CONFIG=/etc/aivillage/rust_clients_config.toml
ENV RUST_LOG=info

# Expose ports
EXPOSE 7777 8880 9443

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD /usr/local/bin/betanet-client health || exit 1

CMD ["/usr/local/bin/agent-fabric"]
```

### Docker Compose

```yaml
# docker-compose.rust-clients.yml
version: '3.8'

services:
  betanet-mixnode:
    build:
      context: .
      dockerfile: Dockerfile.rust-clients
    command: /usr/local/bin/betanet-mixnode
    ports:
      - "9443:9443"
    volumes:
      - betanet-data:/var/lib/aivillage/betanet
      - ./configs/betanet.toml:/etc/aivillage/betanet.toml:ro
    environment:
      - RUST_LOG=info
      - BETANET_CONFIG=/etc/aivillage/betanet.toml
    networks:
      - aivillage-network

  bitchat-node:
    build:
      context: .
      dockerfile: Dockerfile.rust-clients
    command: /usr/local/bin/bitchat-cla
    ports:
      - "7777:7777"
    volumes:
      - bitchat-data:/var/lib/aivillage/bitchat
    environment:
      - RUST_LOG=info
    networks:
      - aivillage-network
    depends_on:
      - betanet-mixnode

  agent-fabric:
    build:
      context: .
      dockerfile: Dockerfile.rust-clients
    command: /usr/local/bin/agent-fabric
    ports:
      - "8880:8880"
    volumes:
      - agent-data:/var/lib/aivillage/agents
      - twin-vault-data:/var/lib/aivillage/twins
    environment:
      - RUST_LOG=info
    networks:
      - aivillage-network
    depends_on:
      - betanet-mixnode
      - bitchat-node

volumes:
  betanet-data:
  bitchat-data:
  agent-data:
  twin-vault-data:

networks:
  aivillage-network:
    driver: bridge
```

### Kubernetes Deployment

```yaml
# k8s/rust-clients.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: betanet-mixnode
  labels:
    app: betanet-mixnode
spec:
  replicas: 3
  selector:
    matchLabels:
      app: betanet-mixnode
  template:
    metadata:
      labels:
        app: betanet-mixnode
    spec:
      containers:
      - name: betanet-mixnode
        image: aivillage/rust-clients:latest
        command: ["/usr/local/bin/betanet-mixnode"]
        ports:
        - containerPort: 9443
        env:
        - name: RUST_LOG
          value: "info"
        volumeMounts:
        - name: config
          mountPath: /etc/aivillage
        - name: data
          mountPath: /var/lib/aivillage
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          exec:
            command:
            - /usr/local/bin/betanet-client
            - health
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - /usr/local/bin/betanet-client
            - ready
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: rust-clients-config
      - name: data
        persistentVolumeClaim:
          claimName: betanet-data-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: betanet-mixnode-service
spec:
  selector:
    app: betanet-mixnode
  ports:
  - port: 9443
    targetPort: 9443
  type: LoadBalancer
```

## Security Considerations

### Secure Configuration

```rust
// security.rs - Security utilities for Rust clients
use rand::rngs::OsRng;
use ring::{aead, digest, hkdf, rand};
use std::collections::HashMap;
use zeroize::Zeroize;

pub struct SecureConfig {
    encryption_key: [u8; 32],
    signing_key: [u8; 32],
    key_derivation_salt: [u8; 32],
}

impl SecureConfig {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let mut encryption_key = [0u8; 32];
        let mut signing_key = [0u8; 32];
        let mut key_derivation_salt = [0u8; 32];

        // Generate cryptographically secure random keys
        rand::SystemRandom::new().fill(&mut encryption_key)?;
        rand::SystemRandom::new().fill(&mut signing_key)?;
        rand::SystemRandom::new().fill(&mut key_derivation_salt)?;

        Ok(Self {
            encryption_key,
            signing_key,
            key_derivation_salt,
        })
    }

    pub fn derive_key(&self, context: &[u8], length: usize) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let salt = hkdf::Salt::new(hkdf::HKDF_SHA256, &self.key_derivation_salt);
        let prk = salt.extract(&self.encryption_key);
        let okm = prk.expand(&[context], length)?;

        let mut derived_key = vec![0u8; length];
        okm.fill(&mut derived_key)?;

        Ok(derived_key)
    }

    pub fn encrypt_data(&self, data: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let key = aead::UnboundKey::new(&aead::CHACHA20_POLY1305, &self.encryption_key)?;
        let key = aead::LessSafeKey::new(key);

        let mut nonce = [0u8; 12];
        rand::SystemRandom::new().fill(&mut nonce)?;

        let nonce = aead::Nonce::assume_unique_for_key(nonce);

        let mut ciphertext = data.to_vec();
        key.seal_in_place_append_tag(nonce, aead::Aad::empty(), &mut ciphertext)?;

        // Prepend nonce to ciphertext
        let mut result = nonce.as_ref().to_vec();
        result.extend_from_slice(&ciphertext);

        Ok(result)
    }

    pub fn decrypt_data(&self, encrypted_data: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        if encrypted_data.len() < 12 {
            return Err("Invalid encrypted data".into());
        }

        let (nonce_bytes, ciphertext) = encrypted_data.split_at(12);
        let nonce = aead::Nonce::try_assume_unique_for_key(nonce_bytes)?;

        let key = aead::UnboundKey::new(&aead::CHACHA20_POLY1305, &self.encryption_key)?;
        let key = aead::LessSafeKey::new(key);

        let mut plaintext = ciphertext.to_vec();
        let plaintext = key.open_in_place(nonce, aead::Aad::empty(), &mut plaintext)?;

        Ok(plaintext.to_vec())
    }
}

impl Drop for SecureConfig {
    fn drop(&mut self) {
        self.encryption_key.zeroize();
        self.signing_key.zeroize();
        self.key_derivation_salt.zeroize();
    }
}

// Secure storage for sensitive configuration
pub struct SecureStorage {
    config_path: String,
    secure_config: SecureConfig,
}

impl SecureStorage {
    pub fn new(config_path: String) -> Result<Self, Box<dyn std::error::Error>> {
        let secure_config = SecureConfig::new()?;

        Ok(Self {
            config_path,
            secure_config,
        })
    }

    pub fn store_sensitive_data(&self, key: &str, data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        let encrypted_data = self.secure_config.encrypt_data(data)?;

        // Store in secure keychain/credential store based on platform
        #[cfg(target_os = "linux")]
        self.store_linux_keyring(key, &encrypted_data)?;

        #[cfg(target_os = "macos")]
        self.store_macos_keychain(key, &encrypted_data)?;

        #[cfg(target_os = "windows")]
        self.store_windows_credential(key, &encrypted_data)?;

        Ok(())
    }

    #[cfg(target_os = "linux")]
    fn store_linux_keyring(&self, key: &str, data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        // Use libsecret or similar secure storage
        // Implementation would go here
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn store_macos_keychain(&self, key: &str, data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        // Use macOS Keychain Services
        // Implementation would go here
        Ok(())
    }

    #[cfg(target_os = "windows")]
    fn store_windows_credential(&self, key: &str, data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        // Use Windows Credential Manager
        // Implementation would go here
        Ok(())
    }
}
```

### Network Security

```rust
// network_security.rs - Network security utilities
use std::net::{IpAddr, SocketAddr};
use std::collections::HashSet;

pub struct NetworkSecurityManager {
    allowed_ips: HashSet<IpAddr>,
    blocked_ips: HashSet<IpAddr>,
    rate_limits: HashMap<IpAddr, RateLimitState>,
}

struct RateLimitState {
    requests: u32,
    window_start: std::time::Instant,
}

impl NetworkSecurityManager {
    pub fn new() -> Self {
        Self {
            allowed_ips: HashSet::new(),
            blocked_ips: HashSet::new(),
            rate_limits: HashMap::new(),
        }
    }

    pub fn add_allowed_ip(&mut self, ip: IpAddr) {
        self.allowed_ips.insert(ip);
    }

    pub fn add_blocked_ip(&mut self, ip: IpAddr) {
        self.blocked_ips.insert(ip);
    }

    pub fn is_connection_allowed(&mut self, addr: &SocketAddr) -> bool {
        let ip = addr.ip();

        // Check if IP is blocked
        if self.blocked_ips.contains(&ip) {
            return false;
        }

        // Check if IP is explicitly allowed
        if !self.allowed_ips.is_empty() && !self.allowed_ips.contains(&ip) {
            return false;
        }

        // Check rate limits
        self.check_rate_limit(ip)
    }

    fn check_rate_limit(&mut self, ip: IpAddr) -> bool {
        let now = std::time::Instant::now();
        const WINDOW_SIZE: std::time::Duration = std::time::Duration::from_secs(60);
        const MAX_REQUESTS: u32 = 100;

        let rate_limit = self.rate_limits.entry(ip).or_insert(RateLimitState {
            requests: 0,
            window_start: now,
        });

        // Reset window if needed
        if now.duration_since(rate_limit.window_start) > WINDOW_SIZE {
            rate_limit.requests = 0;
            rate_limit.window_start = now;
        }

        rate_limit.requests += 1;
        rate_limit.requests <= MAX_REQUESTS
    }

    pub fn cleanup_old_entries(&mut self) {
        let now = std::time::Instant::now();
        const CLEANUP_AGE: std::time::Duration = std::time::Duration::from_secs(3600); // 1 hour

        self.rate_limits.retain(|_, state| {
            now.duration_since(state.window_start) < CLEANUP_AGE
        });
    }
}
```

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Build Failures

```bash
# Clear cargo cache
cargo clean

# Update toolchain
rustup update

# Check for missing dependencies
sudo apt-get install build-essential pkg-config libssl-dev

# Fix permission issues
chmod -R 755 ~/.cargo

# Rebuild with verbose output
cargo build --verbose
```

#### 2. Network Connection Issues

```rust
// debug_network.rs - Network debugging utilities
use std::net::TcpStream;
use std::time::Duration;

pub async fn test_connectivity(endpoints: &[String]) -> Vec<(String, bool)> {
    let mut results = Vec::new();

    for endpoint in endpoints {
        let connected = test_endpoint(endpoint).await;
        results.push((endpoint.clone(), connected));

        if connected {
            println!("✅ {}: Connected", endpoint);
        } else {
            println!("❌ {}: Failed to connect", endpoint);
        }
    }

    results
}

async fn test_endpoint(endpoint: &str) -> bool {
    match TcpStream::connect_timeout(
        &endpoint.parse().unwrap(),
        Duration::from_secs(5)
    ) {
        Ok(_) => true,
        Err(e) => {
            eprintln!("Connection to {} failed: {}", endpoint, e);
            false
        }
    }
}

// Usage in troubleshooting
#[tokio::main]
async fn main() {
    let endpoints = vec![
        "mix1.betanet.ai:9443".to_string(),
        "mix2.betanet.ai:9443".to_string(),
        "127.0.0.1:7777".to_string(),
    ];

    test_connectivity(&endpoints).await;
}
```

#### 3. Performance Issues

```rust
// performance_debug.rs - Performance debugging
use std::time::Instant;

pub struct PerformanceTracker {
    start_time: Instant,
    checkpoints: Vec<(String, Instant)>,
}

impl PerformanceTracker {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            checkpoints: Vec::new(),
        }
    }

    pub fn checkpoint(&mut self, name: &str) {
        self.checkpoints.push((name.to_string(), Instant::now()));
    }

    pub fn report(&self) {
        println!("Performance Report:");
        println!("==================");

        let mut last_time = self.start_time;

        for (name, time) in &self.checkpoints {
            let duration = time.duration_since(last_time);
            let total_duration = time.duration_since(self.start_time);

            println!("{}: {:.2}ms (total: {:.2}ms)",
                name,
                duration.as_secs_f64() * 1000.0,
                total_duration.as_secs_f64() * 1000.0
            );

            last_time = *time;
        }

        let total = Instant::now().duration_since(self.start_time);
        println!("Total execution time: {:.2}ms", total.as_secs_f64() * 1000.0);
    }
}
```

#### 4. Memory Issues

```bash
# Check memory usage
valgrind --tool=massif target/release/betanet-client

# Profile memory with heaptrack (Linux)
heaptrack target/release/betanet-client
heaptrack_gui heaptrack.betanet-client.*

# Use built-in Rust profiling
CARGO_PROFILE_RELEASE_DEBUG=true cargo build --release
perf record --call-graph=dwarf target/release/betanet-client
perf report
```

### Logging Configuration

```rust
// logging.rs - Centralized logging configuration
use tracing::{info, warn, error, debug};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

pub fn init_logging() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    info!("Logging initialized");
}

// Custom structured logging
use serde_json::json;

pub fn log_network_event(event_type: &str, details: serde_json::Value) {
    info!(
        event_type = event_type,
        details = %details,
        "Network event occurred"
    );
}

pub fn log_performance_metric(metric_name: &str, value: f64, unit: &str) {
    info!(
        metric_name = metric_name,
        value = value,
        unit = unit,
        "Performance metric"
    );
}

// Usage example
fn example_usage() {
    log_network_event("peer_connected", json!({
        "peer_id": "peer_123",
        "endpoint": "10.0.1.100:7777",
        "connection_time_ms": 150
    }));

    log_performance_metric("message_latency", 50.5, "ms");
}
```

## Production Deployment Checklist

### Pre-Deployment Verification

- [ ] All crates build successfully in release mode
- [ ] Unit tests pass for all components
- [ ] Integration tests validate cross-component communication
- [ ] Security audit completed with no critical vulnerabilities
- [ ] Performance benchmarks meet requirements
- [ ] Documentation is complete and up-to-date
- [ ] Configuration files are properly secured
- [ ] Logging and monitoring configured
- [ ] Backup and recovery procedures tested

### Deployment Steps

1. **Environment Preparation**
   ```bash
   # Create deployment user
   sudo useradd -m -s /bin/bash aivillage
   sudo usermod -aG docker aivillage

   # Create data directories
   sudo mkdir -p /var/lib/aivillage/{betanet,bitchat,agents,twins}
   sudo chown -R aivillage:aivillage /var/lib/aivillage

   # Install systemd services
   sudo cp configs/systemd/* /etc/systemd/system/
   sudo systemctl daemon-reload
   ```

2. **Security Configuration**
   ```bash
   # Set up TLS certificates
   sudo certbot certonly --standalone -d betanet.yourdomain.com

   # Configure firewall
   sudo ufw allow 22/tcp
   sudo ufw allow 7777/tcp
   sudo ufw allow 8880/tcp
   sudo ufw allow 9443/tcp
   sudo ufw --force enable
   ```

3. **Service Deployment**
   ```bash
   # Deploy with Docker Compose
   docker-compose -f docker-compose.rust-clients.yml up -d

   # Or deploy with systemd
   sudo systemctl enable aivillage-betanet
   sudo systemctl enable aivillage-bitchat
   sudo systemctl enable aivillage-agents
   sudo systemctl start aivillage-betanet
   sudo systemctl start aivillage-bitchat
   sudo systemctl start aivillage-agents
   ```

4. **Post-Deployment Verification**
   ```bash
   # Check service status
   docker-compose ps
   # or
   sudo systemctl status aivillage-*

   # Verify connectivity
   ./scripts/verify_deployment.sh

   # Run smoke tests
   cargo test --test smoke_tests
   ```

### Monitoring Setup

```bash
# Install monitoring stack
docker-compose -f monitoring/docker-compose.yml up -d

# Configure alerts
cp monitoring/alerts.yml /etc/prometheus/
sudo systemctl reload prometheus
```

### Maintenance Procedures

1. **Regular Updates**
   ```bash
   # Update Rust toolchain
   rustup update

   # Update dependencies
   cargo update

   # Rebuild and test
   cargo build --release --all
   cargo test --all

   # Deploy updates
   docker-compose down
   docker-compose build
   docker-compose up -d
   ```

2. **Log Rotation**
   ```bash
   # Configure logrotate
   sudo cp configs/logrotate/aivillage /etc/logrotate.d/
   sudo logrotate -f /etc/logrotate.d/aivillage
   ```

3. **Backup Procedures**
   ```bash
   # Backup configuration and data
   ./scripts/backup.sh

   # Verify backups
   ./scripts/verify_backup.sh
   ```

## Support and Community

### Getting Help

1. **Documentation**: This hub and individual crate README files
2. **GitHub Issues**: [AIVillage Issues](https://github.com/your-org/AIVillage/issues)
3. **Discord Community**: Join our [Discord server](https://discord.gg/aivillage)
4. **Stack Overflow**: Tag questions with `aivillage` and `rust`

### Contributing

1. **Code Contributions**:
   - Fork the repository
   - Create feature branch
   - Write tests for new functionality
   - Ensure all tests pass
   - Submit pull request

2. **Documentation Improvements**:
   - Update this hub for new features
   - Add examples and tutorials
   - Improve troubleshooting guides

3. **Bug Reports**:
   - Use GitHub issue templates
   - Provide minimal reproduction cases
   - Include system information and logs

### Roadmap

**Q1 2024**:
- WebAssembly bindings for browser integration
- Enhanced security features
- Performance optimizations

**Q2 2024**:
- Mobile SDK development
- IoT device support
- Advanced routing algorithms

**Q3 2024**:
- Machine learning integration
- Enhanced privacy features
- Cross-platform GUI tools

**Q4 2024**:
- Production hardening
- Enterprise features
- Comprehensive audit and certification

## Conclusion

The Rust client ecosystem provides a robust, high-performance foundation for the AIVillage distributed computing platform. With 30+ specialized crates covering networking, agent management, security, and storage, developers have access to a comprehensive toolkit for building distributed AI applications.

Key benefits of the Rust implementation:

- **Memory Safety**: Rust's ownership model prevents common security vulnerabilities
- **Performance**: Zero-cost abstractions and efficient runtime performance
- **Concurrency**: Built-in async/await support for scalable networking
- **Cross-platform**: Runs on Linux, macOS, Windows, and embedded systems
- **FFI Support**: Easy integration with C/C++, Python, and other languages
- **Rich Ecosystem**: Leverages the broader Rust ecosystem for additional functionality

For successful deployment:

1. Start with the core crates (BetaNet, BitChat, Agent Fabric)
2. Follow the configuration guidelines for your environment
3. Implement proper monitoring and logging
4. Use the security utilities for production deployments
5. Participate in the community for ongoing support

The Rust client infrastructure enables the creation of secure, scalable, and efficient distributed AI systems that can operate across diverse network conditions and device capabilities.
