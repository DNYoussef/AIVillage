# ADR-0002  Messaging Protocol Choice

**Status**: Proposed  
**Date**: 2025-07-01

## Context
Edge devices may sit behind strict firewalls.  Pure gRPC (HTTP/2) offers type-safety & bi-directional streams but fails on some corporate proxies; WebSocket (HTTP/1.1 upgrade) is widely allowed but lacks strong proto contracts.

## Decision (draft)
Adopt **gRPC primary** with automatic **WebSocket fallback** using JSON-encoded protobuf messages.  Connection manager negotiates highest-capability transport then hands channel to agents.

## Consequences
* Slightly heavier client lib (~100 KB for fallback shim).
* Debugging two code-paths.
* Future QUIC support remains possible.
