# ADR-0001: Twin Extraction Strategy

## Status
Accepted – Implemented

## Context
The project originally envisioned an optional "twin extraction" process that
snapshots an agent's state into a portable form for offline analysis.
This functionality now exists as a standalone microservice running inside the
`atlantis-twin` container on port `8001`.

## Decision
The service exposes `/v1/chat`, `/v1/embeddings` (stub) and `/healthz`. It is
deployed via Docker (`atlantis-twin:0.1.0`) and driven by environment variables
for the model path and runtime settings. Conversation state is stored in an
in-memory `LRUCache` capped at 1000 conversations, ensuring old data is
evicted automatically.

## Consequences
Existing scripts should invoke the microservice via its HTTP API rather than
directly importing internal modules. Old placeholders can be removed.
