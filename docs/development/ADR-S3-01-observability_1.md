# ADR-S3-01: Observability Stack

**Date**: 2025-07-02

**Status**: Accepted

**Sprint**: S3 (Stabilize)

## Context
Sprint 3 requires monitoring to track p99 latency and detect memory leaks during 8h soak tests.

## Decision
Adopt Prometheus v2.52 + Grafana 11 via docker-compose override.

## Consequences
✅ p99 latency visible in <1 min
✅ Memory leak detection automated
❌ Two extra containers (+200MB RAM)
❌ Need to maintain dashboards

## Alternatives
- ❌ ELK: Too heavy (2GB+)
- ❌ Cloud: Vendor lock-in
