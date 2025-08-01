# ADR-S4-01: Alerting Rules

**Date**: 2025-07-12

**Status**: Accepted

**Sprint**: S4 (Buffer)

## Context
Sprint 4 introduces proactive alerting so operators are notified of service degradation. We need rules that balance signal and noise while covering the most critical failure modes from soak testing.

## Decision
Add a Prometheus rule file (`monitoring/alerts.yml`) loaded by the compose stack. Two alerts are defined:

- **GatewayHighLatency** – triggers when p99 latency for the Gateway exceeds 250&nbsp;ms for five minutes.
- **TwinMemoryGrowth** – fires if the Twin's memory usage grows by more than 2% per hour, indicating a potential leak.

These thresholds were chosen based on soak test results and provide early warning without being too chatty.

## Consequences
✅ Operators receive warnings before outages.
✅ Alert rules are version controlled and tested via CI.
❌ More components to maintain (Prometheus reloads required on change).

## Alternatives
- ❌ Log-only metrics: lacks real-time notification.
- ❌ External APM: heavier weight and adds cost.
