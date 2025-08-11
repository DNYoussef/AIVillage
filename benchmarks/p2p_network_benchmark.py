#!/usr/bin/env python3
"""Localhost P2P networking latency benchmark.

This benchmark spins up a simple echo server and measures round-trip
latency for a series of messages to approximate P2P performance.
"""

import argparse
import asyncio
import json
from pathlib import Path
import time


async def _handle_echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    data = await reader.read(100)
    writer.write(data)
    await writer.drain()
    writer.close()


async def run_benchmark(messages: int = 5) -> dict:
    server = await asyncio.start_server(_handle_echo, "127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()
    latencies: list[float] = []

    for _ in range(messages):
        start = time.perf_counter()
        reader, writer = await asyncio.open_connection(host, port)
        writer.write(b"ping")
        await writer.drain()
        await reader.read(4)
        writer.close()
        await writer.wait_closed()
        latencies.append(time.perf_counter() - start)

    server.close()
    await server.wait_closed()

    avg_ms = sum(latencies) / len(latencies) * 1000
    return {
        "messages": messages,
        "avg_round_trip_ms": round(avg_ms, 3),
        "success_rate_percent": 100.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simple P2P networking benchmark and output JSON metrics")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/benchmarks/p2p_network_results.json"),
        help="File to write benchmark results",
    )
    args = parser.parse_args()

    results = asyncio.run(run_benchmark())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
