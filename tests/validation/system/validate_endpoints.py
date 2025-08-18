#!/usr/bin/env python3
"""CODEX Integration Requirements - Service Endpoint Validation
Tests all service endpoints per CODEX specifications
"""

import socket
import time
from typing import Any

import requests


def check_port_availability(host: str, port: int, timeout: float = 5.0) -> dict[str, Any]:
    """Check if a port is available and responding."""
    result = {
        "host": host,
        "port": port,
        "available": False,
        "listening": False,
        "response_time_ms": None,
        "error": None,
    }

    try:
        start_time = time.time()
        sock = socket.create_connection((host, port), timeout)
        end_time = time.time()

        result["available"] = True
        result["listening"] = True
        result["response_time_ms"] = (end_time - start_time) * 1000
        sock.close()

    except TimeoutError:
        result["error"] = "Connection timeout"
    except ConnectionRefusedError:
        result["error"] = "Connection refused - service not running"
    except Exception as e:
        result["error"] = str(e)

    return result


def test_http_endpoint(url: str, timeout: float = 10.0, expected_codes: list[int] = None) -> dict[str, Any]:
    """Test HTTP endpoint availability and response."""
    if expected_codes is None:
        expected_codes = [
            200,
            404,
        ]  # 404 is acceptable if service exists but endpoint not implemented

    result = {
        "url": url,
        "status": "ERROR",
        "status_code": None,
        "response_time_ms": None,
        "headers": None,
        "content_type": None,
        "error": None,
    }

    try:
        start_time = time.time()
        response = requests.get(url, timeout=timeout)
        end_time = time.time()

        result["status_code"] = response.status_code
        result["response_time_ms"] = (end_time - start_time) * 1000
        result["headers"] = dict(response.headers)
        result["content_type"] = response.headers.get("content-type", "")

        if response.status_code in expected_codes:
            result["status"] = "OK"
        else:
            result["status"] = "UNEXPECTED_CODE"
            result["error"] = f"Expected {expected_codes}, got {response.status_code}"

    except requests.exceptions.Timeout:
        result["error"] = "Request timeout"
    except requests.exceptions.ConnectionError:
        result["error"] = "Connection error - service not running"
    except Exception as e:
        result["error"] = str(e)

    return result


def validate_service_endpoints() -> dict[str, Any]:
    """Validate all service endpoints per CODEX Integration Requirements."""
    print("[VALIDATION] Service Endpoint Testing Per CODEX Integration Requirements")
    print("=" * 80)

    # CODEX Required Ports (from CODEX_INTEGRATION_REQUIREMENTS.md)
    required_ports = [
        {
            "service": "LibP2P Main",
            "host": "127.0.0.1",
            "port": 4001,
            "protocol": "TCP/UDP",
            "purpose": "Primary P2P communication",
        },
        {
            "service": "LibP2P WebSocket",
            "host": "127.0.0.1",
            "port": 4002,
            "protocol": "TCP",
            "purpose": "WebSocket transport",
        },
        {
            "service": "Digital Twin API",
            "host": "127.0.0.1",
            "port": 8080,
            "protocol": "HTTP",
            "purpose": "REST API endpoints",
        },
        {
            "service": "Evolution Metrics",
            "host": "127.0.0.1",
            "port": 8081,
            "protocol": "HTTP",
            "purpose": "Metrics collection API",
        },
        {
            "service": "RAG Pipeline",
            "host": "127.0.0.1",
            "port": 8082,
            "protocol": "HTTP",
            "purpose": "Query processing API",
        },
        {
            "service": "Redis (Optional)",
            "host": "127.0.0.1",
            "port": 6379,
            "protocol": "TCP",
            "purpose": "Caching and pub/sub",
        },
    ]

    # CODEX Health Check Endpoints (from CODEX_INTEGRATION_REQUIREMENTS.md)
    health_endpoints = [
        {
            "name": "Evolution Metrics",
            "url": "http://127.0.0.1:8081/health/evolution",
            "expected": [200, 404],
        },
        {
            "name": "RAG Pipeline",
            "url": "http://127.0.0.1:8082/health/rag",
            "expected": [200, 404],
        },
        {
            "name": "P2P Network",
            "url": "http://127.0.0.1:4001/health/p2p",
            "expected": [200, 404],
        },
        {
            "name": "Digital Twin",
            "url": "http://127.0.0.1:8080/health/twin",
            "expected": [200, 404],
        },
    ]

    # Additional endpoints to test
    api_endpoints = [
        {
            "name": "Digital Twin Root",
            "url": "http://127.0.0.1:8080/",
            "expected": [200, 404, 405],
        },
        {
            "name": "Evolution Metrics Root",
            "url": "http://127.0.0.1:8081/",
            "expected": [200, 404, 405],
        },
        {
            "name": "RAG Pipeline Root",
            "url": "http://127.0.0.1:8082/",
            "expected": [200, 404, 405],
        },
        {
            "name": "P2P Network Root",
            "url": "http://127.0.0.1:4001/",
            "expected": [200, 404, 405],
        },
    ]

    results = {
        "port_checks": [],
        "health_checks": [],
        "api_checks": [],
        "summary": {
            "ports_listening": 0,
            "ports_total": len(required_ports),
            "health_endpoints_ok": 0,
            "health_endpoints_total": len(health_endpoints),
            "api_endpoints_ok": 0,
            "api_endpoints_total": len(api_endpoints),
        },
    }

    print("\\n1. PORT AVAILABILITY CHECKS:")
    print("-" * 50)

    # Test port availability
    for port_info in required_ports:
        port_result = check_port_availability(port_info["host"], port_info["port"], timeout=2.0)
        port_result.update(port_info)
        results["port_checks"].append(port_result)

        status_icon = "[LISTENING]" if port_result["listening"] else "[NOT_LISTENING]"
        response_time = f" ({port_result['response_time_ms']:.1f}ms)" if port_result["response_time_ms"] else ""
        error_msg = f" - {port_result['error']}" if port_result["error"] else ""

        print(
            f"{status_icon:<15} {port_info['service']:<20} {port_info['host']}:{port_info['port']}{response_time}{error_msg}"
        )

        if port_result["listening"]:
            results["summary"]["ports_listening"] += 1

    print("\\n2. HEALTH CHECK ENDPOINTS:")
    print("-" * 50)

    # Test health check endpoints
    for endpoint_info in health_endpoints:
        health_result = test_http_endpoint(endpoint_info["url"], timeout=5.0, expected_codes=endpoint_info["expected"])
        health_result.update(endpoint_info)
        results["health_checks"].append(health_result)

        status_icon = "[OK]" if health_result["status"] == "OK" else "[FAIL]"
        status_code = f"({health_result['status_code']})" if health_result["status_code"] else ""
        response_time = f" {health_result['response_time_ms']:.1f}ms" if health_result["response_time_ms"] else ""
        error_msg = f" - {health_result['error']}" if health_result["error"] else ""

        print(f"{status_icon:<10} {endpoint_info['name']:<20} {status_code:<6}{response_time}{error_msg}")

        if health_result["status"] == "OK":
            results["summary"]["health_endpoints_ok"] += 1

    print("\\n3. API ENDPOINT TESTS:")
    print("-" * 50)

    # Test API endpoints
    for endpoint_info in api_endpoints:
        api_result = test_http_endpoint(endpoint_info["url"], timeout=5.0, expected_codes=endpoint_info["expected"])
        api_result.update(endpoint_info)
        results["api_checks"].append(api_result)

        status_icon = "[OK]" if api_result["status"] == "OK" else "[FAIL]"
        status_code = f"({api_result['status_code']})" if api_result["status_code"] else ""
        response_time = f" {api_result['response_time_ms']:.1f}ms" if api_result["response_time_ms"] else ""
        error_msg = f" - {api_result['error']}" if api_result["error"] else ""

        print(f"{status_icon:<10} {endpoint_info['name']:<20} {status_code:<6}{response_time}{error_msg}")

        if api_result["status"] == "OK":
            results["summary"]["api_endpoints_ok"] += 1

    print("\\n" + "=" * 80)
    print("SERVICE ENDPOINT VALIDATION SUMMARY:")
    print("=" * 80)

    print(f"Port Availability: {results['summary']['ports_listening']}/{results['summary']['ports_total']} listening")
    print(
        f"Health Endpoints: {results['summary']['health_endpoints_ok']}/{results['summary']['health_endpoints_total']} responding"
    )
    print(
        f"API Endpoints: {results['summary']['api_endpoints_ok']}/{results['summary']['api_endpoints_total']} responding"
    )

    # Calculate overall health
    total_checks = (
        results["summary"]["ports_total"]
        + results["summary"]["health_endpoints_total"]
        + results["summary"]["api_endpoints_total"]
    )
    total_passing = (
        results["summary"]["ports_listening"]
        + results["summary"]["health_endpoints_ok"]
        + results["summary"]["api_endpoints_ok"]
    )

    health_percentage = (total_passing / total_checks) * 100 if total_checks > 0 else 0

    print(f"\\nOverall Endpoint Health: {total_passing}/{total_checks} ({health_percentage:.1f}%)")

    if health_percentage >= 80:
        print("STATUS: ACCEPTABLE - Most services are operational")
    elif health_percentage >= 50:
        print("STATUS: DEGRADED - Some services need attention")
    else:
        print("STATUS: CRITICAL - Many services are not responding")

    print("\\nNOTE: Services not currently running will show as not listening.")
    print("This is expected in development environment without active services.")

    return results


if __name__ == "__main__":
    validate_service_endpoints()
