"""
HTX Transport Test Suite - Betanet v1.1

Comprehensive test suite for the modular HTX transport implementation.

Test Modules:
- test_frame_format: HTX frame encoding/decoding and buffering
- test_utls_fingerprinting: JA3/JA4 fingerprint calibration and ClientHello generation
- test_noise_protocol: Noise XK handshake pattern and transport encryption
- test_access_tickets: Authentication, rate limiting, and replay protection
- test_transport: Main transport coordinator and component integration
- test_integration: End-to-end integration tests across all components

Usage:
    # Run all HTX tests
    pytest tests/htx/ -v

    # Run specific test module
    pytest tests/htx/test_frame_format.py -v

    # Run with coverage
    pytest tests/htx/ --cov=src.core.p2p.htx --cov-report=html

    # Run smoke tests only
    python -m pytest tests/htx/ -k "smoke_test" -v
"""

__version__ = "1.1.0"
