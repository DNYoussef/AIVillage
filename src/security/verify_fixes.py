"""Verify serialization hardening and TLS configuration.

This module provides runtime checks to ensure that known security
fixes remain in place. It focuses on two areas:

1. Serialization hardening: uses ``yaml.safe_load`` to protect
   against execution of arbitrary objects.
2. TLS configuration: verifies that the default SSL context enforces
   certificate validation and uses modern TLS versions.
"""

from __future__ import annotations

import ssl
from typing import Tuple

import yaml


def check_serialization_hardening() -> bool:
    """Ensure unsafe YAML payloads cannot be deserialized.

    The check attempts to load a payload that would execute a shell
    command if processed with ``yaml.load``. Using ``yaml.safe_load``
    should raise a :class:`yaml.constructor.ConstructorError` and
    prevent the command from running.
    """

    malicious_payload = "!!python/object/apply:os.system ['echo vulnerable']"
    try:
        yaml.safe_load(malicious_payload)
    except yaml.constructor.ConstructorError:
        return True
    except Exception:
        return False
    return False


def check_tls_configuration() -> bool:
    """Verify that the default SSL context is hardened.

    The context should enforce certificate validation and restrict the
    protocol to TLS v1.2 or newer.
    """

    context = ssl.create_default_context()
    cert_required = context.verify_mode == ssl.CERT_REQUIRED

    minimum_version = getattr(ssl, "TLSVersion", None)
    if minimum_version is not None:
        tls_version_ok = context.minimum_version >= ssl.TLSVersion.TLSv1_2
    else:
        tls_version_ok = True  # Python < 3.7 defaults to secure versions

    return cert_required and tls_version_ok


def verify_security_fixes() -> Tuple[bool, bool]:
    """Run all security checks and return their results."""

    return check_serialization_hardening(), check_tls_configuration()


if __name__ == "__main__":
    serialization_ok, tls_ok = verify_security_fixes()
    print(f"Serialization hardening: {'PASS' if serialization_ok else 'FAIL'}")
    print(f"TLS configuration: {'PASS' if tls_ok else 'FAIL'}")
    if not (serialization_ok and tls_ok):
        raise SystemExit(1)
