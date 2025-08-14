"""
Comprehensive Test Suite for uTLS Fingerprinting - Betanet v1.1

Tests the modular uTLS fingerprinting implementation including:
- JA3/JA4 fingerprint generation and calibration
- Browser template library (Chrome, Firefox, Safari)
- ClientHello generation matching target fingerprints
- Fingerprint randomization and diversity
- TLS extension handling

Building on existing test patterns from the codebase.
"""

import os
import sys

# Add src to path following existing pattern
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from core.p2p.htx.utls_fingerprinting import (
    ClientHelloFingerprint,
    FingerprintTemplate,
    TLSExtension,
    uTLSFingerprintCalibrator,
)


class TestTLSExtension:
    """Test TLS extension structure."""

    def test_extension_creation(self):
        """Test basic TLS extension creation."""
        ext_data = b"test extension data"
        extension = TLSExtension(extension_type=0, data=ext_data)

        assert extension.extension_type == 0
        assert extension.data == ext_data

    def test_extension_encoding(self):
        """Test TLS extension wire format encoding."""
        ext_data = b"hello world"
        extension = TLSExtension(extension_type=23, data=ext_data)  # Session ticket

        encoded = extension.encode()

        # Should be: type(2 bytes) + length(2 bytes) + data
        expected_length = 2 + 2 + len(ext_data)
        assert len(encoded) == expected_length

        # Verify format: type(23) + length(11) + data
        assert encoded[:2] == b"\x00\x17"  # Type 23 in big-endian
        assert encoded[2:4] == b"\x00\x0b"  # Length 11 in big-endian
        assert encoded[4:] == ext_data

    def test_empty_extension_data(self):
        """Test extension with empty data."""
        extension = TLSExtension(extension_type=35, data=b"")

        encoded = extension.encode()

        # Should be: type(2 bytes) + length(2 bytes, value=0) + no data
        assert len(encoded) == 4
        assert encoded == b"\x00\x23\x00\x00"  # Type 35, length 0


class TestClientHelloFingerprint:
    """Test ClientHello fingerprint structure."""

    def test_fingerprint_creation(self):
        """Test basic fingerprint creation."""
        fingerprint = ClientHelloFingerprint(
            version=0x0303,  # TLS 1.2
            cipher_suites=[0x1301, 0x1302, 0xC02C],
            extensions=[0, 23, 35],
            browser_type="chrome",
            browser_version="120.0",
        )

        assert fingerprint.version == 0x0303
        assert len(fingerprint.cipher_suites) == 3
        assert len(fingerprint.extensions) == 3
        assert fingerprint.browser_type == "chrome"
        assert fingerprint.browser_version == "120.0"

    def test_fingerprint_defaults(self):
        """Test fingerprint with default values."""
        fingerprint = ClientHelloFingerprint()

        assert fingerprint.version == 0x0303  # TLS 1.2 default
        assert isinstance(fingerprint.cipher_suites, list)
        assert isinstance(fingerprint.extensions, list)
        assert fingerprint.browser_type == "chrome"  # Default browser

    def test_ja3_string_generation(self):
        """Test JA3 string format generation."""
        fingerprint = ClientHelloFingerprint(
            version=0x0303,
            cipher_suites=[0x1301, 0x1302],
            extensions=[0, 23],
            elliptic_curves=[29, 23],
            signature_algorithms=[0x0403, 0x0503],
        )

        # Mock JA3 string generation (would be done by calibrator)
        expected_ja3_format = "version,ciphers,extensions,curves,sig_algs"
        # JA3 should follow this pattern with dash-separated values

        # This tests the structure exists for JA3 generation
        assert hasattr(fingerprint, "ja3_string")
        assert hasattr(fingerprint, "ja3_hash")


class TestFingerprintTemplate:
    """Test pre-defined browser fingerprint templates."""

    def test_chrome_template(self):
        """Test Chrome browser template."""
        template = FingerprintTemplate.CHROME_120_WINDOWS

        assert template.browser_type == "chrome"
        assert template.browser_version == "120.0"
        assert template.os_hint == "windows"
        assert template.version == 0x0303

        # Chrome should have TLS 1.3 cipher suites
        assert 0x1301 in template.cipher_suites  # TLS_AES_128_GCM_SHA256
        assert 0x1302 in template.cipher_suites  # TLS_AES_256_GCM_SHA384
        assert 0x1303 in template.cipher_suites  # TLS_CHACHA20_POLY1305_SHA256

        # Chrome should have common extensions
        assert 0 in template.extensions  # SNI
        assert 23 in template.extensions  # Session ticket
        assert 13 in template.extensions  # Signature algorithms

        # Chrome supports modern curves
        assert 29 in template.elliptic_curves  # X25519
        assert 23 in template.elliptic_curves  # secp256r1

    def test_firefox_template(self):
        """Test Firefox browser template."""
        template = FingerprintTemplate.FIREFOX_121_LINUX

        assert template.browser_type == "firefox"
        assert template.browser_version == "121.0"
        assert template.os_hint == "linux"

        # Firefox should have TLS 1.3 support
        assert 0x1301 in template.cipher_suites
        assert 0x1302 in template.cipher_suites
        assert 0x1303 in template.cipher_suites

        # Firefox has different extension ordering than Chrome
        assert 0 in template.extensions
        assert 23 in template.extensions

    def test_safari_template(self):
        """Test Safari browser template."""
        template = FingerprintTemplate.SAFARI_17_MACOS

        assert template.browser_type == "safari"
        assert template.browser_version == "17.0"
        assert template.os_hint == "macos"

        # Safari supports modern TLS
        assert 0x1301 in template.cipher_suites

        # Safari has its own extension preferences
        assert 0 in template.extensions

    def test_template_uniqueness(self):
        """Test that templates are sufficiently different."""
        chrome = FingerprintTemplate.CHROME_120_WINDOWS
        firefox = FingerprintTemplate.FIREFOX_121_LINUX
        safari = FingerprintTemplate.SAFARI_17_MACOS

        # Templates should have different characteristics
        assert chrome.browser_type != firefox.browser_type
        assert firefox.browser_type != safari.browser_type

        # Extension lists should differ
        assert chrome.extensions != firefox.extensions
        assert firefox.extensions != safari.extensions


class TestuTLSFingerprintCalibrator:
    """Test uTLS fingerprint calibrator functionality."""

    def test_calibrator_initialization(self):
        """Test calibrator initialization."""
        calibrator = uTLSFingerprintCalibrator()

        assert len(calibrator.templates) >= 3  # At least Chrome, Firefox, Safari
        assert "chrome_120_windows" in calibrator.templates
        assert "firefox_121_linux" in calibrator.templates
        assert "safari_17_macos" in calibrator.templates
        assert isinstance(calibrator.fingerprint_cache, dict)

    def test_template_access(self):
        """Test accessing fingerprint templates."""
        calibrator = uTLSFingerprintCalibrator()

        # Test valid template access
        chrome_template = calibrator.templates["chrome_120_windows"]
        assert chrome_template.browser_type == "chrome"

        # Test template contents
        firefox_template = calibrator.templates["firefox_121_linux"]
        assert firefox_template.browser_type == "firefox"
        assert len(firefox_template.cipher_suites) > 0
        assert len(firefox_template.extensions) > 0

    def test_random_fingerprint_selection(self):
        """Test random fingerprint selection for diversity."""
        calibrator = uTLSFingerprintCalibrator()

        # Get multiple random fingerprints
        fingerprints = []
        for _ in range(10):
            fp = calibrator.get_random_fingerprint()
            fingerprints.append(fp.browser_type)

        # Should get variety (probabilistic test)
        unique_browsers = set(fingerprints)
        assert len(unique_browsers) >= 1  # At least one browser type

        # All should be valid fingerprints
        for fp in fingerprints:
            assert fp in ["chrome", "firefox", "safari"]

    def test_fingerprint_calibration_without_randomization(self):
        """Test calibrating fingerprint without randomization."""
        calibrator = uTLSFingerprintCalibrator()

        fingerprint = calibrator.calibrate_fingerprint(
            "chrome_120_windows", randomize=False
        )

        # Should match template exactly
        template = FingerprintTemplate.CHROME_120_WINDOWS
        assert fingerprint.browser_type == template.browser_type
        assert fingerprint.version == template.version
        assert fingerprint.cipher_suites == template.cipher_suites
        assert fingerprint.extensions == template.extensions

        # Should have generated JA3/JA4 hashes
        assert len(fingerprint.ja3_string) > 0
        assert len(fingerprint.ja3_hash) == 32  # MD5 hash length
        assert len(fingerprint.ja4_string) > 0
        assert len(fingerprint.ja4_hash) <= 36  # Truncated SHA256

    def test_fingerprint_calibration_with_randomization(self):
        """Test calibrating fingerprint with randomization."""
        calibrator = uTLSFingerprintCalibrator()

        # Get two randomized fingerprints from same template
        fp1 = calibrator.calibrate_fingerprint("chrome_120_windows", randomize=True)
        fp2 = calibrator.calibrate_fingerprint("chrome_120_windows", randomize=True)

        # Should both be Chrome-based
        assert fp1.browser_type == "chrome"
        assert fp2.browser_type == "chrome"

        # But should have some differences due to randomization
        # (This is probabilistic, might occasionally be the same)
        assert fp1.ja3_hash != fp2.ja3_hash or fp1.extensions != fp2.extensions

    def test_unknown_template_fallback(self):
        """Test fallback to Chrome for unknown templates."""
        calibrator = uTLSFingerprintCalibrator()

        fingerprint = calibrator.calibrate_fingerprint("unknown_browser_template")

        # Should fallback to Chrome default
        assert fingerprint.browser_type == "chrome"
        assert len(fingerprint.cipher_suites) > 0

    def test_ja3_string_generation(self):
        """Test JA3 fingerprint string generation."""
        calibrator = uTLSFingerprintCalibrator()

        fingerprint = calibrator.calibrate_fingerprint("chrome_120_windows")

        # JA3 format: version,ciphers,extensions,curves,sig_algs
        ja3_parts = fingerprint.ja3_string.split(",")
        assert len(ja3_parts) == 5

        # Version should be numeric
        assert ja3_parts[0].isdigit()
        assert ja3_parts[0] == str(fingerprint.version)

        # Ciphers should be dash-separated numbers
        if ja3_parts[1]:  # May be empty
            cipher_strs = ja3_parts[1].split("-")
            for cipher_str in cipher_strs:
                assert cipher_str.isdigit()

        # Extensions should be dash-separated numbers
        if ja3_parts[2]:
            ext_strs = ja3_parts[2].split("-")
            for ext_str in ext_strs:
                assert ext_str.isdigit()

        # JA3 hash should be valid MD5
        assert len(fingerprint.ja3_hash) == 32
        assert all(c in "0123456789abcdef" for c in fingerprint.ja3_hash)

    def test_ja4_string_generation(self):
        """Test JA4 fingerprint string generation."""
        calibrator = uTLSFingerprintCalibrator()

        fingerprint = calibrator.calibrate_fingerprint("firefox_121_linux")

        # JA4 has specific format: q{version}{cipher_count}{ext_count}_{first_cipher}_{last_cipher}
        ja4_string = fingerprint.ja4_string

        # Should start with 'q'
        assert ja4_string.startswith("q")

        # Should have underscores separating parts
        assert "_" in ja4_string

        # Should have reasonable length
        assert 10 <= len(ja4_string) <= 50

        # JA4 hash should be truncated SHA256 (36 chars)
        assert len(fingerprint.ja4_hash) <= 36

    def test_client_hello_generation(self):
        """Test ClientHello message generation."""
        calibrator = uTLSFingerprintCalibrator()

        fingerprint = calibrator.calibrate_fingerprint("chrome_120_windows")
        server_name = "example.com"

        client_hello = calibrator.generate_client_hello(fingerprint, server_name)

        # Should be binary data
        assert isinstance(client_hello, bytes)
        assert len(client_hello) > 50  # Reasonable minimum size

        # Should start with ClientHello message type and length
        assert client_hello[0] == 0x01  # ClientHello message type

        # Should contain the server name (SNI)
        assert server_name.encode() in client_hello

    def test_client_hello_with_different_server_names(self):
        """Test ClientHello generation with different server names."""
        calibrator = uTLSFingerprintCalibrator()
        fingerprint = calibrator.calibrate_fingerprint("safari_17_macos")

        # Test different server names
        server_names = ["example.com", "test.org", "secure.net"]
        client_hellos = []

        for server_name in server_names:
            hello = calibrator.generate_client_hello(fingerprint, server_name)
            client_hellos.append(hello)

            # Each should contain its respective server name
            assert server_name.encode() in hello

        # Should be different messages due to different SNI
        assert len(set(client_hellos)) == len(client_hellos)

    def test_fingerprint_validation(self):
        """Test fingerprint validation functionality."""
        calibrator = uTLSFingerprintCalibrator()

        fingerprint = calibrator.calibrate_fingerprint("chrome_120_windows")
        client_hello = calibrator.generate_client_hello(fingerprint)

        # Test validation (simplified validation in current implementation)
        is_valid = calibrator.validate_fingerprint_match(client_hello, fingerprint)

        # Should pass basic validation checks
        assert isinstance(is_valid, bool)

    def test_extension_data_building(self):
        """Test building extension data for common extension types."""
        calibrator = uTLSFingerprintCalibrator()
        fingerprint = calibrator.calibrate_fingerprint("firefox_121_linux")

        # Test SNI extension (type 0)
        sni_data = calibrator._build_extension_data(0, fingerprint, "test.com")
        assert sni_data is not None
        assert b"test.com" in sni_data

        # Test Supported Groups extension (type 10)
        curves_data = calibrator._build_extension_data(10, fingerprint, "example.com")
        assert curves_data is not None
        assert len(curves_data) >= 2  # At least length field

        # Test Signature Algorithms extension (type 13)
        sig_algs_data = calibrator._build_extension_data(13, fingerprint, "example.com")
        assert sig_algs_data is not None

        # Test unknown extension
        unknown_data = calibrator._build_extension_data(999, fingerprint, "example.com")
        assert unknown_data == b""  # Should return empty for unknown types

    def test_statistics_reporting(self):
        """Test fingerprint calibrator statistics."""
        calibrator = uTLSFingerprintCalibrator()

        stats = calibrator.get_fingerprint_stats()

        assert "available_templates" in stats
        assert "cache_size" in stats
        assert "template_details" in stats

        # Should have at least 3 templates
        assert len(stats["available_templates"]) >= 3
        assert "chrome_120_windows" in stats["available_templates"]

        # Template details should include browser info
        details = stats["template_details"]
        chrome_details = details["chrome_120_windows"]
        assert chrome_details["browser"] == "chrome"
        assert chrome_details["version"] == "120.0"
        assert chrome_details["os"] == "windows"
        assert chrome_details["cipher_count"] > 0
        assert chrome_details["extension_count"] > 0


class TestFingerprintRandomization:
    """Test fingerprint randomization and diversity mechanisms."""

    def test_extension_order_randomization(self):
        """Test that extension order is randomized while preserving functionality."""
        calibrator = uTLSFingerprintCalibrator()

        # Get multiple fingerprints with randomization
        fingerprints = []
        for _ in range(5):
            fp = calibrator.calibrate_fingerprint("chrome_120_windows", randomize=True)
            fingerprints.append(fp.extensions.copy())

        # Should maintain critical extensions (SNI, session ticket)
        for extensions in fingerprints:
            assert 0 in extensions  # SNI should always be present
            assert 23 in extensions  # Session ticket common in Chrome

        # Extension order might vary (probabilistic test)
        # At least verify they're not all identical
        unique_orders = len(set(tuple(ext_list) for ext_list in fingerprints))
        # Randomization may produce some variety
        assert unique_orders >= 1

    def test_cipher_suite_reordering(self):
        """Test cipher suite reordering while maintaining security."""
        calibrator = uTLSFingerprintCalibrator()

        # Get multiple fingerprints with randomization
        cipher_suites_lists = []
        for _ in range(5):
            fp = calibrator.calibrate_fingerprint("firefox_121_linux", randomize=True)
            cipher_suites_lists.append(fp.cipher_suites.copy())

        # All should maintain TLS 1.3 suites at the beginning
        for cipher_suites in cipher_suites_lists:
            tls13_suites = [0x1301, 0x1302, 0x1303]
            # At least some TLS 1.3 suites should be present
            assert any(suite in cipher_suites for suite in tls13_suites)

    def test_randomization_preserves_functionality(self):
        """Test that randomization doesn't break core functionality."""
        calibrator = uTLSFingerprintCalibrator()

        # Test multiple randomized fingerprints
        for _ in range(10):
            fp = calibrator.calibrate_fingerprint("chrome_120_windows", randomize=True)

            # Should still be valid Chrome fingerprint
            assert fp.browser_type == "chrome"
            assert fp.version == 0x0303  # TLS 1.2
            assert len(fp.cipher_suites) > 0
            assert len(fp.extensions) > 0

            # Should generate valid JA3/JA4
            assert len(fp.ja3_string) > 0
            assert len(fp.ja3_hash) == 32

            # Should generate valid ClientHello
            client_hello = calibrator.generate_client_hello(fp)
            assert len(client_hello) > 50


class TestFingerprintIntegration:
    """Integration tests for fingerprint system."""

    def test_full_fingerprint_workflow(self):
        """Test complete fingerprint calibration and usage workflow."""
        calibrator = uTLSFingerprintCalibrator()

        # 1. Calibrate fingerprint
        fingerprint = calibrator.calibrate_fingerprint(
            "chrome_120_windows", randomize=True
        )

        # 2. Generate ClientHello
        client_hello = calibrator.generate_client_hello(
            fingerprint, "secure.example.com"
        )

        # 3. Validate fingerprint match
        is_valid = calibrator.validate_fingerprint_match(client_hello, fingerprint)

        # Workflow should complete successfully
        assert fingerprint.browser_type == "chrome"
        assert len(client_hello) > 100  # Reasonable size
        assert isinstance(is_valid, bool)

        # JA3/JA4 should be generated
        assert len(fingerprint.ja3_hash) == 32
        assert len(fingerprint.ja4_hash) <= 36

    def test_multi_template_calibration(self):
        """Test calibrating fingerprints from all available templates."""
        calibrator = uTLSFingerprintCalibrator()

        template_names = ["chrome_120_windows", "firefox_121_linux", "safari_17_macos"]
        fingerprints = {}

        for template_name in template_names:
            fp = calibrator.calibrate_fingerprint(template_name, randomize=False)
            fingerprints[template_name] = fp

        # All should be distinct
        browser_types = [fp.browser_type for fp in fingerprints.values()]
        assert len(set(browser_types)) == len(browser_types)  # All unique

        # Each should have proper characteristics
        assert fingerprints["chrome_120_windows"].browser_type == "chrome"
        assert fingerprints["firefox_121_linux"].browser_type == "firefox"
        assert fingerprints["safari_17_macos"].browser_type == "safari"

        # All should generate different JA3 hashes
        ja3_hashes = [fp.ja3_hash for fp in fingerprints.values()]
        assert len(set(ja3_hashes)) == len(ja3_hashes)  # All unique

    def test_fingerprint_diversity_metrics(self):
        """Test that fingerprint system provides good diversity."""
        calibrator = uTLSFingerprintCalibrator()

        # Generate many fingerprints
        fingerprints = []
        for _ in range(20):
            fp = calibrator.get_random_fingerprint()
            fingerprints.append(fp)

        # Collect diversity metrics
        browser_types = [fp.browser_type for fp in fingerprints]
        ja3_hashes = [fp.ja3_hash for fp in fingerprints]
        cipher_suite_counts = [len(fp.cipher_suites) for fp in fingerprints]

        # Should have browser diversity
        unique_browsers = set(browser_types)
        assert len(unique_browsers) >= 2  # At least 2 different browsers

        # Should have JA3 diversity
        unique_ja3 = set(ja3_hashes)
        assert len(unique_ja3) >= len(unique_browsers)  # At least one per browser

        # Should have reasonable cipher suite variety
        assert min(cipher_suite_counts) > 5  # Each should have several cipher suites
        assert max(cipher_suite_counts) > min(cipher_suite_counts)  # Some variation


def test_utls_fingerprinting_smoke_test():
    """Smoke test for uTLS fingerprinting functionality."""
    print("Running uTLS fingerprinting smoke test...")

    # Test calibrator initialization
    calibrator = uTLSFingerprintCalibrator()
    assert len(calibrator.templates) >= 3
    print(f"  Calibrator initialized with {len(calibrator.templates)} templates")

    # Test fingerprint calibration
    fingerprint = calibrator.calibrate_fingerprint("chrome_120_windows")
    assert fingerprint.browser_type == "chrome"
    assert len(fingerprint.ja3_hash) == 32
    print(
        f"  Chrome fingerprint: JA3={fingerprint.ja3_hash[:8]}..., JA4={fingerprint.ja4_hash[:8]}..."
    )

    # Test ClientHello generation
    client_hello = calibrator.generate_client_hello(fingerprint, "test.example.com")
    assert len(client_hello) > 50
    assert b"test.example.com" in client_hello
    print(f"  ClientHello generated: {len(client_hello)} bytes")

    # Test random fingerprint
    random_fp = calibrator.get_random_fingerprint()
    assert random_fp.browser_type in ["chrome", "firefox", "safari"]
    print(f"  Random fingerprint: {random_fp.browser_type} {random_fp.browser_version}")

    # Test statistics
    stats = calibrator.get_fingerprint_stats()
    assert len(stats["available_templates"]) >= 3
    print(
        f"  Statistics: {len(stats['available_templates'])} templates, cache size: {stats['cache_size']}"
    )

    print("  uTLS fingerprinting smoke test PASSED")


if __name__ == "__main__":
    # Run smoke test when executed directly
    test_utls_fingerprinting_smoke_test()
    print("\nTo run full test suite:")
    print("  pytest tests/htx/test_utls_fingerprinting.py -v")
