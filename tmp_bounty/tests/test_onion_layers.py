"""Test Onion Layer Encryption for Betanet

Tests the real X25519 + ChaCha20-Poly1305 onion routing implementation
that replaces the JSON placeholder encryption.
"""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Direct module loading to avoid import chain issues
import importlib.util


def load_module_direct(name, path):
    """Load a module directly from file path"""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load onion crypto module
src_path = Path(__file__).parent.parent.parent / "src"
onion_crypto = load_module_direct("onion_crypto", src_path / "core/p2p/crypto/onion.py")

# Import classes
OnionCrypto = onion_crypto.OnionCrypto
OnionHop = onion_crypto.OnionHop
build_onion_layers = onion_crypto.build_onion_layers
peel_onion_layer = onion_crypto.peel_onion_layer
generate_keypair = onion_crypto.generate_keypair
CRYPTO_AVAILABLE = onion_crypto.CRYPTO_AVAILABLE


class TestOnionLayers:
    """Test onion routing cryptography"""

    def test_crypto_availability(self):
        """Test that cryptographic primitives are available"""
        print(f"Crypto available: {CRYPTO_AVAILABLE}")
        if not CRYPTO_AVAILABLE:
            pytest.skip("Cryptography library not available")

    def test_keypair_generation(self):
        """Test X25519 keypair generation"""
        if not CRYPTO_AVAILABLE:
            pytest.skip("Crypto not available")

        private_key, public_key = generate_keypair()

        assert len(private_key) == 32, "Private key should be 32 bytes"
        assert len(public_key) == 32, "Public key should be 32 bytes"
        assert private_key != public_key, "Private and public keys should be different"

        # Generate another pair to ensure uniqueness
        private_key2, public_key2 = generate_keypair()
        assert private_key != private_key2, "Keys should be unique"
        assert public_key != public_key2, "Public keys should be unique"

    def test_round_trip_through_3_hops(self):
        """Test round-trip encryption/decryption through 3 hops"""
        if not CRYPTO_AVAILABLE:
            pytest.skip("Crypto not available")

        print("\n=== Testing 3-Hop Onion Routing ===")

        # Generate keypairs for 3 hops + destination
        hop_keys = []
        crypto_instances = []

        for i in range(4):  # 3 intermediate hops + 1 destination
            crypto = OnionCrypto()
            hop_id = f"hop_{i}" if i < 3 else "destination"
            public_key = crypto.get_public_key_bytes()

            hop_keys.append((hop_id, public_key))
            crypto_instances.append(crypto)
            print(f"Generated keys for {hop_id}")

        # Original payload
        original_payload = b"Secret message through 3 hops!"
        print(f"Original payload: {len(original_payload)} bytes")

        # Build onion layers
        onion = build_onion_layers(original_payload, hop_keys)
        print(f"Built onion: {len(onion)} bytes")

        # Verify onion is larger than original (due to encryption overhead)
        assert len(onion) > len(original_payload), (
            "Onion should be larger than original payload"
        )

        # Peel layers one by one
        current_onion = onion

        for i, (hop_id, _) in enumerate(hop_keys):
            crypto = crypto_instances[i]

            next_hop, inner_payload = crypto.peel_layer(current_onion)

            if i < len(hop_keys) - 1:
                # Intermediate hop
                expected_next_hop = hop_keys[i + 1][0]
                assert next_hop == expected_next_hop, f"Next hop mismatch at layer {i}"
                print(
                    f"Hop {i} ({hop_id}): next_hop={next_hop}, payload_size={len(inner_payload)}"
                )
                current_onion = inner_payload
            else:
                # Final destination
                assert next_hop is None, "Final hop should have no next hop"
                assert inner_payload == original_payload, (
                    "Final payload should match original"
                )
                print("Final destination: payload matches original ✓")

        print("3-hop onion routing test passed! ✓")

    def test_tamper_detection_aead_fail(self):
        """Test that tampering with encrypted layers is detected"""
        if not CRYPTO_AVAILABLE:
            pytest.skip("Crypto not available")

        print("\n=== Testing Tamper Detection ===")

        # Create simple 2-hop route
        crypto1 = OnionCrypto()
        crypto2 = OnionCrypto()

        hop_keys = [
            ("hop1", crypto1.get_public_key_bytes()),
            ("hop2", crypto2.get_public_key_bytes()),
        ]

        original_payload = b"Tamper test message"
        onion = build_onion_layers(original_payload, hop_keys)

        # Tamper with the onion data
        tampered_onion = bytearray(onion)
        tampered_onion[50] = tampered_onion[50] ^ 0xFF  # Flip bits
        tampered_onion = bytes(tampered_onion)

        print(f"Tampered with byte 50: {onion[50]:02x} -> {tampered_onion[50]:02x}")

        # Attempt to decrypt tampered onion should fail
        try:
            next_hop, inner = crypto1.peel_layer(tampered_onion)
            # If we get here without exception, the AEAD didn't catch tampering
            pytest.fail("AEAD should have detected tampering")
        except Exception as e:
            print(f"Tamper detection successful: {type(e).__name__}: {e}")
            # This is expected - AEAD should fail on tampered data

        print("Tamper detection test passed! ✓")

    def test_layer_isolation(self):
        """Test that each layer only reveals next hop, not full route"""
        if not CRYPTO_AVAILABLE:
            pytest.skip("Crypto not available")

        print("\n=== Testing Layer Isolation ===")

        # Create 4-hop route
        hop_names = ["alice", "bob", "charlie", "destination"]
        crypto_instances = [OnionCrypto() for _ in hop_names]

        hop_keys = [
            (name, crypto.get_public_key_bytes())
            for name, crypto in zip(hop_names, crypto_instances, strict=False)
        ]

        original_payload = b"Layer isolation test"
        onion = build_onion_layers(original_payload, hop_keys)

        # Each hop should only see the next hop
        current_onion = onion

        for i, (hop_name, crypto) in enumerate(
            zip(hop_names, crypto_instances, strict=False)
        ):
            next_hop, inner = crypto.peel_layer(current_onion)

            if i < len(hop_names) - 1:
                expected_next = hop_names[i + 1]
                assert next_hop == expected_next, (
                    f"Hop {hop_name} should only see next hop"
                )

                # Verify this hop can't see beyond next hop
                # (The inner payload should be encrypted for subsequent hops)
                if i < len(hop_names) - 2:
                    # Try to decode inner as if it were a direct message
                    # It should not contain readable route information
                    assert hop_names[i + 2].encode() not in inner, (
                        f"Hop {hop_name} should not see beyond next hop"
                    )

                print(f"Hop {hop_name}: can only see next hop {next_hop} ✓")
                current_onion = inner
            else:
                # Final hop sees original payload
                assert next_hop is None
                assert inner == original_payload
                print("Destination: received original payload ✓")

        print("Layer isolation test passed! ✓")

    def test_variable_payload_sizes(self):
        """Test onion routing with different payload sizes"""
        if not CRYPTO_AVAILABLE:
            pytest.skip("Crypto not available")

        print("\n=== Testing Variable Payload Sizes ===")

        # Create 2-hop route
        crypto1 = OnionCrypto()
        crypto2 = OnionCrypto()
        hop_keys = [
            ("hop1", crypto1.get_public_key_bytes()),
            ("hop2", crypto2.get_public_key_bytes()),
        ]

        # Test different payload sizes
        test_sizes = [1, 16, 256, 1024, 4096]

        for size in test_sizes:
            payload = b"x" * size

            # Build and test onion
            onion = build_onion_layers(payload, hop_keys)

            # Peel layers
            next_hop, inner = crypto1.peel_layer(onion)
            assert next_hop == "hop2"

            final_next, final_payload = crypto2.peel_layer(inner)
            assert final_next is None
            assert final_payload == payload

            print(f"Payload size {size} bytes: OK ✓")

        print("Variable payload size test passed! ✓")


# Standalone test runner
def run_onion_tests():
    """Run onion layer tests standalone"""
    print("=" * 60)
    print("ONION LAYER ENCRYPTION TESTS")
    print("=" * 60)

    test_suite = TestOnionLayers()

    try:
        # Test 1: Crypto availability
        print("\n1. Testing crypto availability...")
        test_suite.test_crypto_availability()
        print("   ✓ Cryptographic primitives available")

        # Test 2: Keypair generation
        print("\n2. Testing keypair generation...")
        test_suite.test_keypair_generation()
        print("   ✓ X25519 keypair generation working")

        # Test 3: 3-hop round trip
        print("\n3. Testing 3-hop round trip...")
        test_suite.test_round_trip_through_3_hops()
        print("   ✓ 3-hop onion routing successful")

        # Test 4: Tamper detection
        print("\n4. Testing tamper detection...")
        test_suite.test_tamper_detection_aead_fail()
        print("   ✓ AEAD tamper detection working")

        # Test 5: Layer isolation
        print("\n5. Testing layer isolation...")
        test_suite.test_layer_isolation()
        print("   ✓ Each hop only sees next hop")

        # Test 6: Variable payload sizes
        print("\n6. Testing variable payload sizes...")
        test_suite.test_variable_payload_sizes()
        print("   ✓ Different payload sizes work")

        print("\n" + "=" * 60)
        print("ALL ONION LAYER TESTS PASSED! ✓")
        print("Real X25519 + ChaCha20-Poly1305 onion routing is working.")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_onion_tests()
    sys.exit(0 if success else 1)
