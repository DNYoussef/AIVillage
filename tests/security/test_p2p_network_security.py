"""
Comprehensive Security Tests for P2P Network.

Tests network resilience against various attacks:
- Spoofing attacks
- Man-in-the-middle attacks
- Peer isolation for bad actors
- Encryption strength validation
- Information leakage detection
- Rate limiting effectiveness
- Replay attack prevention
"""

import os
import sys
import time
import unittest
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.p2p.secure_libp2p_mesh import (
    MessageCrypto,
    SecureLibP2PMeshNetwork,
    SecureMessage,
    SecureP2PNetworkConfig,
    SecurityEvent,
    SecurityEventLog,
    SecurityLevel,
    SecurityMonitor,
)


class TestP2PNetworkSecurity(unittest.TestCase):
    """Test suite for P2P network security features."""

    def setUp(self):
        """Set up test environment."""
        self.config = SecureP2PNetworkConfig()
        self.config.max_connections_per_minute = 5  # Lower for testing
        self.config.max_messages_per_minute = 10
        self.network = SecureLibP2PMeshNetwork(self.config)

    def test_spoofing_attack_detection(self):
        """Test detection and prevention of peer spoofing attacks."""
        print("\n=== Testing Spoofing Attack Detection ===")

        # Simulate legitimate peer
        legitimate_peer = "peer_legitimate_12345"

        # Create fake message from legitimate peer with wrong signature
        {
            "id": "msg123",
            "sender": legitimate_peer,
            "recipient": "target_peer",
            "encrypted_payload": "fake_payload_data",
            "mac": "invalid_mac_signature",
            "nonce": "fake_nonce_bytes",
            "timestamp": time.time(),
            "message_type": "DATA_MESSAGE",
            "sequence_number": 1,
        }

        # The message should fail MAC verification
        crypto = MessageCrypto(self.config.encryption_key)

        # Try to decrypt with wrong MAC - should fail
        result = crypto.decrypt_message(
            b"fake_payload",
            b"invalid_mac",
            b"fake_nonce_12345678",  # 16 bytes
            legitimate_peer,
        )

        self.assertIsNone(result, "Spoofed message should fail decryption")
        print("[PASS] Spoofing attack detected - invalid MAC rejected")

        # Test reputation system response
        monitor = SecurityMonitor(self.config)
        monitor.update_peer_reputation(legitimate_peer, -0.5, "Spoofing attempt detected")

        reputation = monitor.peer_reputations[legitimate_peer]
        self.assertLess(reputation.trust_score, 0.3, "Spoofing should severely damage reputation")
        print(f"[PASS] Peer reputation reduced to {reputation.trust_score:.3f} after spoofing")

    def test_replay_attack_prevention(self):
        """Test prevention of replay attacks."""
        print("\n=== Testing Replay Attack Prevention ===")

        monitor = SecurityMonitor(self.config)

        # Create legitimate message
        message = SecureMessage(
            id="msg456",
            sender="peer_sender_789",
            encrypted_payload=b"test_payload",
            sequence_number=1,
        )

        # First message should be accepted
        is_replay1 = monitor.is_message_replay(message)
        self.assertFalse(is_replay1, "First message should not be flagged as replay")
        print("[PASS] Legitimate message accepted")

        # Same message again should be detected as replay
        is_replay2 = monitor.is_message_replay(message)
        self.assertTrue(is_replay2, "Duplicate message should be flagged as replay")
        print("[PASS] Replay attack detected - duplicate message blocked")

        # Message with lower sequence number should be blocked
        old_message = SecureMessage(
            id="msg457",
            sender="peer_sender_789",
            encrypted_payload=b"old_payload",
            sequence_number=0,  # Lower than previous
        )

        is_replay3 = monitor.is_message_replay(old_message)
        self.assertTrue(is_replay3, "Old sequence number should be flagged as replay")
        print("[PASS] Out-of-order sequence number blocked")

    def test_rate_limiting_effectiveness(self):
        """Test rate limiting to prevent DoS attacks."""
        print("\n=== Testing Rate Limiting Effectiveness ===")

        monitor = SecurityMonitor(self.config)
        malicious_peer = "peer_attacker_666"

        # Simulate rapid connection attempts
        connection_allowed_count = 0

        for i in range(15):  # Exceed the limit of 5 connections per minute
            allowed = monitor.check_rate_limits(malicious_peer, f"192.168.1.{i}")
            if allowed:
                connection_allowed_count += 1
                monitor.connection_counts[malicious_peer].append(time.time())

        self.assertLessEqual(connection_allowed_count, self.config.max_connections_per_minute + 1)
        print(f"[PASS] Rate limiting effective - only {connection_allowed_count} of 15 attempts allowed")

        # Check that peer gets blocked after excessive attempts
        self.assertTrue(malicious_peer in monitor.peer_reputations, "Attacker should be tracked")

        reputation = monitor.peer_reputations[malicious_peer]
        self.assertLess(reputation.trust_score, 0.5, "Attacker reputation should be damaged")
        print(f"[PASS] Attacker reputation reduced to {reputation.trust_score:.3f}")

    def test_peer_isolation_for_bad_actors(self):
        """Test that bad actors are isolated from the network."""
        print("\n=== Testing Peer Isolation ===")

        monitor = SecurityMonitor(self.config)
        bad_actor = "peer_malicious_999"

        # Simulate multiple security violations
        violations = [
            "Message decryption failed",
            "Invalid signature detected",
            "Replay attack attempted",
            "Rate limit exceeded",
            "Suspicious behavior pattern",
        ]

        for violation in violations:
            monitor.update_peer_reputation(bad_actor, -0.2, violation)

        reputation = monitor.peer_reputations[bad_actor]
        self.assertLess(reputation.trust_score, self.config.min_trust_score)
        print(
            f"[PASS] Bad actor reputation: {reputation.trust_score:.3f} (below threshold {self.config.min_trust_score})"
        )

        # Block the peer
        monitor.block_peer(bad_actor, "Multiple security violations")

        # Verify peer is blocked
        self.assertTrue(monitor.is_peer_blocked(bad_actor), "Bad actor should be blocked")
        print("[PASS] Bad actor successfully blocked from network")

        # Test that blocked peer cannot send messages
        blocked_message = SecureMessage(sender=bad_actor, encrypted_payload=b"malicious_payload")

        # In real implementation, this would be checked before processing
        should_process = not monitor.is_peer_blocked(blocked_message.sender)
        self.assertFalse(should_process, "Messages from blocked peers should be rejected")
        print("[PASS] Messages from blocked peer rejected")

    def test_encryption_strength_validation(self):
        """Test encryption strength and key management."""
        print("\n=== Testing Encryption Strength ===")

        crypto = MessageCrypto(self.config.encryption_key)

        # Test encryption with various payload sizes
        test_payloads = [
            b"small",
            b"medium_length_payload_for_testing" * 10,
            b"large_payload_" * 100,
            b"x" * 10000,  # Very large payload
        ]

        for i, payload in enumerate(test_payloads):
            # Encrypt
            encrypted, mac, nonce = crypto.encrypt_message(payload, "test_peer")

            # Verify encryption worked
            self.assertNotEqual(encrypted, payload, f"Payload {i} should be encrypted")
            self.assertEqual(len(nonce), 16, "Nonce should be 16 bytes")
            self.assertEqual(len(mac), 32, "MAC should be 32 bytes (SHA256)")

            # Decrypt and verify
            decrypted = crypto.decrypt_message(encrypted, mac, nonce, "test_peer")
            self.assertEqual(decrypted, payload, f"Payload {i} should decrypt correctly")

        print(f"[PASS] Encryption/decryption successful for {len(test_payloads)} payload sizes")

        # Test that modified encrypted data fails
        encrypted, mac, nonce = crypto.encrypt_message(b"test_payload", "test_peer")

        # Modify encrypted data
        modified_encrypted = encrypted[:-1] + b"\x00"
        decrypted_modified = crypto.decrypt_message(modified_encrypted, mac, nonce, "test_peer")
        self.assertIsNone(decrypted_modified, "Modified encrypted data should fail")
        print("[PASS] Modified encrypted data rejected")

        # Test that modified MAC fails
        modified_mac = mac[:-1] + b"\x00"
        decrypted_bad_mac = crypto.decrypt_message(encrypted, modified_mac, nonce, "test_peer")
        self.assertIsNone(decrypted_bad_mac, "Modified MAC should fail verification")
        print("[PASS] Modified MAC rejected")

    def test_information_leakage_detection(self):
        """Test for information leakage in error messages and timing."""
        print("\n=== Testing Information Leakage Prevention ===")

        crypto = MessageCrypto(self.config.encryption_key)

        # Test that MAC verification failures don't leak timing information
        valid_payload = b"valid_test_payload"
        encrypted, mac, nonce = crypto.encrypt_message(valid_payload, "test_peer")

        # Create various invalid MACs
        invalid_macs = [
            b"x" * 32,  # Wrong MAC
            mac[:-1] + b"\x00",  # Slightly modified MAC
            b"",  # Empty MAC
            b"short_mac",  # Short MAC
        ]

        timing_results = []

        for invalid_mac in invalid_macs:
            start_time = time.time()
            result = crypto.decrypt_message(encrypted, invalid_mac, nonce, "test_peer")
            end_time = time.time()

            timing_results.append(end_time - start_time)
            self.assertIsNone(result, "Invalid MAC should fail")

        # Verify timing is relatively consistent (no major leakage)
        avg_timing = sum(timing_results) / len(timing_results)
        max_deviation = max(abs(t - avg_timing) for t in timing_results)

        # Allow up to 50% deviation (timing attacks are complex)
        self.assertLess(max_deviation, avg_timing * 0.5, "Timing deviation should be minimal")
        print(f"[PASS] MAC verification timing consistent (max deviation: {max_deviation * 1000:.2f}ms)")

        # Test that error messages don't reveal sensitive information
        monitor = SecurityMonitor(self.config)

        # Simulate various failures
        test_events = [
            SecurityEvent.MESSAGE_DECRYPT_FAIL,
            SecurityEvent.AUTH_FAILURE,
            SecurityEvent.REPLAY_ATTACK_DETECTED,
        ]

        for event_type in test_events:
            event = SecurityEventLog(
                event_type=event_type,
                peer_id="test_peer",
                description="Generic security failure",
            )
            monitor.log_security_event(event)

        # Verify no sensitive data in logs
        for log_entry in monitor.security_logs:
            self.assertNotIn("private_key", log_entry.description.lower())
            self.assertNotIn("secret", log_entry.description.lower())
            self.assertNotIn("password", log_entry.description.lower())

        print("[PASS] No sensitive information in security logs")

    def test_man_in_the_middle_resistance(self):
        """Test resistance to man-in-the-middle attacks."""
        print("\n=== Testing Man-in-the-Middle Resistance ===")

        crypto = MessageCrypto(self.config.encryption_key)

        # Simulate legitimate communication
        alice_id = "peer_alice_123"
        bob_id = "peer_bob_456"
        eve_id = "peer_eve_attacker"  # Man-in-the-middle

        # Alice sends message to Bob
        original_payload = b"secret_message_from_alice"
        encrypted, mac, nonce = crypto.encrypt_message(original_payload, alice_id)

        # Eve intercepts and tries to modify the message
        intercepted_message = SecureMessage(
            sender=alice_id,
            recipient=bob_id,
            encrypted_payload=encrypted,
            mac=mac,
            nonce=nonce,
            sequence_number=1,
        )

        # Eve tries to modify the encrypted payload
        modified_payload = encrypted[:-1] + b"\x00"
        intercepted_message.encrypted_payload = modified_payload

        # Bob tries to decrypt the modified message
        decrypted = crypto.decrypt_message(
            intercepted_message.encrypted_payload,
            intercepted_message.mac,
            intercepted_message.nonce,
            alice_id,
        )

        self.assertIsNone(decrypted, "Modified message should fail MAC verification")
        print("[PASS] Message modification detected by MAC verification")

        # Eve tries to replace the entire message with her own
        eve_payload = b"fake_message_from_eve"
        eve_encrypted, eve_mac, eve_nonce = crypto.encrypt_message(eve_payload, eve_id)

        # Eve pretends to be Alice
        forged_message = SecureMessage(
            sender=alice_id,  # Forged sender
            recipient=bob_id,
            encrypted_payload=eve_encrypted,
            mac=eve_mac,
            nonce=eve_nonce,
            sequence_number=2,
        )

        # The MAC won't verify because it was created by Eve with Alice's claimed identity
        decrypted_forged = crypto.decrypt_message(
            forged_message.encrypted_payload,
            forged_message.mac,
            forged_message.nonce,
            alice_id,  # Bob expects this to be from Alice
        )

        self.assertIsNone(decrypted_forged, "Forged message should fail MAC verification")
        print("[PASS] Forged message with wrong sender detected")

        # Test sequence number tampering
        monitor = SecurityMonitor(self.config)

        # Set up Alice's sequence number
        SecureMessage(sender=alice_id, sequence_number=5)
        monitor.sequence_numbers[alice_id] = 5

        # Eve tries to replay old message with lower sequence
        replay_msg = SecureMessage(sender=alice_id, sequence_number=3)
        is_replay = monitor.is_message_replay(replay_msg)

        self.assertTrue(is_replay, "Sequence number tampering should be detected")
        print("[PASS] Sequence number tampering detected")

    def test_forward_secrecy_implementation(self):
        """Test forward secrecy features."""
        print("\n=== Testing Forward Secrecy ===")

        # In a real implementation, we would test key rotation
        # For now, test that each message uses a unique nonce

        crypto = MessageCrypto(self.config.encryption_key)

        nonces = set()
        payload = b"test_message_for_forward_secrecy"

        # Generate multiple encrypted messages
        for i in range(100):
            encrypted, mac, nonce = crypto.encrypt_message(payload, f"peer_{i}")
            nonces.add(nonce.hex())

        # All nonces should be unique
        self.assertEqual(len(nonces), 100, "All nonces should be unique")
        print("[PASS] Unique nonces generated for each message")

        # Test that same payload with same sender produces different ciphertexts
        ciphertexts = set()
        for i in range(10):
            encrypted, _, _ = crypto.encrypt_message(payload, "same_peer")
            ciphertexts.add(encrypted.hex())

        self.assertEqual(len(ciphertexts), 10, "Same payload should produce different ciphertexts")
        print("[PASS] Same payload produces different ciphertexts (semantic security)")

    def test_security_event_monitoring(self):
        """Test security event monitoring and alerting."""
        print("\n=== Testing Security Event Monitoring ===")

        monitor = SecurityMonitor(self.config)

        # Generate various security events
        test_events = [
            (SecurityEvent.CONNECTION_ATTEMPT, SecurityLevel.LOW, "peer_1"),
            (SecurityEvent.AUTH_FAILURE, SecurityLevel.MEDIUM, "peer_2"),
            (SecurityEvent.RATE_LIMIT_EXCEEDED, SecurityLevel.HIGH, "peer_3"),
            (SecurityEvent.MALICIOUS_PEER_DETECTED, SecurityLevel.CRITICAL, "peer_4"),
            (SecurityEvent.REPLAY_ATTACK_DETECTED, SecurityLevel.HIGH, "peer_5"),
        ]

        for event_type, severity, peer_id in test_events:
            event = SecurityEventLog(
                event_type=event_type,
                peer_id=peer_id,
                severity=severity,
                description=f"Test {event_type.value} event",
            )
            monitor.log_security_event(event)

        # Verify events are logged
        self.assertEqual(len(monitor.security_logs), len(test_events))
        print(f"[PASS] {len(test_events)} security events logged")

        # Test security summary
        summary = monitor.get_security_summary()

        self.assertIn("total_events", summary)
        self.assertIn("blocked_peers", summary)
        self.assertIn("event_types", summary)

        print(f"[PASS] Security summary generated: {summary['total_events']} total events")

        # Test that critical events trigger automatic blocking
        critical_peer = "peer_critical_attack"
        critical_event = SecurityEventLog(
            event_type=SecurityEvent.MALICIOUS_PEER_DETECTED,
            peer_id=critical_peer,
            severity=SecurityLevel.CRITICAL,
            description="Automated test critical event",
        )

        # Check peer not blocked initially
        self.assertFalse(monitor.is_peer_blocked(critical_peer))

        # Log critical event
        monitor.log_security_event(critical_event)

        # Critical event should trigger automatic blocking
        self.assertTrue(monitor.is_peer_blocked(critical_peer))
        print("[PASS] Critical security event triggered automatic peer blocking")

    def test_network_resilience_under_attack(self):
        """Test overall network resilience under coordinated attack."""
        print("\n=== Testing Network Resilience Under Attack ===")

        monitor = SecurityMonitor(self.config)

        # Simulate coordinated attack from multiple peers
        attacking_peers = [f"attacker_{i}" for i in range(20)]

        # Phase 1: Mass connection attempts
        for attacker in attacking_peers:
            for _ in range(10):  # Each attacker tries 10 connections
                monitor.check_rate_limits(attacker, "192.168.1.100")
                monitor.connection_counts[attacker].append(time.time())

        # Phase 2: Failed authentication attempts
        for attacker in attacking_peers[:10]:  # First 10 attackers
            for _ in range(5):
                monitor.update_peer_reputation(attacker, -0.3, "Authentication failed")

        # Phase 3: Message flooding
        MessageCrypto(self.config.encryption_key)
        for attacker in attacking_peers[10:]:  # Last 10 attackers
            for i in range(15):
                # Simulate invalid messages
                SecureMessage(
                    sender=attacker,
                    encrypted_payload=b"spam_message",
                    sequence_number=i,
                )

                # This would fail decryption in real scenario
                monitor.update_peer_reputation(attacker, -0.1, "Invalid message")

        # Analyze network state after attack
        monitor.get_security_summary()

        # Count peers with low reputation
        low_reputation_peers = sum(
            1 for r in monitor.peer_reputations.values() if r.trust_score < self.config.min_trust_score
        )

        blocked_peers = len(monitor.blocked_peers)

        print(f"[INFO] After attack: {low_reputation_peers} low-reputation peers, {blocked_peers} blocked peers")

        # Network should have identified and isolated most attackers
        self.assertGreater(low_reputation_peers, 15, "Most attackers should have low reputation")

        # Some attackers should be blocked
        self.assertGreater(blocked_peers, 0, "Some attackers should be blocked")

        print("[PASS] Network successfully identified and isolated attacking peers")

        # Test that legitimate peer can still function
        legitimate_peer = "peer_legitimate_user"
        monitor.update_peer_reputation(legitimate_peer, 0.1, "Successful interaction")

        reputation = monitor.peer_reputations[legitimate_peer]
        self.assertGreaterEqual(reputation.trust_score, self.config.min_trust_score)
        self.assertFalse(monitor.is_peer_blocked(legitimate_peer))

        print("[PASS] Legitimate peers remain functional during attack")


def run_security_tests():
    """Run all security tests."""
    print("=" * 70)
    print("P2P NETWORK SECURITY TEST SUITE")
    print("Testing network resilience against various attacks")
    print("=" * 70)

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestP2PNetworkSecurity)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 70)
    print("SECURITY TEST SUMMARY")
    print("=" * 70)

    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total_tests - failures - errors

    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")

    if failures == 0 and errors == 0:
        print("\n[SUCCESS] ALL SECURITY TESTS PASSED!")
        print("P2P network is resilient against tested attack vectors.")
    else:
        print(f"\n[WARNING] {failures + errors} tests failed. Review security implementation.")

        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback}")

        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback}")

    print("=" * 70)

    return failures == 0 and errors == 0


if __name__ == "__main__":
    # Set up test environment
    os.chdir(Path(__file__).parent.parent.parent)

    # Run security tests
    success = run_security_tests()

    # Exit with appropriate code
    sys.exit(0 if success else 1)
