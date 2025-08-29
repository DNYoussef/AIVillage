# Secret Sanitization Validation Report

```
================================================================================
    SECRET SANITIZATION VALIDATION REPORT
================================================================================

Overall Status: PASS_WITH_WARNINGS

SUMMARY:
  Files Processed: 7/7
  Files with Issues: 2
  Total Issues: 27
  Validated Test Secrets: 89

VALIDATION CRITERIA:
  [OK] Test secrets must have '# pragma: allowlist secret' comments
  [OK] Test secrets should use 'test_' prefixes and obvious fake values
  [OK] No production-like secret patterns should remain
  [OK] API keys should be clearly marked as test/mock values

FILE: tests\security\test_auth_system.py
------------------------------------------------------------
  [FAIL] ISSUES FOUND (26)
    Line 54: ambiguous_secret
      hash1 = pm.hash_password(password)  # pragma: allowlist secret...
    Line 55: ambiguous_secret
      hash2 = pm.hash_password(password)  # pragma: allowlist secret...
    Line 67: ambiguous_secret
      password_hash = pm.hash_password(password)  # pragma: allowlist secret...
    Line 82: ambiguous_secret
      is_valid, errors = pm.validate_password_strength(strong_password, config)  # pragma: allowlist secre...
    Line 88: ambiguous_secret
      is_valid, errors = pm.validate_password_strength(weak_password, config)  # pragma: allowlist secret...
    Line 95: ambiguous_secret
      is_valid, errors = pm.validate_password_strength(no_upper, config)  # pragma: allowlist secret...
    Line 101: ambiguous_secret
      is_valid, errors = pm.validate_password_strength(no_symbols, config)  # pragma: allowlist secret...
    Line 180: ambiguous_secret
      assert secret1 != secret2  # pragma: allowlist secret...
    Line 233: ambiguous_secret
      )  # pragma: allowlist secret...
    Line 311: ambiguous_secret
      with pytest.raises(ValueError, match="Password validation failed"):  # pragma: allowlist secret...
    Line 313: ambiguous_secret
      username="weakpass", email="weak@example.com", password="weak"  # pragma: allowlist secret...
    Line 379: ambiguous_secret
      )  # pragma: allowlist secret...
    Line 384: ambiguous_secret
      username="locktest", password="WrongPassword", ip_address="127.0.0.1"  # pragma: allowlist secret...
    Line 409: ambiguous_secret
      ip_address="127.0.0.1",  # pragma: allowlist secret...
    Line 487: ambiguous_secret
      secret = self.auth_manager.enable_mfa(user.user_id)  # pragma: allowlist secret...
    Line 530: ambiguous_secret
      secret = self.auth_manager.enable_mfa(user.user_id)  # pragma: allowlist secret...
    Line 609: ambiguous_secret
      password_hash="hash",  # pragma: allowlist secret...
    Line 626: ambiguous_secret
      password_hash="hash",  # pragma: allowlist secret...
    Line 646: ambiguous_secret
      password_hash="hash",  # pragma: allowlist secret...
    Line 667: ambiguous_secret
      password_hash="hash",  # pragma: allowlist secret...
    Line 762: ambiguous_secret
      password=f"wrong_password_{i}",  # pragma: allowlist secret...
    Line 773: ambiguous_secret
      ip_address="192.168.1.100",  # pragma: allowlist secret...
    Line 796: ambiguous_secret
      hash_result = pm.hash_password(password)  # pragma: allowlist secret...
    Line 797: ambiguous_secret
      verification = pm.verify_password(password, hash_result)  # pragma: allowlist secret...
    Line 798: ambiguous_secret
      print(f"OK Password hashing and verification: {verification}")  # pragma: allowlist secret...
    Line 853: ambiguous_secret
      password_hash="hash",  # pragma: allowlist secret...
  [OK] Validated Secrets: 64

FILE: tests\integration\test_end_to_end_system.py
------------------------------------------------------------
  [PASS] No issues found
  [OK] Found 8 properly sanitized test secrets
  [OK] Validated Secrets: 8

FILE: tests\integration\test_integration_simple.py
------------------------------------------------------------
  [FAIL] ISSUES FOUND (1)
    Line 338: ambiguous_secret
      ip_address="127.0.0.1",  # pragma: allowlist secret...
  [OK] Validated Secrets: 6

FILE: tests\fixtures\service_fixtures.py
------------------------------------------------------------
  [PASS] No issues found
  [OK] Found 2 properly sanitized test secrets
  [OK] Validated Secrets: 2

FILE: tests\conftest_fixtures.py
------------------------------------------------------------
  [PASS] No issues found
  [OK] Found 2 properly sanitized test secrets
  [OK] Validated Secrets: 2

FILE: tests\test_federation_integration.py
------------------------------------------------------------
  [PASS] No issues found
  [OK] Found 1 properly sanitized test secrets
  [OK] Validated Secrets: 1

FILE: tests\benchmarks\test_performance_benchmarks.py
------------------------------------------------------------
  [PASS] No issues found
  [OK] Found 6 properly sanitized test secrets
  [OK] Validated Secrets: 6

RECOMMENDATIONS:
  1. Add '# pragma: allowlist secret' comments to all test credentials
  2. Replace production-like secrets with obvious test values
  3. Use 'test_' prefixes for all test passwords and keys
  4. Ensure API keys are clearly marked as 'test_mock_api_key'
  5. Verify no real credentials are in test files

VALIDATION COMPLETE
================================================================================
```
