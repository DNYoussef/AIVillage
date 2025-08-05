"""Test Security Gate with dangerous inputs"""
from src.twin_runtime.guard import risk_gate

# Test dangerous patterns
dangerous_inputs = [
    "'; DROP TABLE users; --",
    "rm -rf /",
    "format c:",
    "<script>alert('XSS')</script>",
    "../../etc/passwd",
    "eval(__import__('os').system('rm -rf /'))",
    "DELETE FROM users WHERE 1=1",
    "os.system('shutdown -h now')",
    "subprocess.call(['rm', '-rf', '/'])",
    "exec('__import__(\"os\").system(\"rm -rf /\")')"
]

# Test sensitive data
sensitive_inputs = [
    "My SSN is 123-45-6789",
    "Credit card: 4111111111111111",
    "api_key=sk_live_xxxxx",
    "password=supersecret123",
    "private_key=-----BEGIN RSA PRIVATE KEY-----"
]

# Test safe inputs
safe_inputs = [
    "Hello, how are you?",
    "Calculate 2 + 2",
    "What's the weather today?"
]

print("=== Testing Dangerous Inputs ===")
dangerous_blocked = 0
for inp in dangerous_inputs:
    result = risk_gate({"content": inp, "type": "command"})
    print(f"Input: {inp[:50]}... -> {result}")
    if result in ["deny", "ask"]:
        dangerous_blocked += 1

print(f"\nBlocked {dangerous_blocked}/{len(dangerous_inputs)} dangerous inputs ({dangerous_blocked/len(dangerous_inputs)*100:.1f}%)")

print("\n=== Testing Sensitive Data ===")
sensitive_caught = 0
for inp in sensitive_inputs:
    result = risk_gate({"content": inp, "type": "text"})
    print(f"Input: {inp[:50]}... -> {result}")
    if result in ["deny", "ask"]:
        sensitive_caught += 1

print(f"\nCaught {sensitive_caught}/{len(sensitive_inputs)} sensitive inputs ({sensitive_caught/len(sensitive_inputs)*100:.1f}%)")

print("\n=== Testing Safe Inputs ===")
safe_allowed = 0
for inp in safe_inputs:
    result = risk_gate({"content": inp, "type": "text"})
    print(f"Input: {inp[:50]}... -> {result}")
    if result == "allow":
        safe_allowed += 1

print(f"\nAllowed {safe_allowed}/{len(safe_inputs)} safe inputs ({safe_allowed/len(safe_inputs)*100:.1f}%)")

# Test risk thresholds
print("\n=== Testing Risk Thresholds ===")
test_cases = [
    ({"content": "hello", "type": "text"}, 0.2),  # Low risk
    ({"content": "select * from users", "type": "query"}, 0.5),  # Medium risk
    ({"content": "DROP TABLE users; DELETE FROM logs;", "type": "command"}, 0.9)  # High risk
]

for msg, risk in test_cases:
    result = risk_gate(msg, risk)
    print(f"Risk {risk} -> {result}")