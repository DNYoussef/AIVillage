#!/usr/bin/env python3
"""
Apply critical security patches to AIVillage codebase
Run this script to fix identified security vulnerabilities
"""

import os
import re
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityPatcher:
    """Apply security patches automatically."""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.backup_dir = self.base_path / "security_patches_backup"
        self.patches_applied = []

    def backup_file(self, file_path: Path):
        """Create backup before modifying file."""
        if not self.backup_dir.exists():
            self.backup_dir.mkdir(parents=True)

        backup_path = self.backup_dir / file_path.name
        shutil.copy2(file_path, backup_path)
        logger.info(f"Backed up {file_path} to {backup_path}")

    def apply_jwt_secret_fix(self):
        """Fix hardcoded JWT secrets."""
        files_to_fix = [
            "mcp_servers/hyperag/server.py",
            "mcp_servers/hyperag/mcp_server.py"
        ]

        for file_path in files_to_fix:
            full_path = self.base_path / file_path
            if not full_path.exists():
                logger.warning(f"File not found: {full_path}")
                continue

            self.backup_file(full_path)

            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Replace hardcoded secrets
            patterns = [
                (r'"dev-secret-change-in-production"', 'os.environ.get("AIVILLAGE_JWT_SECRET", "dev-fallback-secret")'),
                (r'"mcp-local-secret"', 'os.environ.get("AIVILLAGE_MCP_SECRET", "dev-fallback-secret")'),
                (r'jwt_secret="dev-secret-change-in-production"', 'jwt_secret=os.environ.get("AIVILLAGE_JWT_SECRET", "dev-fallback-secret")'),
            ]

            modified = False
            for pattern, replacement in patterns:
                if re.search(pattern, content):
                    content = re.sub(pattern, replacement, content)
                    modified = True

            # Add import if not present
            if modified and 'import os' not in content:
                content = 'import os\n' + content

            if modified:
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"Applied JWT secret fix to {file_path}")
                self.patches_applied.append(f"JWT secret fix: {file_path}")

    def apply_sql_injection_fix(self):
        """Fix SQL injection vulnerabilities."""
        file_path = self.base_path / "mcp_servers/hyperag/memory/hippo_index.py"

        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return

        self.backup_file(file_path)

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Fix SQL injection in query building
        sql_fixes = [
            # Fix 1: Replace f-string query construction
            (
                r'f"""[\s\S]*?WHERE {where_clause}[\s\S]*?"""',
                '"""\n                SELECT id, content, node_type, memory_type, confidence,\n                       embedding, created_at, last_accessed, access_count,\n                       user_id, gdc_flags, popularity_rank, importance_score,\n                       decay_rate, ttl, uncertainty, confidence_type, metadata\n                FROM hippo_nodes\n                WHERE """ + " AND ".join(["?" for _ in conditions]) + """\n                ORDER BY importance_score DESC, created_at DESC\n                LIMIT ?\n            """'
            ),
            # Add parameterized query helper method
            (
                r'(class HippoIndex.*?def __init__.*?\n)',
                r'\1\n    def _build_safe_query(self, conditions: list, order_by: str = "importance_score DESC, created_at DESC") -> str:\n        """Build safe parameterized query."""\n        base_query = """\n            SELECT id, content, node_type, memory_type, confidence,\n                   embedding, created_at, last_accessed, access_count,\n                   user_id, gdc_flags, popularity_rank, importance_score,\n                   decay_rate, ttl, uncertainty, confidence_type, metadata\n            FROM hippo_nodes\n        """\n        if conditions:\n            where_clause = " AND ".join(["?" for _ in conditions])\n            base_query += f" WHERE {where_clause}"\n        base_query += f" ORDER BY {order_by} LIMIT ?"\n        return base_query\n\n'
            )
        ]

        modified = False
        for pattern, replacement in sql_fixes:
            if re.search(pattern, content, re.DOTALL):
                content = re.sub(pattern, replacement, content, flags=re.DOTALL)
                modified = True

        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Applied SQL injection fix to {file_path}")
            self.patches_applied.append(f"SQL injection fix: {file_path}")

    def apply_weak_crypto_fix(self):
        """Fix weak cryptographic algorithms."""
        file_path = self.base_path / "mcp_servers/hyperag/models.py"

        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return

        self.backup_file(file_path)

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Replace MD5 with SHA-256
        crypto_fixes = [
            (r'hashlib\.md5\(([^)]+)\)\.hexdigest\(\)', r'hashlib.sha256(\1, usedforsecurity=True).hexdigest()'),
        ]

        modified = False
        for pattern, replacement in crypto_fixes:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                modified = True

        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Applied cryptographic fix to {file_path}")
            self.patches_applied.append(f"Weak crypto fix: {file_path}")

    def apply_model_download_fix(self):
        """Fix unsafe model downloads."""
        files_to_fix = [
            "mcp_servers/hyperag/lora/train_adapter.py",
            "mcp_servers/hyperag/repair/llm_driver.py"
        ]

        for file_path in files_to_fix:
            full_path = self.base_path / file_path
            if not full_path.exists():
                logger.warning(f"File not found: {full_path}")
                continue

            self.backup_file(full_path)

            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Add security parameters to model downloads
            model_fixes = [
                (
                    r'(AutoTokenizer\.from_pretrained\(\s*[^)]+)\)',
                    r'\1, trust_remote_code=False)'
                ),
                (
                    r'(AutoModelForCausalLM\.from_pretrained\(\s*[^)]+)\)',
                    r'\1, trust_remote_code=False)'
                ),
            ]

            modified = False
            for pattern, replacement in model_fixes:
                if re.search(pattern, content, re.DOTALL):
                    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
                    modified = True

            if modified:
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"Applied model download security fix to {file_path}")
                self.patches_applied.append(f"Model download fix: {file_path}")

    def apply_error_handling_fix(self):
        """Fix insecure error handling."""
        files_to_fix = [
            "mcp_servers/hyperag/guardian/audit.py",
            "communications/protocol.py"
        ]

        for file_path in files_to_fix:
            full_path = self.base_path / file_path
            if not full_path.exists():
                logger.warning(f"File not found: {full_path}")
                continue

            self.backup_file(full_path)

            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Replace broad exception handling
            error_fixes = [
                (
                    r'except Exception:\s*continue',
                    'except (IOError, OSError) as e:\n            logger.warning(f"File processing error: {type(e).__name__}")\n            continue'
                ),
                (
                    r'except Exception:\s*pass',
                    'except (IOError, OSError) as e:\n            logger.warning(f"Operation failed: {type(e).__name__}")\n            pass'
                ),
            ]

            modified = False
            for pattern, replacement in error_fixes:
                if re.search(pattern, content):
                    content = re.sub(pattern, replacement, content)
                    modified = True

            # Add logging import if not present
            if modified and 'import logging' not in content:
                content = 'import logging\n' + content
                if 'logger = logging.getLogger(__name__)' not in content:
                    # Add after imports
                    content = content.replace(
                        'import logging\n',
                        'import logging\n\nlogger = logging.getLogger(__name__)\n'
                    )

            if modified:
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"Applied error handling fix to {file_path}")
                self.patches_applied.append(f"Error handling fix: {file_path}")

    def create_security_env_template(self):
        """Create secure environment configuration template."""
        env_template = """# AIVillage Security Environment Configuration
# Copy to .env and configure with actual secure values

# CRITICAL: Generate secure secrets for production
# Use: openssl rand -base64 32

# JWT Authentication
AIVILLAGE_JWT_SECRET=your_jwt_secret_at_least_32_characters_long

# Database Security
AIVILLAGE_DB_PASSWORD=your_secure_database_password

# API Key Hashes (use SHA-256 of actual keys)
AIVILLAGE_DEV_KEY_HASH=sha256_hash_of_dev_api_key
AIVILLAGE_KING_KEY_HASH=sha256_hash_of_king_api_key
AIVILLAGE_SAGE_KEY_HASH=sha256_hash_of_sage_api_key

# Encryption
AIVILLAGE_ENCRYPTION_KEY=your_fernet_encryption_key

# MCP Server Security
AIVILLAGE_MCP_SECRET=your_mcp_server_secret

# Environment
ENVIRONMENT=production

# Security Monitoring
SECURITY_LOGGING_LEVEL=INFO
SECURITY_ALERT_WEBHOOK=your_webhook_url_for_alerts

# Model Security
TRUST_REMOTE_CODE=false
MODEL_REVISION_PINNING=true

# Network Security
WEBSOCKET_MAX_SIZE=65536
RATE_LIMIT_REQUESTS_PER_MINUTE=1000
SESSION_TIMEOUT_HOURS=1

# Database Security
DB_QUERY_TIMEOUT=30
DB_MAX_CONNECTIONS=20
"""

        env_file = self.base_path / ".env.security.template"
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(env_template)

        logger.info(f"Created security environment template: {env_file}")
        self.patches_applied.append("Security environment template created")

    def generate_security_report(self):
        """Generate security patch report."""
        report = f"""# Security Patches Applied - {os.environ.get('USER', 'unknown')}
Date: {os.popen('date').read().strip()}

## Patches Applied:
"""
        for patch in self.patches_applied:
            report += f"- {patch}\n"

        report += f"""
## Files Backed Up:
Backups stored in: {self.backup_dir}

## Next Steps:
1. Review applied patches in backed up files
2. Configure .env.security.template with actual secrets
3. Run security tests: python security_fixes.py
4. Deploy with new environment configuration
5. Monitor security logs for any issues

## Manual Actions Required:
1. Generate secure JWT secret: openssl rand -base64 32
2. Update database passwords
3. Configure API key hashes
4. Set up security monitoring
5. Review and test all patched functionality

## Verification Commands:
```bash
# Test JWT functionality
python -c "from security_fixes import SecureJWTManager; jwt = SecureJWTManager(); print('JWT OK')"

# Check for remaining hardcoded secrets
grep -r "dev-secret" mcp_servers/ || echo "No hardcoded secrets found"

# Verify SQL injection fixes
python -c "from security_fixes import SecureDatabaseManager; print('DB security OK')"
```
"""

        report_file = self.base_path / "SECURITY_PATCHES_REPORT.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"Generated security patch report: {report_file}")

    def apply_all_patches(self):
        """Apply all security patches."""
        logger.info("Starting application of security patches...")

        self.apply_jwt_secret_fix()
        self.apply_sql_injection_fix()
        self.apply_weak_crypto_fix()
        self.apply_model_download_fix()
        self.apply_error_handling_fix()
        self.create_security_env_template()
        self.generate_security_report()

        logger.info(f"Security patching complete! Applied {len(self.patches_applied)} patches.")
        logger.info("IMPORTANT: Configure .env.security.template before deployment!")

def main():
    """Main function to apply security patches."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    patcher = SecurityPatcher(base_path)

    print("🔒 AIVillage Security Patcher")
    print("=" * 40)
    print("This will apply critical security fixes to the codebase.")
    print(f"Backups will be created in: {patcher.backup_dir}")
    print()

    response = input("Continue with security patching? [y/N]: ")
    if response.lower() != 'y':
        print("Security patching cancelled.")
        return

    try:
        patcher.apply_all_patches()
        print("\n✅ Security patches applied successfully!")
        print("📋 Review SECURITY_PATCHES_REPORT.md for details.")
        print("⚠️  Configure .env.security.template before deployment.")

    except Exception as e:
        logger.error(f"Security patching failed: {e}")
        print(f"\n❌ Security patching failed: {e}")
        print("Check logs and backed up files for recovery.")

if __name__ == "__main__":
    main()
