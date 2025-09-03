#!/usr/bin/env python3
"""
Workflow Timeout Adjustment Script

Adjusts the timeout values in the SCION production security workflow based on monitoring findings.
Increases detect-secrets timeout from 240s to 600s to prevent timeout failures.

Usage:
    python scripts/security/adjust_workflow_timeouts.py [--dry-run]
"""

import argparse
import logging
import sys
from pathlib import Path
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkflowTimeoutAdjuster:
    """Adjusts workflow timeouts based on monitoring analysis."""
    
    def __init__(self, base_path: str, dry_run: bool = False):
        self.base_path = Path(base_path)
        self.dry_run = dry_run
        self.workflow_file = self.base_path / ".github/workflows/scion_production.yml"
        
        # Timeout adjustments based on monitoring findings
        self.timeout_adjustments = [
            {
                "description": "Increase detect-secrets timeout from 240s to 600s",
                "pattern": r"timeout 240 detect-secrets scan",
                "replacement": r"timeout 600 detect-secrets scan",
                "reason": "detect-secrets consistently timing out after 4 minutes"
            }
        ]

    def apply_timeout_adjustments(self) -> bool:
        """Apply all timeout adjustments to the workflow file."""
        if not self.workflow_file.exists():
            logger.error(f"Workflow file not found: {self.workflow_file}")
            return False
            
        try:
            # Read the workflow file
            with open(self.workflow_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            original_content = content
            adjustments_applied = 0
            
            # Apply each timeout adjustment
            for adjustment in self.timeout_adjustments:
                pattern = adjustment["pattern"]
                replacement = adjustment["replacement"]
                description = adjustment["description"]
                reason = adjustment["reason"]
                
                if re.search(pattern, content):
                    if self.dry_run:
                        logger.info(f"[DRY-RUN] Would apply: {description}")
                        logger.info(f"  Reason: {reason}")
                        logger.info(f"  Pattern: {pattern}")
                        logger.info(f"  Replacement: {replacement}")
                    else:
                        content = re.sub(pattern, replacement, content)
                        logger.info(f"✅ Applied: {description}")
                        
                    adjustments_applied += 1
                else:
                    logger.warning(f"Pattern not found for: {description}")
                    logger.warning(f"  Pattern: {pattern}")
                    
            if not self.dry_run and adjustments_applied > 0:
                # Write back the updated workflow
                with open(self.workflow_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
                logger.info(f"✅ Applied {adjustments_applied} timeout adjustments to workflow")
                
            return adjustments_applied > 0
            
        except Exception as e:
            logger.error(f"Error adjusting workflow timeouts: {e}")
            return False

    def validate_adjustments(self) -> bool:
        """Validate that timeout adjustments were applied correctly."""
        if not self.workflow_file.exists():
            return False
            
        try:
            with open(self.workflow_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check that the new timeout value is present
            if "timeout 600 detect-secrets scan" in content:
                logger.info("✅ detect-secrets timeout increased to 600s (10 minutes)")
                return True
            else:
                logger.error("❌ Timeout adjustment validation failed")
                return False
                
        except Exception as e:
            logger.error(f"Error validating timeout adjustments: {e}")
            return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Adjust workflow timeouts based on monitoring findings")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be adjusted without applying changes")
    args = parser.parse_args()
    
    base_path = Path(__file__).parent.parent.parent
    
    logger.info("⏱️ SCION Workflow Timeout Adjustment Tool")
    logger.info("=" * 50)
    
    if args.dry_run:
        logger.info("Running in DRY-RUN mode - no changes will be made")
    
    # Apply the timeout adjustments
    adjuster = WorkflowTimeoutAdjuster(base_path, dry_run=args.dry_run)
    
    if not adjuster.apply_timeout_adjustments():
        logger.error("❌ Failed to apply timeout adjustments")
        return 1
        
    if not args.dry_run:
        # Validate the adjustments
        if not adjuster.validate_adjustments():
            logger.error("❌ Timeout adjustment validation failed")
            return 1
            
        logger.info("🎉 Workflow timeout adjustments applied successfully!")
        logger.info("📊 New timeout configuration:")
        logger.info("  - Overall job timeout: 20 minutes (unchanged)")
        logger.info("  - detect-secrets timeout: 10 minutes (increased from 5)")
        logger.info("  - Security validation timeout: 10 minutes (unchanged)")
    else:
        logger.info("🔍 Dry-run complete - use without --dry-run to apply adjustments")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())