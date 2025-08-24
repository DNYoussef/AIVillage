#!/usr/bin/env python3
"""
Phase 2.2 Cleanup: Remove remaining empty P2P directories and merge __init__.py files
"""

import logging
from pathlib import Path
import shutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Clean up remaining P2P directories and merge __init__.py files."""

    base_path = Path.cwd()
    libs_p2p = base_path / "libs" / "p2p"

    # Merge __init__.py files if needed
    remaining_init_files = [
        (libs_p2p / "betanet" / "__init__.py", base_path / "infrastructure" / "p2p" / "betanet" / "__init__.py"),
        (libs_p2p / "bitchat" / "__init__.py", base_path / "infrastructure" / "p2p" / "bitchat" / "__init__.py"),
        (libs_p2p / "__init__.py", base_path / "infrastructure" / "p2p" / "__init__.py"),
    ]

    for source, target in remaining_init_files:
        if source.exists():
            if not target.exists():
                logger.info(f"Moving {source} to {target}")
                shutil.move(str(source), str(target))
            else:
                logger.info(f"Target __init__.py exists, keeping existing: {target}")
                source.unlink()  # Remove the source since target exists

    # Remove empty directories
    empty_dirs = [libs_p2p / "betanet", libs_p2p / "bitchat"]

    for dir_path in empty_dirs:
        if dir_path.exists() and not any(dir_path.iterdir()):
            logger.info(f"Removing empty directory: {dir_path}")
            dir_path.rmdir()

    # Check if libs/p2p is now empty and can be removed
    if libs_p2p.exists():
        remaining_items = list(libs_p2p.iterdir())
        if not remaining_items:
            logger.info("Removing empty libs/p2p directory")
            libs_p2p.rmdir()
        else:
            logger.info(f"libs/p2p still contains: {[item.name for item in remaining_items]}")

    logger.info("P2P cleanup completed")


if __name__ == "__main__":
    main()
