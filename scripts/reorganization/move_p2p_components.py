#!/usr/bin/env python3
"""
Phase 2.2: Complete P2P infrastructure reorganization
Move P2P components from libs/p2p/* to their appropriate clean architecture locations.
"""

import logging
import shutil
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("p2p_reorganization.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def safe_move(source: Path, target: Path) -> bool:
    """
    Safely move source to target, creating directories as needed.
    Returns True if successful, False if target already exists.
    """
    try:
        if not source.exists():
            logger.warning(f"Source does not exist: {source}")
            return False

        if target.exists():
            logger.warning(f"Target already exists, skipping: {target}")
            return False

        # Create parent directories
        target.parent.mkdir(parents=True, exist_ok=True)

        # Move the file or directory
        shutil.move(str(source), str(target))
        logger.info(f"Moved: {source} → {target}")
        return True

    except Exception as e:
        logger.error(f"Failed to move {source} to {target}: {e}")
        return False


def merge_directory_contents(source_dir: Path, target_dir: Path) -> bool:
    """
    Merge contents of source_dir into target_dir, moving individual files
    to avoid overwriting existing directories.
    """
    if not source_dir.exists() or not source_dir.is_dir():
        logger.warning(f"Source directory does not exist: {source_dir}")
        return False

    target_dir.mkdir(parents=True, exist_ok=True)
    moved_count = 0

    for item in source_dir.iterdir():
        target_item = target_dir / item.name

        if item.is_file():
            if not target_item.exists():
                shutil.move(str(item), str(target_item))
                logger.info(f"Merged file: {item} → {target_item}")
                moved_count += 1
            else:
                logger.warning(f"File already exists, skipping: {target_item}")
        elif item.is_dir():
            # Recursively merge directory contents
            if merge_directory_contents(item, target_item):
                moved_count += 1

    # Remove source directory if it's now empty
    if moved_count > 0 and not any(source_dir.iterdir()):
        source_dir.rmdir()
        logger.info(f"Removed empty directory: {source_dir}")

    return moved_count > 0


def main():
    """Execute Phase 2.2: P2P infrastructure reorganization."""

    base_path = Path.cwd()
    libs_p2p = base_path / "libs" / "p2p"

    if not libs_p2p.exists():
        logger.error(f"Source directory does not exist: {libs_p2p}")
        return False

    logger.info("Starting Phase 2.2: P2P infrastructure reorganization")

    # Define move operations
    moves = [
        # 1. Move libs/p2p/betanet/* → infrastructure/p2p/betanet/ (merge with existing)
        {
            "source": libs_p2p / "betanet",
            "target": base_path / "infrastructure" / "p2p" / "betanet",
            "operation": "merge",
        },
        # 2. Move libs/p2p/bitchat/* → infrastructure/p2p/bitchat/ (merge with existing)
        {
            "source": libs_p2p / "bitchat",
            "target": base_path / "infrastructure" / "p2p" / "bitchat",
            "operation": "merge",
        },
        # 3. Move libs/p2p/communications/* → infrastructure/p2p/communications/
        {
            "source": libs_p2p / "communications",
            "target": base_path / "infrastructure" / "p2p" / "communications",
            "operation": "move",
        },
        # 4. Move libs/p2p/betanet-bounty/* → integrations/bounties/betanet/ (external bounty)
        {
            "source": libs_p2p / "betanet-bounty",
            "target": base_path / "integrations" / "bounties" / "betanet",
            "operation": "move",
        },
        # 5. Move libs/p2p/bridges/* → integrations/bridges/p2p/
        {"source": libs_p2p / "bridges", "target": base_path / "integrations" / "bridges" / "p2p", "operation": "move"},
        # 6. Move libs/p2p/core/* → infrastructure/p2p/core/
        {"source": libs_p2p / "core", "target": base_path / "infrastructure" / "p2p" / "core", "operation": "move"},
        # 7. Move other remaining P2P files appropriately
        {
            "source": libs_p2p / "bounty-tmp",
            "target": base_path / "integrations" / "bounties" / "tmp",
            "operation": "move",
        },
        {
            "source": libs_p2p / "mobile_integration",
            "target": base_path / "infrastructure" / "p2p" / "mobile_integration",
            "operation": "move",
        },
        {
            "source": libs_p2p / "legacy_src",
            "target": base_path / "infrastructure" / "p2p" / "legacy",
            "operation": "move",
        },
        # Move individual files
        {
            "source": libs_p2p / "scion_gateway.py",
            "target": base_path / "infrastructure" / "p2p" / "scion_gateway.py",
            "operation": "move",
        },
        {
            "source": libs_p2p / "__init__.py",
            "target": base_path / "infrastructure" / "p2p" / "__init__.py",
            "operation": "merge_content",
        },
    ]

    success_count = 0
    total_count = len(moves)

    for move_op in moves:
        source = move_op["source"]
        target = move_op["target"]
        operation = move_op["operation"]

        logger.info(f"Processing: {source} → {target} ({operation})")

        if operation == "move":
            if safe_move(source, target):
                success_count += 1
        elif operation == "merge":
            if merge_directory_contents(source, target):
                success_count += 1
        elif operation == "merge_content":
            # Special case for __init__.py files - merge content
            if source.exists() and source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    safe_move(source, target)
                    success_count += 1
                else:
                    logger.info(f"Target __init__.py exists, manual merge may be needed: {target}")

    # Clean up empty directories
    if libs_p2p.exists() and not any(libs_p2p.iterdir()):
        libs_p2p.rmdir()
        logger.info(f"Removed empty directory: {libs_p2p}")

    logger.info(f"Phase 2.2 completed: {success_count}/{total_count} operations successful")

    # Verify new structure
    logger.info("Verifying new P2P structure:")
    p2p_infrastructure = base_path / "infrastructure" / "p2p"
    if p2p_infrastructure.exists():
        for item in sorted(p2p_infrastructure.rglob("*")):
            if item.is_file() and item.suffix == ".py":
                logger.info(f"  Found: {item.relative_to(base_path)}")

    bounties_integration = base_path / "integrations" / "bounties"
    if bounties_integration.exists():
        logger.info("Bounty integrations structure:")
        for item in sorted(bounties_integration.iterdir()):
            logger.info(f"  Found: {item.relative_to(base_path)}")

    bridges_integration = base_path / "integrations" / "bridges"
    if bridges_integration.exists():
        logger.info("Bridge integrations structure:")
        for item in sorted(bridges_integration.iterdir()):
            logger.info(f"  Found: {item.relative_to(base_path)}")

    return success_count == total_count


if __name__ == "__main__":
    success = main()
    if success:
        print("✅ Phase 2.2: P2P infrastructure reorganization completed successfully!")
    else:
        print("❌ Phase 2.2: P2P infrastructure reorganization completed with some issues. Check logs.")
