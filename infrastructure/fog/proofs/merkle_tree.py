"""
Merkle Tree Implementation for Proof System

Provides efficient Merkle tree construction and verification
for cryptographic proof aggregation and batch verification.
"""

from dataclasses import dataclass
import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MerkleProof:
    """Merkle inclusion proof structure"""

    leaf_hash: str
    root_hash: str
    leaf_index: int
    proof_path: list[tuple[str, str]]  # (hash, position) - "left" or "right"
    tree_size: int

    def verify(self) -> bool:
        """Verify this Merkle proof"""
        try:
            current_hash = self.leaf_hash

            for sibling_hash, position in self.proof_path:
                if position == "left":
                    combined = sibling_hash + current_hash
                else:  # position == "right"
                    combined = current_hash + sibling_hash

                current_hash = hashlib.sha256(combined.encode()).hexdigest()

            return current_hash == self.root_hash

        except Exception as e:
            logger.error(f"Merkle proof verification error: {e}")
            return False


class MerkleTree:
    """Merkle tree implementation for cryptographic proof aggregation"""

    def __init__(self, data_hashes: list[str]):
        """Initialize Merkle tree from list of data hashes"""
        if not data_hashes:
            raise ValueError("Cannot create Merkle tree from empty data")

        self.original_hashes = data_hashes.copy()
        self.tree_levels = []
        self.depth = 0

        self._build_tree()

        logger.debug(f"Built Merkle tree with {len(data_hashes)} leaves, depth {self.depth}")

    def _build_tree(self):
        """Build the Merkle tree bottom-up"""
        current_level = self.original_hashes.copy()
        self.tree_levels.append(current_level)

        while len(current_level) > 1:
            next_level = []

            # Process pairs of hashes
            for i in range(0, len(current_level), 2):
                left_hash = current_level[i]

                # Handle odd number of hashes by duplicating the last one
                if i + 1 < len(current_level):
                    right_hash = current_level[i + 1]
                else:
                    right_hash = left_hash

                # Create parent hash
                combined = left_hash + right_hash
                parent_hash = hashlib.sha256(combined.encode()).hexdigest()
                next_level.append(parent_hash)

            current_level = next_level
            self.tree_levels.append(current_level)
            self.depth += 1

        self.depth = len(self.tree_levels) - 1

    def get_root(self) -> str:
        """Get the Merkle root hash"""
        if not self.tree_levels or not self.tree_levels[-1]:
            raise ValueError("Tree not properly constructed")
        return self.tree_levels[-1][0]

    def get_proof(self, leaf_index: int) -> MerkleProof | None:
        """Generate Merkle inclusion proof for a specific leaf"""
        if leaf_index < 0 or leaf_index >= len(self.original_hashes):
            return None

        try:
            leaf_hash = self.original_hashes[leaf_index]
            proof_path = []
            current_index = leaf_index

            # Build proof path from bottom to top
            for level in range(len(self.tree_levels) - 1):
                level_data = self.tree_levels[level]

                # Find sibling
                if current_index % 2 == 0:  # Left child
                    sibling_index = current_index + 1
                    position = "right"
                else:  # Right child
                    sibling_index = current_index - 1
                    position = "left"

                # Get sibling hash
                if sibling_index < len(level_data):
                    sibling_hash = level_data[sibling_index]
                else:
                    # Odd number of nodes, sibling is same as current
                    sibling_hash = level_data[current_index]

                proof_path.append((sibling_hash, position))

                # Move to parent index
                current_index = current_index // 2

            return MerkleProof(
                leaf_hash=leaf_hash,
                root_hash=self.get_root(),
                leaf_index=leaf_index,
                proof_path=proof_path,
                tree_size=len(self.original_hashes),
            )

        except Exception as e:
            logger.error(f"Error generating Merkle proof: {e}")
            return None

    def get_all_proofs(self) -> list[tuple[str, str]]:
        """Get simplified proof paths for all leaves (for batch verification)"""
        proofs = []

        for i in range(len(self.original_hashes)):
            merkle_proof = self.get_proof(i)
            if merkle_proof:
                proofs.append(merkle_proof.proof_path)
            else:
                proofs.append([])

        return proofs

    def verify_leaf(self, leaf_hash: str, leaf_index: int, proof_path: list[tuple[str, str]]) -> bool:
        """Verify a leaf using provided proof path"""
        if leaf_index < 0 or leaf_index >= len(self.original_hashes):
            return False

        try:
            current_hash = leaf_hash

            for sibling_hash, position in proof_path:
                if position == "left":
                    combined = sibling_hash + current_hash
                else:
                    combined = current_hash + sibling_hash

                current_hash = hashlib.sha256(combined.encode()).hexdigest()

            return current_hash == self.get_root()

        except Exception as e:
            logger.error(f"Leaf verification error: {e}")
            return False

    @classmethod
    def verify_proof_static(cls, merkle_proof: MerkleProof) -> bool:
        """Static method to verify a Merkle proof without tree instance"""
        return merkle_proof.verify()

    def get_tree_info(self) -> dict[str, Any]:
        """Get information about the tree structure"""
        return {
            "total_leaves": len(self.original_hashes),
            "tree_depth": self.depth,
            "root_hash": self.get_root(),
            "levels": len(self.tree_levels),
            "nodes_per_level": [len(level) for level in self.tree_levels],
        }

    def get_leaf_hashes(self) -> list[str]:
        """Get all leaf hashes"""
        return self.original_hashes.copy()

    def find_leaf_index(self, leaf_hash: str) -> int | None:
        """Find the index of a specific leaf hash"""
        try:
            return self.original_hashes.index(leaf_hash)
        except ValueError:
            return None

    def update_leaf(self, leaf_index: int, new_hash: str) -> bool:
        """Update a leaf hash and rebuild the tree"""
        if leaf_index < 0 or leaf_index >= len(self.original_hashes):
            return False

        try:
            self.original_hashes[leaf_index] = new_hash
            self.tree_levels.clear()
            self._build_tree()

            logger.debug(f"Updated leaf {leaf_index} and rebuilt tree")
            return True

        except Exception as e:
            logger.error(f"Error updating leaf: {e}")
            return False

    def add_leaf(self, leaf_hash: str) -> int:
        """Add a new leaf and rebuild the tree"""
        try:
            self.original_hashes.append(leaf_hash)
            self.tree_levels.clear()
            self._build_tree()

            new_index = len(self.original_hashes) - 1
            logger.debug(f"Added new leaf at index {new_index}")
            return new_index

        except Exception as e:
            logger.error(f"Error adding leaf: {e}")
            return -1

    @staticmethod
    def compute_hash(data: str) -> str:
        """Utility method to compute SHA-256 hash"""
        return hashlib.sha256(data.encode()).hexdigest()

    @classmethod
    def from_data_list(cls, data_list: list[str]) -> "MerkleTree":
        """Create Merkle tree from list of data strings"""
        hashes = [cls.compute_hash(data) for data in data_list]
        return cls(hashes)

    def __str__(self) -> str:
        return f"MerkleTree(leaves={len(self.original_hashes)}, depth={self.depth}, root={self.get_root()[:16]}...)"

    def __repr__(self) -> str:
        return self.__str__()
