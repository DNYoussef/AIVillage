"""
Merkle Tree Aggregator for Proof System

Provides efficient Merkle tree construction and proof generation
for batch verification of fog computing proofs.
"""

from dataclasses import dataclass
import hashlib
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class MerkleNode:
    """Merkle tree node representation"""

    hash: str
    left: Optional["MerkleNode"] = None
    right: Optional["MerkleNode"] = None
    data: str | None = None  # Leaf nodes store original data
    level: int = 0


@dataclass
class MerkleProof:
    """Merkle inclusion proof for verifying individual items in tree"""

    leaf_hash: str
    root_hash: str
    proof_path: list[tuple[str, str]]  # (hash, position) pairs - "left" or "right"
    leaf_index: int
    tree_size: int

    def is_valid(self) -> bool:
        """Verify the Merkle proof is valid"""
        try:
            current_hash = self.leaf_hash

            for proof_hash, position in self.proof_path:
                if position == "left":
                    combined = proof_hash + current_hash
                else:  # position == "right"
                    combined = current_hash + proof_hash
                current_hash = hashlib.sha256(combined.encode()).hexdigest()

            return current_hash == self.root_hash

        except Exception:
            return False


class MerkleAggregator:
    """
    Merkle Tree Aggregator for cryptographic proof aggregation

    Features:
    - Efficient Merkle tree construction from proof lists
    - Merkle inclusion proof generation for individual proofs
    - Batch verification capabilities
    - Support for different proof types in single tree
    - Incremental tree updates for streaming proofs
    """

    def __init__(self):
        self.trees: dict[str, MerkleNode] = {}
        self.proof_mappings: dict[str, dict[str, Any]] = {}

        # Statistics
        self.stats = {"trees_built": 0, "proofs_generated": 0, "verifications_performed": 0, "total_leaf_nodes": 0}

        logger.info("Merkle aggregator initialized")

    async def build_merkle_tree(self, proof_data: list[dict[str, Any]], tree_id: str) -> MerkleNode:
        """
        Build Merkle tree from list of proof data

        Args:
            proof_data: List of proof dictionaries to aggregate
            tree_id: Unique identifier for this tree

        Returns:
            Root node of constructed Merkle tree
        """
        if not proof_data:
            raise ValueError("Cannot build tree from empty proof data")

        try:
            # Convert proof data to leaf hashes
            leaf_hashes = []
            leaf_data = []

            for i, proof in enumerate(proof_data):
                # Create canonical representation for hashing
                proof_json = json.dumps(proof, sort_keys=True)
                leaf_hash = hashlib.sha256(proof_json.encode()).hexdigest()
                leaf_hashes.append(leaf_hash)
                leaf_data.append(proof_json)

            # Create leaf nodes
            leaf_nodes = [
                MerkleNode(hash=hash_val, data=data, level=0) for hash_val, data in zip(leaf_hashes, leaf_data)
            ]

            # Build tree bottom-up
            root = await self._build_tree_recursive(leaf_nodes, 1)

            # Store tree and create mapping
            self.trees[tree_id] = root
            self.proof_mappings[tree_id] = {
                "leaf_hashes": leaf_hashes,
                "leaf_data": leaf_data,
                "tree_size": len(proof_data),
            }

            # Update statistics
            self.stats["trees_built"] += 1
            self.stats["total_leaf_nodes"] += len(proof_data)

            logger.info(f"Built Merkle tree {tree_id} with {len(proof_data)} leaves")
            return root

        except Exception as e:
            logger.error(f"Failed to build Merkle tree: {e}")
            raise

    async def _build_tree_recursive(self, nodes: list[MerkleNode], level: int) -> MerkleNode:
        """
        Recursively build Merkle tree from nodes

        Args:
            nodes: List of nodes at current level
            level: Current tree level

        Returns:
            Root node of tree/subtree
        """
        if len(nodes) == 1:
            return nodes[0]

        next_level_nodes = []

        # Process pairs of nodes
        for i in range(0, len(nodes), 2):
            left = nodes[i]
            # Handle odd number of nodes by duplicating last node
            right = nodes[i + 1] if i + 1 < len(nodes) else nodes[i]

            # Create parent hash
            combined = left.hash + right.hash
            parent_hash = hashlib.sha256(combined.encode()).hexdigest()

            # Create parent node
            parent = MerkleNode(hash=parent_hash, left=left, right=right, level=level)

            next_level_nodes.append(parent)

        return await self._build_tree_recursive(next_level_nodes, level + 1)

    async def generate_inclusion_proof(self, tree_id: str, leaf_index: int) -> MerkleProof | None:
        """
        Generate Merkle inclusion proof for specific leaf

        Args:
            tree_id: Tree identifier
            leaf_index: Index of leaf to prove (0-based)

        Returns:
            MerkleProof object or None if invalid
        """
        if tree_id not in self.trees or tree_id not in self.proof_mappings:
            logger.error(f"Tree {tree_id} not found")
            return None

        mapping = self.proof_mappings[tree_id]

        if leaf_index >= mapping["tree_size"] or leaf_index < 0:
            logger.error(f"Invalid leaf index {leaf_index} for tree size {mapping['tree_size']}")
            return None

        try:
            root = self.trees[tree_id]
            leaf_hash = mapping["leaf_hashes"][leaf_index]

            # Generate proof path
            proof_path = await self._generate_proof_path(root, leaf_index, mapping["tree_size"])

            merkle_proof = MerkleProof(
                leaf_hash=leaf_hash,
                root_hash=root.hash,
                proof_path=proof_path,
                leaf_index=leaf_index,
                tree_size=mapping["tree_size"],
            )

            # Update statistics
            self.stats["proofs_generated"] += 1

            logger.debug(f"Generated inclusion proof for leaf {leaf_index} in tree {tree_id}")
            return merkle_proof

        except Exception as e:
            logger.error(f"Failed to generate inclusion proof: {e}")
            return None

    async def _generate_proof_path(self, node: MerkleNode, target_index: int, tree_size: int) -> list[tuple[str, str]]:
        """
        Generate proof path from root to target leaf

        Args:
            node: Current node in traversal
            target_index: Index of target leaf
            tree_size: Total tree size

        Returns:
            List of (sibling_hash, position) tuples
        """
        proof_path = []
        current_index = target_index
        current_size = tree_size
        current_node = node

        while current_node.left is not None:  # Not a leaf
            # Calculate split point for this level
            left_subtree_size = self._get_left_subtree_size(current_size)

            if current_index < left_subtree_size:
                # Target is in left subtree
                if current_node.right:
                    proof_path.append((current_node.right.hash, "right"))
                current_node = current_node.left
                current_size = left_subtree_size
            else:
                # Target is in right subtree
                if current_node.left:
                    proof_path.append((current_node.left.hash, "left"))
                current_node = current_node.right
                current_index -= left_subtree_size
                current_size = current_size - left_subtree_size

        return proof_path

    def _get_left_subtree_size(self, total_size: int) -> int:
        """Calculate size of left subtree for given total size"""
        if total_size <= 1:
            return total_size

        # Find largest power of 2 <= total_size
        power = 1
        while power * 2 <= total_size:
            power *= 2

        # If remaining nodes fit in left subtree, put them there
        remaining = total_size - power
        left_subtree_size = power // 2

        if remaining <= left_subtree_size:
            return power // 2 + remaining
        else:
            return power

    async def verify_inclusion_proof(self, proof: MerkleProof) -> bool:
        """
        Verify Merkle inclusion proof

        Args:
            proof: MerkleProof to verify

        Returns:
            True if proof is valid
        """
        try:
            is_valid = proof.is_valid()

            # Update statistics
            self.stats["verifications_performed"] += 1

            if is_valid:
                logger.debug("Merkle proof verification successful")
            else:
                logger.warning("Merkle proof verification failed")

            return is_valid

        except Exception as e:
            logger.error(f"Proof verification error: {e}")
            return False

    async def aggregate_multiple_trees(self, tree_ids: list[str], aggregated_tree_id: str) -> MerkleNode | None:
        """
        Aggregate multiple existing trees into a single super-tree

        Args:
            tree_ids: List of tree IDs to aggregate
            aggregated_tree_id: ID for the new aggregated tree

        Returns:
            Root of aggregated tree or None if failed
        """
        if not tree_ids:
            return None

        # Collect root hashes from existing trees
        root_nodes = []
        total_leaves = 0

        for tree_id in tree_ids:
            if tree_id in self.trees:
                root = self.trees[tree_id]
                root_nodes.append(MerkleNode(hash=root.hash, level=0))
                total_leaves += self.proof_mappings[tree_id]["tree_size"]
            else:
                logger.warning(f"Tree {tree_id} not found for aggregation")

        if not root_nodes:
            return None

        try:
            # Build super-tree from root hashes
            aggregated_root = await self._build_tree_recursive(root_nodes, 1)

            # Store aggregated tree
            self.trees[aggregated_tree_id] = aggregated_root
            self.proof_mappings[aggregated_tree_id] = {
                "leaf_hashes": [node.hash for node in root_nodes],
                "leaf_data": [f"tree_root_{i}" for i in range(len(root_nodes))],
                "tree_size": len(root_nodes),
                "aggregated_from": tree_ids,
                "total_original_leaves": total_leaves,
            }

            logger.info(
                f"Aggregated {len(tree_ids)} trees into {aggregated_tree_id} " f"({total_leaves} total original leaves)"
            )

            return aggregated_root

        except Exception as e:
            logger.error(f"Failed to aggregate trees: {e}")
            return None

    async def get_tree_info(self, tree_id: str) -> dict[str, Any] | None:
        """Get information about a specific tree"""
        if tree_id not in self.trees:
            return None

        mapping = self.proof_mappings[tree_id]
        root = self.trees[tree_id]

        return {
            "tree_id": tree_id,
            "root_hash": root.hash,
            "tree_size": mapping["tree_size"],
            "tree_depth": await self._calculate_tree_depth(root),
            "is_aggregated": "aggregated_from" in mapping,
            "total_original_leaves": mapping.get("total_original_leaves", mapping["tree_size"]),
        }

    async def _calculate_tree_depth(self, node: MerkleNode) -> int:
        """Calculate depth of tree from given node"""
        if not node.left and not node.right:
            return 1

        left_depth = 0
        right_depth = 0

        if node.left:
            left_depth = await self._calculate_tree_depth(node.left)
        if node.right:
            right_depth = await self._calculate_tree_depth(node.right)

        return 1 + max(left_depth, right_depth)

    def get_statistics(self) -> dict[str, Any]:
        """Get aggregator statistics"""
        return {**self.stats, "active_trees": len(self.trees), "total_mappings": len(self.proof_mappings)}

    async def cleanup_tree(self, tree_id: str) -> bool:
        """Remove tree and associated data"""
        try:
            if tree_id in self.trees:
                del self.trees[tree_id]
            if tree_id in self.proof_mappings:
                del self.proof_mappings[tree_id]

            logger.info(f"Cleaned up tree {tree_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cleanup tree {tree_id}: {e}")
            return False
