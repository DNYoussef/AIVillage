"""Importance Flow Utility Mathematics

Mathematical utilities for flow-based importance calculations in hypergraphs.
Supports PageRank, random walks, and uncertainty propagation.
"""

import logging

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm

logger = logging.getLogger(__name__)


class ImportanceFlow:
    """Utility mathematics for flow calculations in hypergraphs

    Features:
    - Sparse matrix PageRank computation
    - Uncertainty flow propagation
    - Hyperedge weight distribution
    - Random walk sampling
    """

    def __init__(self, damping: float = 0.85):
        self.damping = damping

    def compute_pagerank_sparse(
        self,
        adjacency_matrix: csr_matrix,
        personalization: np.ndarray | None = None,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> np.ndarray:
        """Compute PageRank using sparse matrix operations

        Args:
            adjacency_matrix: Sparse adjacency matrix (N x N)
            personalization: Personalization vector (N,) or None for uniform
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance

        Returns:
            PageRank scores (N,)
        """
        n = adjacency_matrix.shape[0]

        # Initialize personalization vector
        if personalization is None:
            personalization = np.ones(n) / n
        else:
            personalization = personalization / np.sum(personalization)

        # Normalize adjacency matrix (column-stochastic)
        outdegree = np.array(adjacency_matrix.sum(axis=1)).flatten()
        outdegree[outdegree == 0] = 1  # Handle dangling nodes

        # Create transition matrix
        transition_matrix = adjacency_matrix.multiply(1.0 / outdegree[:, np.newaxis])

        # Initialize PageRank vector
        pagerank = np.ones(n) / n

        # Power iteration
        for iteration in range(max_iterations):
            prev_pagerank = pagerank.copy()

            # PageRank update: r = d * M * r + (1-d) * p
            pagerank = (
                self.damping * transition_matrix.T.dot(pagerank)
                + (1 - self.damping) * personalization
            )

            # Check convergence
            diff = norm(pagerank - prev_pagerank, ord=1)
            if diff < tolerance:
                logger.debug(f"PageRank converged after {iteration + 1} iterations")
                break

        return pagerank

    def hyperedge_importance_distribution(
        self,
        hyperedge_participants: list[str],
        node_scores: dict[str, float],
        distribution_method: str = "proportional",
    ) -> dict[str, float]:
        """Distribute importance across hyperedge participants

        Args:
            hyperedge_participants: List of node IDs in the hyperedge
            node_scores: Current importance scores for nodes
            distribution_method: "proportional", "uniform", or "max_flow"

        Returns:
            Dict mapping participant_id to distributed importance
        """
        if not hyperedge_participants:
            return {}

        participant_scores = {
            node_id: node_scores.get(node_id, 0.0) for node_id in hyperedge_participants
        }

        if distribution_method == "uniform":
            # Equal distribution
            uniform_score = 1.0 / len(hyperedge_participants)
            return dict.fromkeys(hyperedge_participants, uniform_score)

        if distribution_method == "proportional":
            # Proportional to current node scores
            total_score = sum(participant_scores.values())
            if total_score == 0:
                return self.hyperedge_importance_distribution(
                    hyperedge_participants, node_scores, "uniform"
                )

            return {
                node_id: score / total_score
                for node_id, score in participant_scores.items()
            }

        if distribution_method == "max_flow":
            # Max-flow based distribution (simplified)
            max_score = max(participant_scores.values()) if participant_scores else 0.0
            if max_score == 0:
                return self.hyperedge_importance_distribution(
                    hyperedge_participants, node_scores, "uniform"
                )

            # Normalize by max score
            return {
                node_id: score / max_score
                for node_id, score in participant_scores.items()
            }

        raise ValueError(f"Unknown distribution method: {distribution_method}")

    def uncertainty_propagation(
        self,
        source_uncertainties: dict[str, float],
        edge_confidences: dict[tuple[str, str], float],
        propagation_steps: int = 3,
        decay_factor: float = 0.8,
    ) -> dict[str, float]:
        """Propagate uncertainty through the graph

        Args:
            source_uncertainties: Initial uncertainties {node_id: uncertainty}
            edge_confidences: Edge confidences {(source, target): confidence}
            propagation_steps: Number of propagation steps
            decay_factor: Uncertainty decay per step

        Returns:
            Updated uncertainties after propagation
        """
        current_uncertainties = source_uncertainties.copy()

        for step in range(propagation_steps):
            new_uncertainties = current_uncertainties.copy()

            for (source, target), confidence in edge_confidences.items():
                source_uncertainty = current_uncertainties.get(source, 0.0)
                target_uncertainty = current_uncertainties.get(target, 0.0)

                # Propagate uncertainty along edge
                # Higher confidence edges propagate less uncertainty
                propagated_uncertainty = (
                    source_uncertainty * (1.0 - confidence) * decay_factor
                )

                # Update target uncertainty (take maximum)
                new_uncertainties[target] = max(
                    new_uncertainties.get(target, 0.0), propagated_uncertainty
                )

            current_uncertainties = new_uncertainties
            decay_factor *= 0.9  # Decay the decay factor

        return current_uncertainties

    def random_walk_sampling(
        self,
        start_nodes: list[str],
        adjacency_dict: dict[str, list[tuple[str, float]]],
        walk_length: int = 10,
        num_walks: int = 100,
        restart_probability: float = 0.15,
    ) -> dict[str, int]:
        """Sample nodes using random walks for importance estimation

        Args:
            start_nodes: Starting nodes for walks
            adjacency_dict: {node_id: [(neighbor_id, weight), ...]}
            walk_length: Length of each walk
            num_walks: Number of random walks
            restart_probability: Probability of restarting walk

        Returns:
            Node visit counts {node_id: visit_count}
        """
        visit_counts = {}

        for walk_idx in range(num_walks):
            # Choose random start node
            current_node = np.random.choice(start_nodes)

            for step in range(walk_length):
                # Record visit
                visit_counts[current_node] = visit_counts.get(current_node, 0) + 1

                # Check for restart
                if np.random.random() < restart_probability:
                    current_node = np.random.choice(start_nodes)
                    continue

                # Get neighbors
                neighbors = adjacency_dict.get(current_node, [])
                if not neighbors:
                    # Dead end - restart
                    current_node = np.random.choice(start_nodes)
                    continue

                # Choose next node based on edge weights
                neighbor_ids, weights = zip(*neighbors, strict=False)
                weights = np.array(weights)
                weights = weights / np.sum(weights)  # Normalize

                next_node = np.random.choice(neighbor_ids, p=weights)
                current_node = next_node

        return visit_counts

    def compute_centrality_measures(
        self, adjacency_matrix: csr_matrix, node_ids: list[str]
    ) -> dict[str, dict[str, float]]:
        """Compute various centrality measures

        Args:
            adjacency_matrix: Sparse adjacency matrix
            node_ids: List of node IDs corresponding to matrix rows/cols

        Returns:
            {node_id: {centrality_type: score, ...}}
        """
        n = len(node_ids)
        centralities = {node_id: {} for node_id in node_ids}

        # PageRank centrality
        pagerank_scores = self.compute_pagerank_sparse(adjacency_matrix)
        for i, node_id in enumerate(node_ids):
            centralities[node_id]["pagerank"] = pagerank_scores[i]

        # Degree centrality
        degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()
        for i, node_id in enumerate(node_ids):
            centralities[node_id]["degree"] = degrees[i] / (n - 1) if n > 1 else 0.0

        # Closeness centrality (simplified - would need shortest paths)
        for i, node_id in enumerate(node_ids):
            # Simplified as inverse of average distance (using degree as proxy)
            centralities[node_id]["closeness"] = degrees[i] / n if n > 0 else 0.0

        # Betweenness centrality (simplified - computationally expensive)
        for i, node_id in enumerate(node_ids):
            # Simplified using degree and PageRank
            degree_norm = degrees[i] / np.max(degrees) if np.max(degrees) > 0 else 0.0
            pagerank_norm = (
                pagerank_scores[i] / np.max(pagerank_scores)
                if np.max(pagerank_scores) > 0
                else 0.0
            )
            centralities[node_id]["betweenness"] = (degree_norm + pagerank_norm) / 2

        return centralities

    def flow_based_ranking(
        self,
        source_nodes: list[str],
        target_nodes: list[str],
        flow_matrix: np.ndarray,
        node_ids: list[str],
    ) -> dict[str, float]:
        """Rank nodes based on flow from sources to targets

        Args:
            source_nodes: Source node IDs
            target_nodes: Target node IDs
            flow_matrix: Flow matrix (N x N)
            node_ids: Node IDs corresponding to matrix indices

        Returns:
            {node_id: flow_score}
        """
        node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}

        # Get source and target indices
        source_indices = [
            node_to_idx[node_id] for node_id in source_nodes if node_id in node_to_idx
        ]
        target_indices = [
            node_to_idx[node_id] for node_id in target_nodes if node_id in node_to_idx
        ]

        if not source_indices or not target_indices:
            return dict.fromkeys(node_ids, 0.0)

        # Compute flow scores
        flow_scores = {}

        for i, node_id in enumerate(node_ids):
            # Sum of flows from sources to this node
            inflow = np.sum(flow_matrix[source_indices, i])

            # Sum of flows from this node to targets
            outflow = np.sum(flow_matrix[i, target_indices])

            # Combined flow score
            flow_scores[node_id] = (inflow + outflow) / 2

        return flow_scores

    def compute_resistance_distance(
        self, adjacency_matrix: csr_matrix, source_idx: int, target_idx: int
    ) -> float:
        """Compute resistance distance between two nodes (simplified)

        Args:
            adjacency_matrix: Sparse adjacency matrix
            source_idx: Source node index
            target_idx: Target node index

        Returns:
            Resistance distance
        """
        try:
            # Simplified resistance distance using random walk
            # In practice, would use Laplacian pseudoinverse

            n = adjacency_matrix.shape[0]

            # Create Laplacian matrix
            degree = np.array(adjacency_matrix.sum(axis=1)).flatten()
            laplacian = np.diag(degree) - adjacency_matrix.toarray()

            # Add small regularization for numerical stability
            laplacian += 1e-6 * np.eye(n)

            # Compute effective resistance (simplified)
            # Would typically use pseudoinverse of Laplacian
            try:
                inv_laplacian = np.linalg.pinv(laplacian)
                resistance = (
                    inv_laplacian[source_idx, source_idx]
                    + inv_laplacian[target_idx, target_idx]
                    - 2 * inv_laplacian[source_idx, target_idx]
                )
                return max(0.0, resistance)
            except:
                # Fallback to simple distance measure
                return 1.0 / (1.0 + adjacency_matrix[source_idx, target_idx])

        except Exception as e:
            logger.warning(f"Resistance distance computation failed: {e!s}")
            return 1.0  # Default distance

    def normalize_scores(
        self, scores: dict[str, float], method: str = "minmax"
    ) -> dict[str, float]:
        """Normalize scores using various methods

        Args:
            scores: {node_id: score}
            method: "minmax", "zscore", "softmax"

        Returns:
            Normalized scores
        """
        if not scores:
            return {}

        values = np.array(list(scores.values()))

        if method == "minmax":
            min_val, max_val = np.min(values), np.max(values)
            if max_val > min_val:
                normalized_values = (values - min_val) / (max_val - min_val)
            else:
                normalized_values = np.ones_like(values)

        elif method == "zscore":
            mean_val, std_val = np.mean(values), np.std(values)
            if std_val > 0:
                normalized_values = (values - mean_val) / std_val
                # Shift to positive range
                normalized_values = normalized_values - np.min(normalized_values)
            else:
                normalized_values = np.ones_like(values)

        elif method == "softmax":
            # Apply softmax for probability distribution
            exp_values = np.exp(values - np.max(values))  # Numerical stability
            normalized_values = exp_values / np.sum(exp_values)

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return {
            node_id: normalized_values[i] for i, node_id in enumerate(scores.keys())
        }


# Utility functions


def build_sparse_adjacency(
    edges: list[tuple[str, str, float]], node_ids: list[str]
) -> csr_matrix:
    """Build sparse adjacency matrix from edge list"""
    node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
    n = len(node_ids)

    row_indices = []
    col_indices = []
    data = []

    for source, target, weight in edges:
        if source in node_to_idx and target in node_to_idx:
            row_indices.append(node_to_idx[source])
            col_indices.append(node_to_idx[target])
            data.append(weight)

    return csr_matrix((data, (row_indices, col_indices)), shape=(n, n))


def create_importance_flow(damping: float = 0.85) -> ImportanceFlow:
    """Create an ImportanceFlow instance"""
    return ImportanceFlow(damping=damping)
