"""
HippoRAG - Neural-Biological Memory System

Implements hippocampus-inspired episodic memory with biological memory consolidation,
replay mechanisms, and context-dependent retrieval based on neuroscientific principles.

Key Features:
- Episodic vs semantic memory separation
- Memory consolidation and strengthening
- Context-dependent retrieval with attention mechanisms
- Memory replay for pattern reinforcement
- Temporal decay and forgetting curves
- Spatial-temporal binding for contextual memory
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import math
import time
from typing import Any
import uuid

import numpy as np
from scipy.special import softmax

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory in hippocampus-inspired system."""

    EPISODIC = "episodic"  # Event-based, contextual memories
    SEMANTIC = "semantic"  # Fact-based, abstracted knowledge
    PROCEDURAL = "procedural"  # How-to knowledge
    WORKING = "working"  # Active processing memory


class ConsolidationState(Enum):
    """Memory consolidation states."""

    FRESH = "fresh"  # Recently encoded, labile
    CONSOLIDATING = "consolidating"  # Undergoing consolidation
    CONSOLIDATED = "consolidated"  # Stable, long-term
    FORGOTTEN = "forgotten"  # Decayed below threshold


@dataclass
class MemoryTrace:
    """Individual memory trace with biological properties."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    embedding: np.ndarray = field(default_factory=lambda: np.array([]))

    # Biological properties
    memory_type: MemoryType = MemoryType.EPISODIC
    consolidation_state: ConsolidationState = ConsolidationState.FRESH
    strength: float = 1.0  # Memory strength (0-1)
    accessibility: float = 1.0  # Current accessibility (0-1)

    # Temporal properties
    encoded_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0

    # Context binding
    spatial_context: dict[str, Any] = field(default_factory=dict)
    temporal_context: dict[str, Any] = field(default_factory=dict)
    semantic_context: dict[str, Any] = field(default_factory=dict)

    # Associations
    associated_memories: list[str] = field(default_factory=list)
    consolidation_links: list[str] = field(default_factory=list)

    # Quality metrics
    confidence: float = 1.0
    relevance_history: list[float] = field(default_factory=list)

    # Metadata
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def calculate_decay(self, current_time: datetime) -> float:
        """Calculate memory decay based on forgetting curve."""
        time_diff = (current_time - self.last_accessed).total_seconds()
        days_elapsed = time_diff / (24 * 3600)

        # Ebbinghaus forgetting curve with consolidation factor
        consolidation_factor = {
            ConsolidationState.FRESH: 1.0,
            ConsolidationState.CONSOLIDATING: 0.7,
            ConsolidationState.CONSOLIDATED: 0.3,
            ConsolidationState.FORGOTTEN: 2.0,
        }[self.consolidation_state]

        # Strength-dependent decay
        decay_rate = 0.1 * consolidation_factor / (self.strength + 0.1)
        decay_factor = math.exp(-decay_rate * days_elapsed)

        return min(1.0, decay_factor)

    def update_accessibility(self, current_time: datetime):
        """Update accessibility based on decay and consolidation."""
        decay_factor = self.calculate_decay(current_time)

        # Consolidation boosts accessibility
        consolidation_boost = {
            ConsolidationState.FRESH: 0.0,
            ConsolidationState.CONSOLIDATING: 0.1,
            ConsolidationState.CONSOLIDATED: 0.2,
            ConsolidationState.FORGOTTEN: -0.5,
        }[self.consolidation_state]

        self.accessibility = max(0.0, min(1.0, decay_factor * self.strength + consolidation_boost))

    def reinforce(self, strength_boost: float = 0.1):
        """Reinforce memory through retrieval or relevance."""
        self.strength = min(1.0, self.strength + strength_boost)
        self.access_count += 1
        self.last_accessed = datetime.now()

        # Move toward consolidation with repeated access
        if self.access_count > 5 and self.consolidation_state == ConsolidationState.FRESH:
            self.consolidation_state = ConsolidationState.CONSOLIDATING
        elif self.access_count > 20 and self.consolidation_state == ConsolidationState.CONSOLIDATING:
            self.consolidation_state = ConsolidationState.CONSOLIDATED


@dataclass
class RetrievalResult:
    """Result from hippocampus memory retrieval."""

    memory_traces: list[MemoryTrace] = field(default_factory=list)
    retrieval_confidence: float = 0.0
    context_match_scores: list[float] = field(default_factory=list)
    total_memories_searched: int = 0
    retrieval_time_ms: float = 0.0

    # Neural activation patterns
    activation_pattern: np.ndarray = field(default_factory=lambda: np.array([]))
    attention_weights: np.ndarray = field(default_factory=lambda: np.array([]))

    # Consolidation insights
    consolidation_opportunities: list[dict[str, Any]] = field(default_factory=list)


class HippoRAG:
    """
    Hippocampus-Inspired Neural Memory System

    Implements neurobiological memory principles including:
    - Episodic memory formation and retrieval
    - Memory consolidation processes
    - Context-dependent recall mechanisms
    - Attention-based retrieval
    - Memory replay and pattern completion
    - Forgetting curves and decay
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        max_episodic_memories: int = 10000,
        consolidation_threshold: float = 0.7,
        forgetting_threshold: float = 0.1,
        replay_frequency: int = 100,  # Every N retrievals
    ):
        self.embedding_dim = embedding_dim
        self.max_episodic_memories = max_episodic_memories
        self.consolidation_threshold = consolidation_threshold
        self.forgetting_threshold = forgetting_threshold
        self.replay_frequency = replay_frequency

        # Memory stores
        self.episodic_memories: dict[str, MemoryTrace] = {}
        self.semantic_memories: dict[str, MemoryTrace] = {}
        self.working_memory: list[str] = []  # IDs of currently active memories

        # Neural patterns
        self.memory_embeddings: np.ndarray = np.empty((0, embedding_dim))
        self.memory_id_mapping: list[str] = []

        # Context networks
        self.spatial_network: dict[str, list[str]] = {}
        self.temporal_network: dict[str, list[str]] = {}
        self.semantic_network: dict[str, list[str]] = {}

        # Attention and consolidation
        self.attention_mechanism = AttentionMechanism(embedding_dim)
        self.consolidator = MemoryConsolidator(consolidation_threshold)

        # Statistics and performance tracking
        self.stats = {
            "memories_encoded": 0,
            "retrievals_performed": 0,
            "consolidations_performed": 0,
            "forgotten_memories": 0,
            "replay_sessions": 0,
            "average_retrieval_time": 0.0,
        }

        self.retrieval_count = 0
        self.initialized = False

    async def initialize(self):
        """Initialize the hippocampus memory system."""
        logger.info("Initializing HippoRAG neural memory system...")

        # Initialize attention mechanism
        await self.attention_mechanism.initialize()

        # Initialize consolidator
        await self.consolidator.initialize()

        # Set up background consolidation task
        asyncio.create_task(self._background_consolidation())

        self.initialized = True
        logger.info("🧠 HippoRAG neural memory system ready")

    async def encode_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        spatial_context: dict[str, Any] | None = None,
        temporal_context: dict[str, Any] | None = None,
        semantic_context: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Encode new memory with contextual binding."""
        try:
            # Create memory trace
            memory_trace = MemoryTrace(
                content=content,
                memory_type=memory_type,
                spatial_context=spatial_context or {},
                temporal_context=temporal_context or {},
                semantic_context=semantic_context or {},
                metadata=metadata or {},
            )

            # Generate neural embedding
            memory_trace.embedding = await self._create_contextual_embedding(
                content, spatial_context, temporal_context, semantic_context
            )

            # Store in appropriate memory system
            if memory_type == MemoryType.EPISODIC:
                # Check capacity limit for episodic memories
                if len(self.episodic_memories) >= self.max_episodic_memories:
                    await self._forget_oldest_memories()

                self.episodic_memories[memory_trace.id] = memory_trace
            else:
                self.semantic_memories[memory_trace.id] = memory_trace

            # Update neural networks
            await self._update_context_networks(memory_trace)
            await self._add_to_neural_index(memory_trace)

            self.stats["memories_encoded"] += 1
            logger.debug(f"Encoded {memory_type.value} memory: {memory_trace.id}")

            return memory_trace.id

        except Exception as e:
            logger.error(f"Failed to encode memory: {e}")
            return ""

    async def retrieve_memories(
        self,
        query: str,
        k: int = 10,
        memory_types: list[MemoryType] | None = None,
        spatial_context: dict[str, Any] | None = None,
        temporal_context: dict[str, Any] | None = None,
        semantic_context: dict[str, Any] | None = None,
        attention_focus: str | None = None,
    ) -> RetrievalResult:
        """Retrieve memories using context-dependent recall."""
        start_time = time.time()

        try:
            # Create query embedding with context
            query_embedding = await self._create_contextual_embedding(
                query, spatial_context, temporal_context, semantic_context
            )

            # Determine which memory stores to search
            search_stores = []
            if not memory_types:
                memory_types = [MemoryType.EPISODIC, MemoryType.SEMANTIC]

            for mem_type in memory_types:
                if mem_type == MemoryType.EPISODIC:
                    search_stores.extend(self.episodic_memories.values())
                elif mem_type == MemoryType.SEMANTIC:
                    search_stores.extend(self.semantic_memories.values())

            # Update accessibility for all candidate memories
            current_time = datetime.now()
            for memory in search_stores:
                memory.update_accessibility(current_time)

            # Filter out forgotten memories
            accessible_memories = [m for m in search_stores if m.accessibility > self.forgetting_threshold]

            # Calculate similarity scores
            candidates = []
            for memory in accessible_memories:
                # Neural similarity
                neural_sim = await self._calculate_neural_similarity(query_embedding, memory.embedding)

                # Context similarity
                context_sim = self._calculate_context_similarity(
                    memory, spatial_context, temporal_context, semantic_context
                )

                # Combine scores with attention mechanism
                combined_score = await self.attention_mechanism.compute_attention(
                    neural_sim, context_sim, memory.accessibility, memory.strength, attention_focus
                )

                candidates.append((memory, combined_score, context_sim))

            # Rank by combined score
            candidates.sort(key=lambda x: x[1], reverse=True)

            # Get top k results
            top_candidates = candidates[:k]
            memory_traces = [c[0] for c in top_candidates]
            context_scores = [c[2] for c in top_candidates]

            # Reinforce retrieved memories
            for memory, score, _ in top_candidates:
                memory.reinforce(strength_boost=score * 0.1)

            # Create activation pattern
            activation_pattern = self._create_activation_pattern(top_candidates)
            attention_weights = self.attention_mechanism.get_last_attention_weights()

            # Check for consolidation opportunities
            consolidation_ops = await self._identify_consolidation_opportunities(memory_traces)

            # Update statistics
            retrieval_time = (time.time() - start_time) * 1000
            self.stats["retrievals_performed"] += 1
            self._update_average_retrieval_time(retrieval_time)

            # Trigger replay if needed
            self.retrieval_count += 1
            if self.retrieval_count % self.replay_frequency == 0:
                asyncio.create_task(self._memory_replay(memory_traces))

            result = RetrievalResult(
                memory_traces=memory_traces,
                retrieval_confidence=np.mean([c[1] for c in top_candidates]) if top_candidates else 0.0,
                context_match_scores=context_scores,
                total_memories_searched=len(accessible_memories),
                retrieval_time_ms=retrieval_time,
                activation_pattern=activation_pattern,
                attention_weights=attention_weights,
                consolidation_opportunities=consolidation_ops,
            )

            return result

        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            return RetrievalResult()

    async def consolidate_memories(self, force: bool = False) -> int:
        """Perform memory consolidation process."""
        try:
            consolidations = 0
            current_time = datetime.now()

            # Find memories ready for consolidation
            consolidation_candidates = []

            for memory in self.episodic_memories.values():
                if (
                    memory.consolidation_state == ConsolidationState.FRESH
                    and memory.strength >= self.consolidation_threshold
                ):
                    # Check temporal criteria
                    age_hours = (current_time - memory.encoded_at).total_seconds() / 3600
                    if age_hours >= 1 or force:  # At least 1 hour old
                        consolidation_candidates.append(memory)

            # Perform consolidation
            for memory in consolidation_candidates:
                success = await self.consolidator.consolidate_memory(memory)
                if success:
                    consolidations += 1

                    # Move to semantic memory if highly consolidated
                    if memory.access_count > 50 and memory.consolidation_state == ConsolidationState.CONSOLIDATED:
                        semantic_memory = await self._convert_to_semantic(memory)
                        self.semantic_memories[semantic_memory.id] = semantic_memory
                        del self.episodic_memories[memory.id]

            self.stats["consolidations_performed"] += consolidations
            logger.info(f"Consolidated {consolidations} memories")

            return consolidations

        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")
            return 0

    async def forget_memories(self, threshold: float | None = None) -> int:
        """Remove memories below forgetting threshold."""
        if threshold is None:
            threshold = self.forgetting_threshold

        forgotten_count = 0
        current_time = datetime.now()

        # Check episodic memories
        to_forget = []
        for memory_id, memory in self.episodic_memories.items():
            memory.update_accessibility(current_time)
            if memory.accessibility < threshold:
                to_forget.append(memory_id)

        for memory_id in to_forget:
            del self.episodic_memories[memory_id]
            forgotten_count += 1

        self.stats["forgotten_memories"] += forgotten_count
        logger.info(f"Forgot {forgotten_count} memories")

        return forgotten_count

    async def get_status(self) -> dict[str, Any]:
        """Get system status and statistics."""
        try:
            current_time = datetime.now()

            # Calculate memory health metrics
            total_memories = len(self.episodic_memories) + len(self.semantic_memories)

            # Accessibility distribution
            accessibility_scores = []
            for memory in list(self.episodic_memories.values()) + list(self.semantic_memories.values()):
                memory.update_accessibility(current_time)
                accessibility_scores.append(memory.accessibility)

            avg_accessibility = np.mean(accessibility_scores) if accessibility_scores else 0.0

            # Consolidation state distribution
            consolidation_dist = {state.value: 0 for state in ConsolidationState}
            for memory in self.episodic_memories.values():
                consolidation_dist[memory.consolidation_state.value] += 1

            return {
                "status": "healthy",
                "memory_statistics": {
                    "total_memories": total_memories,
                    "episodic_memories": len(self.episodic_memories),
                    "semantic_memories": len(self.semantic_memories),
                    "working_memory_size": len(self.working_memory),
                },
                "health_metrics": {
                    "average_accessibility": avg_accessibility,
                    "consolidation_distribution": consolidation_dist,
                    "neural_index_size": len(self.memory_id_mapping),
                },
                "performance": {
                    "average_retrieval_time_ms": self.stats["average_retrieval_time"],
                    "total_retrievals": self.stats["retrievals_performed"],
                    "consolidation_rate": self.stats["consolidations_performed"]
                    / max(1, self.stats["memories_encoded"]),
                },
                "configuration": {
                    "embedding_dim": self.embedding_dim,
                    "max_episodic_memories": self.max_episodic_memories,
                    "consolidation_threshold": self.consolidation_threshold,
                    "forgetting_threshold": self.forgetting_threshold,
                },
                "statistics": self.stats.copy(),
            }

        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return {"status": "error", "error": str(e)}

    # Private implementation methods

    async def _create_contextual_embedding(
        self,
        content: str,
        spatial_context: dict[str, Any] | None = None,
        temporal_context: dict[str, Any] | None = None,
        semantic_context: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Create embedding with contextual information."""
        try:
            # Base content embedding (deterministic for testing)
            content_hash = str(hash(content))
            np.random.seed(int(content_hash[-8:], 16) % 2**32)
            base_embedding = np.random.normal(0, 1, self.embedding_dim).astype(np.float32)

            # Context modulation
            context_modulation = np.zeros(self.embedding_dim)

            # Spatial context influence
            if spatial_context:
                spatial_hash = str(hash(str(spatial_context)))
                np.random.seed(int(spatial_hash[-8:], 16) % 2**32)
                spatial_influence = np.random.normal(0, 0.1, self.embedding_dim)
                context_modulation += spatial_influence

            # Temporal context influence
            if temporal_context:
                temporal_hash = str(hash(str(temporal_context)))
                np.random.seed(int(temporal_hash[-8:], 16) % 2**32)
                temporal_influence = np.random.normal(0, 0.1, self.embedding_dim)
                context_modulation += temporal_influence

            # Semantic context influence
            if semantic_context:
                semantic_hash = str(hash(str(semantic_context)))
                np.random.seed(int(semantic_hash[-8:], 16) % 2**32)
                semantic_influence = np.random.normal(0, 0.1, self.embedding_dim)
                context_modulation += semantic_influence

            # Combine base and context
            final_embedding = base_embedding + context_modulation

            # Normalize
            norm = np.linalg.norm(final_embedding)
            if norm > 0:
                final_embedding = final_embedding / norm

            return final_embedding

        except Exception as e:
            logger.warning(f"Embedding creation failed: {e}")
            return np.random.normal(0, 1, self.embedding_dim).astype(np.float32)

    async def _calculate_neural_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate neural similarity between embeddings."""
        try:
            # Cosine similarity
            similarity = np.dot(embedding1, embedding2)
            return float(max(0.0, similarity))  # Ensure non-negative
        except Exception:
            return 0.0

    def _calculate_context_similarity(
        self,
        memory: MemoryTrace,
        spatial_context: dict[str, Any] | None = None,
        temporal_context: dict[str, Any] | None = None,
        semantic_context: dict[str, Any] | None = None,
    ) -> float:
        """Calculate context similarity score."""
        try:
            similarities = []

            # Spatial context similarity
            if spatial_context and memory.spatial_context:
                spatial_sim = self._dict_similarity(spatial_context, memory.spatial_context)
                similarities.append(spatial_sim)

            # Temporal context similarity
            if temporal_context and memory.temporal_context:
                temporal_sim = self._dict_similarity(temporal_context, memory.temporal_context)
                similarities.append(temporal_sim)

            # Semantic context similarity
            if semantic_context and memory.semantic_context:
                semantic_sim = self._dict_similarity(semantic_context, memory.semantic_context)
                similarities.append(semantic_sim)

            return np.mean(similarities) if similarities else 0.5

        except Exception:
            return 0.0

    def _dict_similarity(self, dict1: dict[str, Any], dict2: dict[str, Any]) -> float:
        """Calculate similarity between dictionaries."""
        try:
            # Simple key overlap similarity
            keys1 = set(dict1.keys())
            keys2 = set(dict2.keys())

            if not keys1 and not keys2:
                return 1.0
            if not keys1 or not keys2:
                return 0.0

            overlap = len(keys1.intersection(keys2))
            union = len(keys1.union(keys2))

            return overlap / union if union > 0 else 0.0

        except Exception:
            return 0.0

    async def _update_context_networks(self, memory: MemoryTrace):
        """Update context association networks."""
        try:
            memory_id = memory.id

            # Update spatial network
            for key in memory.spatial_context.keys():
                if key not in self.spatial_network:
                    self.spatial_network[key] = []
                self.spatial_network[key].append(memory_id)

            # Update temporal network
            for key in memory.temporal_context.keys():
                if key not in self.temporal_network:
                    self.temporal_network[key] = []
                self.temporal_network[key].append(memory_id)

            # Update semantic network
            for key in memory.semantic_context.keys():
                if key not in self.semantic_network:
                    self.semantic_network[key] = []
                self.semantic_network[key].append(memory_id)

        except Exception as e:
            logger.warning(f"Failed to update context networks: {e}")

    async def _add_to_neural_index(self, memory: MemoryTrace):
        """Add memory to neural embedding index."""
        try:
            # Add to embedding matrix
            if self.memory_embeddings.shape[0] == 0:
                self.memory_embeddings = memory.embedding.reshape(1, -1)
            else:
                self.memory_embeddings = np.vstack([self.memory_embeddings, memory.embedding.reshape(1, -1)])

            # Add to ID mapping
            self.memory_id_mapping.append(memory.id)

        except Exception as e:
            logger.warning(f"Failed to add to neural index: {e}")

    def _create_activation_pattern(self, candidates: list[tuple]) -> np.ndarray:
        """Create neural activation pattern from retrieval results."""
        try:
            if not candidates:
                return np.array([])

            # Create activation pattern based on scores
            scores = [c[1] for c in candidates]
            activation = softmax(np.array(scores))

            return activation.astype(np.float32)

        except Exception:
            return np.array([])

    async def _identify_consolidation_opportunities(self, memory_traces: list[MemoryTrace]) -> list[dict[str, Any]]:
        """Identify opportunities for memory consolidation."""
        try:
            opportunities = []

            # Look for memories with similar patterns that could be consolidated
            for i, memory1 in enumerate(memory_traces):
                for j, memory2 in enumerate(memory_traces[i + 1 :], i + 1):
                    similarity = await self._calculate_neural_similarity(memory1.embedding, memory2.embedding)

                    if similarity > 0.8:  # High similarity threshold
                        opportunities.append(
                            {
                                "type": "similar_memories",
                                "memory_ids": [memory1.id, memory2.id],
                                "similarity": similarity,
                                "consolidation_potential": similarity * 0.7,
                            }
                        )

            return opportunities

        except Exception as e:
            logger.warning(f"Failed to identify consolidation opportunities: {e}")
            return []

    async def _forget_oldest_memories(self):
        """Remove oldest memories when capacity limit is reached."""
        try:
            # Sort by last accessed time and remove oldest 10%
            sorted_memories = sorted(self.episodic_memories.items(), key=lambda x: x[1].last_accessed)

            num_to_forget = len(sorted_memories) // 10
            for i in range(num_to_forget):
                memory_id = sorted_memories[i][0]
                del self.episodic_memories[memory_id]
                self.stats["forgotten_memories"] += 1

        except Exception as e:
            logger.warning(f"Failed to forget oldest memories: {e}")

    async def _convert_to_semantic(self, episodic_memory: MemoryTrace) -> MemoryTrace:
        """Convert episodic memory to semantic memory."""
        semantic_memory = MemoryTrace(
            content=episodic_memory.content,
            embedding=episodic_memory.embedding.copy(),
            memory_type=MemoryType.SEMANTIC,
            consolidation_state=ConsolidationState.CONSOLIDATED,
            strength=episodic_memory.strength,
            accessibility=episodic_memory.accessibility,
            semantic_context=episodic_memory.semantic_context,
            metadata=episodic_memory.metadata.copy(),
        )

        return semantic_memory

    async def _background_consolidation(self):
        """Background task for periodic memory consolidation."""
        while True:
            try:
                await asyncio.sleep(3600)  # Every hour
                await self.consolidate_memories()
                await self.forget_memories()
            except Exception as e:
                logger.error(f"Background consolidation failed: {e}")

    async def _memory_replay(self, recent_memories: list[MemoryTrace]):
        """Perform memory replay to strengthen patterns."""
        try:
            # Select memories for replay based on importance and recency
            replay_candidates = [m for m in recent_memories if m.strength > 0.5 and m.access_count > 2]

            # Strengthen associations between replayed memories
            for memory in replay_candidates:
                memory.reinforce(strength_boost=0.05)

            self.stats["replay_sessions"] += 1
            logger.debug(f"Memory replay strengthened {len(replay_candidates)} memories")

        except Exception as e:
            logger.warning(f"Memory replay failed: {e}")

    def _update_average_retrieval_time(self, new_time: float):
        """Update rolling average retrieval time."""
        current_avg = self.stats["average_retrieval_time"]
        count = self.stats["retrievals_performed"]

        if count == 1:
            self.stats["average_retrieval_time"] = new_time
        else:
            self.stats["average_retrieval_time"] = (current_avg * (count - 1) + new_time) / count


class AttentionMechanism:
    """Attention mechanism for memory retrieval."""

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.last_attention_weights = np.array([])

    async def initialize(self):
        """Initialize attention mechanism."""
        logger.info("AttentionMechanism initialized")

    async def compute_attention(
        self,
        neural_similarity: float,
        context_similarity: float,
        accessibility: float,
        strength: float,
        attention_focus: str | None = None,
    ) -> float:
        """Compute attention-weighted retrieval score."""
        try:
            # Base attention from similarities
            base_attention = neural_similarity * 0.4 + context_similarity * 0.3 + accessibility * 0.2 + strength * 0.1

            # Focus modulation
            focus_boost = 0.0
            if attention_focus:
                # Simple focus boost (would be more sophisticated in production)
                focus_boost = 0.1

            final_score = min(1.0, base_attention + focus_boost)

            # Store attention weights for analysis
            self.last_attention_weights = np.array(
                [neural_similarity * 0.4, context_similarity * 0.3, accessibility * 0.2, strength * 0.1, focus_boost]
            )

            return final_score

        except Exception:
            return neural_similarity  # Fallback to simple similarity

    def get_last_attention_weights(self) -> np.ndarray:
        """Get the last computed attention weights."""
        return self.last_attention_weights


class MemoryConsolidator:
    """Memory consolidation system."""

    def __init__(self, consolidation_threshold: float):
        self.consolidation_threshold = consolidation_threshold

    async def initialize(self):
        """Initialize memory consolidator."""
        logger.info("MemoryConsolidator initialized")

    async def consolidate_memory(self, memory: MemoryTrace) -> bool:
        """Consolidate a single memory."""
        try:
            if memory.strength >= self.consolidation_threshold:
                # Update consolidation state
                if memory.consolidation_state == ConsolidationState.FRESH:
                    memory.consolidation_state = ConsolidationState.CONSOLIDATING
                elif memory.consolidation_state == ConsolidationState.CONSOLIDATING:
                    memory.consolidation_state = ConsolidationState.CONSOLIDATED

                # Boost strength slightly during consolidation
                memory.strength = min(1.0, memory.strength + 0.05)

                logger.debug(f"Consolidated memory {memory.id} to {memory.consolidation_state.value}")
                return True

        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")

        return False


# Factory functions


def create_spatial_context(location: str, environment: str) -> dict[str, Any]:
    """Create spatial context dictionary."""
    return {"location": location, "environment": environment, "spatial_type": "location_based"}


def create_temporal_context(timestamp: datetime | None = None, event_type: str = "general") -> dict[str, Any]:
    """Create temporal context dictionary."""
    return {"timestamp": timestamp or datetime.now(), "event_type": event_type, "temporal_type": "event_based"}


def create_semantic_context(domain: str, topic: str, keywords: list[str] | None = None) -> dict[str, Any]:
    """Create semantic context dictionary."""
    return {"domain": domain, "topic": topic, "keywords": keywords or [], "semantic_type": "topic_based"}


if __name__ == "__main__":

    async def test_hippo_rag():
        """Test HippoRAG functionality."""
        # Create system
        hippo_rag = HippoRAG(
            embedding_dim=384, max_episodic_memories=1000, consolidation_threshold=0.7, forgetting_threshold=0.1
        )

        await hippo_rag.initialize()

        # Create contexts
        spatial_ctx = create_spatial_context("laboratory", "research_facility")
        temporal_ctx = create_temporal_context(event_type="learning_session")
        semantic_ctx = create_semantic_context("neuroscience", "memory_research", ["hippocampus", "consolidation"])

        # Encode memories
        memory_id = await hippo_rag.encode_memory(
            content="The hippocampus is crucial for episodic memory formation and spatial navigation.",
            memory_type=MemoryType.EPISODIC,
            spatial_context=spatial_ctx,
            temporal_context=temporal_ctx,
            semantic_context=semantic_ctx,
            metadata={"importance": "high", "source": "research_paper"},
        )

        print(f"Encoded memory: {memory_id}")

        # Retrieve memories
        results = await hippo_rag.retrieve_memories(
            query="hippocampus memory formation",
            k=5,
            spatial_context=spatial_ctx,
            temporal_context=temporal_ctx,
            semantic_context=semantic_ctx,
        )

        print(f"Retrieved {len(results.memory_traces)} memories")
        print(f"Retrieval confidence: {results.retrieval_confidence:.3f}")

        # Check status
        status = await hippo_rag.get_status()
        print(f"System status: {status['status']}")
        print(f"Memory statistics: {status['memory_statistics']}")

        # Perform consolidation
        consolidations = await hippo_rag.consolidate_memories(force=True)
        print(f"Consolidated {consolidations} memories")

    import asyncio

    asyncio.run(test_hippo_rag())
