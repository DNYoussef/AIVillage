"""
HippoRAG Memory System - Neurobiologically-Inspired Memory Architecture

Hippocampus-inspired memory system with rapid episodic storage, consolidation,
and retrieval using Memory MCP for persistent storage and learning patterns.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory in the hippocampal system."""
    
    EPISODIC = "episodic"      # Recent experiences and events
    SEMANTIC = "semantic"      # Consolidated factual knowledge
    WORKING = "working"        # Short-term active memory
    PROCEDURAL = "procedural"  # How-to knowledge


class ConsolidationState(Enum):
    """States of memory consolidation."""
    
    NEW = "new"                    # Just encoded
    CONSOLIDATING = "consolidating" # Being processed
    CONSOLIDATED = "consolidated"   # Stable long-term
    RECONSOLIDATING = "reconsolidating" # Being updated


@dataclass
class EpisodicMemory:
    """A single episodic memory with hippocampal properties."""
    
    memory_id: str
    content: str
    context: str = ""
    
    # Hippocampal properties
    encoding_strength: float = 1.0
    consolidation_state: ConsolidationState = ConsolidationState.NEW
    retrieval_count: int = 0
    last_retrieved: Optional[datetime] = None
    
    # Temporal properties
    encoded_at: datetime = field(default_factory=datetime.now)
    decay_rate: float = 0.1
    forgetting_curve: float = 1.0
    
    # Associative links
    associated_memories: List[str] = field(default_factory=list)
    semantic_links: List[str] = field(default_factory=list)
    
    # Neural patterns
    pattern_completion: float = 0.0
    pattern_separation: float = 0.0
    
    # Embeddings
    content_embedding: Optional[np.ndarray] = None
    context_embedding: Optional[np.ndarray] = None
    
    # Metadata
    source_document: str = ""
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def apply_forgetting_curve(self) -> float:
        """Apply Ebbinghaus forgetting curve to memory strength."""
        if not self.last_retrieved:
            time_since_encoding = (datetime.now() - self.encoded_at).total_seconds() / 3600  # hours
            self.forgetting_curve = self.encoding_strength * np.exp(-self.decay_rate * time_since_encoding)
        else:
            time_since_retrieval = (datetime.now() - self.last_retrieved).total_seconds() / 3600
            self.forgetting_curve = self.encoding_strength * np.exp(-self.decay_rate * time_since_retrieval)
        
        return self.forgetting_curve
    
    def strengthen_memory(self, reinforcement: float = 0.2):
        """Strengthen memory through retrieval or rehearsal."""
        self.encoding_strength = min(2.0, self.encoding_strength + reinforcement)
        self.retrieval_count += 1
        self.last_retrieved = datetime.now()
        
        # Reset forgetting curve
        self.forgetting_curve = self.encoding_strength


@dataclass
class MemoryQuery:
    """Query for memory retrieval."""
    
    query_text: str
    context: str = ""
    memory_types: List[MemoryType] = field(default_factory=lambda: [MemoryType.EPISODIC])
    
    # Retrieval parameters
    max_results: int = 10
    min_strength: float = 0.1
    max_age_hours: Optional[int] = None
    
    # Pattern matching
    enable_pattern_completion: bool = True
    similarity_threshold: float = 0.7
    
    # User context
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class MemoryResult:
    """Result from memory retrieval."""
    
    memories: List[EpisodicMemory] = field(default_factory=list)
    total_found: int = 0
    retrieval_time_ms: float = 0.0
    
    # Retrieval metadata
    pattern_matches: int = 0
    consolidated_matches: int = 0
    episodic_matches: int = 0
    
    # Quality metrics
    avg_strength: float = 0.0
    avg_relevance: float = 0.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class HippoMemorySystem:
    """
    Hippocampus-Inspired Memory System
    
    Neurobiologically-inspired memory architecture that mimics hippocampal
    function with rapid episodic encoding, consolidation, and associative
    retrieval using Memory MCP for persistent storage and learning.
    
    Features:
    - Rapid episodic memory encoding
    - Automatic consolidation to semantic memory
    - Forgetting curves and memory decay
    - Pattern completion and separation
    - Associative memory networks
    - MCP integration for persistent learning
    """
    
    def __init__(
        self,
        mcp_coordinator=None,
        consolidation_threshold: float = 3.0,  # Strength needed for consolidation
        consolidation_interval: int = 3600,    # 1 hour in seconds
        max_episodic_memories: int = 10000
    ):
        self.mcp_coordinator = mcp_coordinator
        self.consolidation_threshold = consolidation_threshold
        self.consolidation_interval = consolidation_interval
        self.max_episodic_memories = max_episodic_memories
        
        # Memory storage
        self.episodic_memories: Dict[str, EpisodicMemory] = {}
        self.semantic_memories: Dict[str, Any] = {}
        self.working_memory: Dict[str, Any] = {}
        
        # Consolidation system
        self.consolidation_queue: List[str] = []
        self.last_consolidation: datetime = datetime.now()
        
        # Neural patterns
        self.pattern_index: Dict[str, List[str]] = {}  # pattern -> memory_ids
        self.association_matrix: np.ndarray = np.array([])
        
        # Statistics
        self.stats = {
            "memories_encoded": 0,
            "memories_consolidated": 0,
            "retrievals_performed": 0,
            "pattern_completions": 0,
            "mcp_operations": 0,
            "consolidation_cycles": 0
        }
        
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the hippocampal memory system."""
        try:
            logger.info("🧠 Initializing HippoRAG Memory System...")
            
            # Load persistent memories from MCP if available
            if self.mcp_coordinator:
                await self._load_persistent_memories()
            
            # Initialize neural patterns
            await self._initialize_pattern_networks()
            
            # Start consolidation process
            asyncio.create_task(self._consolidation_worker())
            
            # Start memory maintenance
            asyncio.create_task(self._memory_maintenance())
            
            self.initialized = True
            logger.info("✅ HippoRAG Memory System initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ HippoRAG Memory System initialization failed: {e}")
            return False
    
    async def encode_memory(
        self,
        content: str,
        context: str = "",
        source_document: str = "",
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Encode new episodic memory with hippocampal processing."""
        try:
            # Create memory ID
            memory_id = f"mem_{int(time.time() * 1000)}_{hash(content) % 10000}"
            
            # Create episodic memory
            memory = EpisodicMemory(
                memory_id=memory_id,
                content=content,
                context=context,
                source_document=source_document,
                user_id=user_id,
                metadata=metadata or {}
            )
            
            # Generate embeddings using MCP
            if self.mcp_coordinator:
                try:
                    embeddings = await self.mcp_coordinator.generate_embeddings([content])
                    if embeddings is not None and len(embeddings) > 0:
                        memory.content_embedding = embeddings[0]
                        
                    if context:
                        context_embeddings = await self.mcp_coordinator.generate_embeddings([context])
                        if context_embeddings is not None and len(context_embeddings) > 0:
                            memory.context_embedding = context_embeddings[0]
                    
                    self.stats["mcp_operations"] += 1
                    
                except Exception as e:
                    logger.warning(f"MCP embedding generation failed: {e}")
            
            # Apply hippocampal processing
            await self._hippocampal_processing(memory)
            
            # Store in episodic memory
            self.episodic_memories[memory_id] = memory
            
            # Update pattern networks
            await self._update_pattern_networks(memory)
            
            # Check for consolidation eligibility
            if memory.encoding_strength >= self.consolidation_threshold:
                self.consolidation_queue.append(memory_id)
            
            # Manage memory capacity
            await self._manage_memory_capacity()
            
            # Store in MCP for persistence
            if self.mcp_coordinator:
                await self._store_memory_in_mcp(memory)
            
            self.stats["memories_encoded"] += 1
            logger.debug(f"Encoded memory {memory_id}: strength={memory.encoding_strength:.2f}")
            
            return memory_id
            
        except Exception as e:
            logger.error(f"Memory encoding failed: {e}")
            return ""
    
    async def retrieve_memories(self, query: MemoryQuery) -> MemoryResult:
        """Retrieve memories using hippocampal pattern matching."""
        start_time = time.time()
        
        try:
            logger.debug(f"Retrieving memories for query: '{query.query_text[:50]}...'")
            
            candidate_memories = []
            
            # Pattern completion search
            if query.enable_pattern_completion:
                pattern_matches = await self._pattern_completion_search(query)
                candidate_memories.extend(pattern_matches)
            
            # Similarity search using embeddings
            if self.mcp_coordinator:
                embedding_matches = await self._embedding_similarity_search(query)
                candidate_memories.extend(embedding_matches)
            
            # Associative search
            associative_matches = await self._associative_search(query)
            candidate_memories.extend(associative_matches)
            
            # Remove duplicates
            unique_memories = {}
            for memory in candidate_memories:
                if memory.memory_id not in unique_memories:
                    unique_memories[memory.memory_id] = memory
            
            candidate_memories = list(unique_memories.values())
            
            # Apply filters
            filtered_memories = await self._apply_retrieval_filters(candidate_memories, query)
            
            # Apply forgetting curves and strengthen retrieved memories
            for memory in filtered_memories:
                memory.apply_forgetting_curve()
                memory.strengthen_memory(0.1)  # Small reinforcement from retrieval
            
            # Sort by relevance and strength
            filtered_memories.sort(key=lambda m: m.forgetting_curve * m.encoding_strength, reverse=True)
            
            # Limit results
            final_memories = filtered_memories[:query.max_results]
            
            # Create result
            retrieval_time = (time.time() - start_time) * 1000
            
            result = MemoryResult(
                memories=final_memories,
                total_found=len(candidate_memories),
                retrieval_time_ms=retrieval_time,
                pattern_matches=len([m for m in final_memories if m.pattern_completion > 0.5]),
                consolidated_matches=len([m for m in final_memories if m.consolidation_state == ConsolidationState.CONSOLIDATED]),
                episodic_matches=len([m for m in final_memories if m.consolidation_state in [ConsolidationState.NEW, ConsolidationState.CONSOLIDATING]]),
                avg_strength=np.mean([m.encoding_strength for m in final_memories]) if final_memories else 0.0,
                avg_relevance=np.mean([m.forgetting_curve for m in final_memories]) if final_memories else 0.0
            )
            
            # Update statistics
            self.stats["retrievals_performed"] += 1
            if query.enable_pattern_completion:
                self.stats["pattern_completions"] += 1
            
            logger.debug(f"Retrieved {len(final_memories)} memories in {retrieval_time:.1f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            return MemoryResult(retrieval_time_ms=(time.time() - start_time) * 1000)
    
    async def store_document(self, content: str, doc_id: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store document as episodic memories."""
        try:
            # Chunk document for episodic storage
            chunks = await self._chunk_for_episodic_storage(content, doc_id)
            
            success_count = 0
            for i, chunk in enumerate(chunks):
                memory_id = await self.encode_memory(
                    content=chunk,
                    context=f"Document: {doc_id}, Chunk: {i+1}/{len(chunks)}",
                    source_document=doc_id,
                    metadata={**(metadata or {}), "chunk_index": i, "total_chunks": len(chunks)}
                )
                if memory_id:
                    success_count += 1
            
            logger.info(f"Stored document {doc_id} as {success_count} episodic memories")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Document storage failed: {e}")
            return False
    
    async def query(self, question: str, limit: int = 10, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query memories and return results for RAG system integration."""
        query = MemoryQuery(
            query_text=question,
            max_results=limit,
            user_id=user_id,
            memory_types=[MemoryType.EPISODIC, MemoryType.SEMANTIC]
        )
        
        result = await self.retrieve_memories(query)
        
        # Convert to standard format for RAG integration
        formatted_results = []
        for memory in result.memories:
            formatted_results.append({
                "id": memory.memory_id,
                "content": memory.content,
                "context": memory.context,
                "confidence": memory.forgetting_curve,
                "source": memory.source_document,
                "metadata": {
                    **memory.metadata,
                    "memory_type": "episodic",
                    "encoding_strength": memory.encoding_strength,
                    "retrieval_count": memory.retrieval_count,
                    "consolidation_state": memory.consolidation_state.value
                }
            })
        
        return formatted_results
    
    async def _hippocampal_processing(self, memory: EpisodicMemory):
        """Apply hippocampal processing to new memory."""
        # Pattern separation: ensure memory is distinct from existing ones
        memory.pattern_separation = await self._calculate_pattern_separation(memory)
        
        # Encoding strength based on novelty and importance
        novelty_bonus = memory.pattern_separation * 0.5
        memory.encoding_strength += novelty_bonus
        
        # Context processing
        if memory.context:
            memory.encoding_strength += 0.2  # Context bonus
    
    async def _pattern_completion_search(self, query: MemoryQuery) -> List[EpisodicMemory]:
        """Search using hippocampal pattern completion."""
        matches = []
        
        # Extract query patterns (simple keyword matching)
        query_words = set(query.query_text.lower().split())
        
        for memory in self.episodic_memories.values():
            memory_words = set(memory.content.lower().split())
            
            # Calculate pattern overlap
            overlap = len(query_words.intersection(memory_words))
            total_words = len(query_words.union(memory_words))
            
            if total_words > 0:
                pattern_score = overlap / total_words
                memory.pattern_completion = pattern_score
                
                if pattern_score >= query.similarity_threshold:
                    matches.append(memory)
        
        return matches
    
    async def _embedding_similarity_search(self, query: MemoryQuery) -> List[EpisodicMemory]:
        """Search using embedding similarity via MCP."""
        matches = []
        
        try:
            if not self.mcp_coordinator:
                return matches
            
            # Generate query embedding
            query_embeddings = await self.mcp_coordinator.generate_embeddings([query.query_text])
            if not query_embeddings:
                return matches
            
            query_embedding = query_embeddings[0]
            
            # Compare with memory embeddings
            for memory in self.episodic_memories.values():
                if memory.content_embedding is not None:
                    similarity = np.dot(query_embedding, memory.content_embedding)
                    
                    if similarity >= query.similarity_threshold:
                        memory.pattern_completion = max(memory.pattern_completion, similarity)
                        matches.append(memory)
            
            self.stats["mcp_operations"] += 1
            
        except Exception as e:
            logger.warning(f"Embedding similarity search failed: {e}")
        
        return matches
    
    async def _associative_search(self, query: MemoryQuery) -> List[EpisodicMemory]:
        """Search using associative memory networks."""
        matches = []
        
        # Find memories with associative links to query concepts
        query_concepts = query.query_text.lower().split()
        
        for memory in self.episodic_memories.values():
            # Check direct associations
            for concept in query_concepts:
                if concept in memory.content.lower():
                    # Follow associative links
                    for associated_id in memory.associated_memories:
                        if associated_id in self.episodic_memories:
                            associated_memory = self.episodic_memories[associated_id]
                            if associated_memory not in matches:
                                matches.append(associated_memory)
                    
                    if memory not in matches:
                        matches.append(memory)
        
        return matches
    
    async def _apply_retrieval_filters(self, memories: List[EpisodicMemory], query: MemoryQuery) -> List[EpisodicMemory]:
        """Apply retrieval filters to candidate memories."""
        filtered = []
        
        for memory in memories:
            # Memory type filter
            if MemoryType.EPISODIC in query.memory_types and memory.consolidation_state in [ConsolidationState.NEW, ConsolidationState.CONSOLIDATING]:
                pass
            elif MemoryType.SEMANTIC in query.memory_types and memory.consolidation_state == ConsolidationState.CONSOLIDATED:
                pass
            else:
                continue
            
            # Strength filter
            current_strength = memory.apply_forgetting_curve()
            if current_strength < query.min_strength:
                continue
            
            # Age filter
            if query.max_age_hours:
                age_hours = (datetime.now() - memory.encoded_at).total_seconds() / 3600
                if age_hours > query.max_age_hours:
                    continue
            
            # User filter
            if query.user_id and memory.user_id and memory.user_id != query.user_id:
                continue
            
            filtered.append(memory)
        
        return filtered
    
    async def _calculate_pattern_separation(self, new_memory: EpisodicMemory) -> float:
        """Calculate pattern separation for new memory."""
        if not self.episodic_memories:
            return 1.0  # Maximum separation for first memory
        
        # Compare with recent memories
        recent_memories = [
            m for m in self.episodic_memories.values() 
            if (datetime.now() - m.encoded_at).total_seconds() < 3600  # Last hour
        ]
        
        if not recent_memories:
            return 1.0
        
        # Simple content-based separation
        new_words = set(new_memory.content.lower().split())
        
        min_similarity = float('inf')
        for memory in recent_memories:
            memory_words = set(memory.content.lower().split())
            if memory_words:
                overlap = len(new_words.intersection(memory_words))
                union = len(new_words.union(memory_words))
                similarity = overlap / union if union > 0 else 0
                min_similarity = min(min_similarity, similarity)
        
        return 1.0 - min_similarity  # Higher separation = less similar
    
    async def _update_pattern_networks(self, memory: EpisodicMemory):
        """Update pattern networks with new memory."""
        # Extract patterns (keywords) from memory
        words = memory.content.lower().split()
        
        for word in words:
            if len(word) > 3:  # Filter short words
                if word not in self.pattern_index:
                    self.pattern_index[word] = []
                self.pattern_index[word].append(memory.memory_id)
        
        # Create associative links with similar memories
        for existing_id, existing_memory in self.episodic_memories.items():
            if existing_id != memory.memory_id:
                similarity = await self._calculate_memory_similarity(memory, existing_memory)
                if similarity > 0.3:  # Threshold for association
                    memory.associated_memories.append(existing_id)
                    existing_memory.associated_memories.append(memory.memory_id)
    
    async def _calculate_memory_similarity(self, memory1: EpisodicMemory, memory2: EpisodicMemory) -> float:
        """Calculate similarity between two memories."""
        if memory1.content_embedding is not None and memory2.content_embedding is not None:
            return float(np.dot(memory1.content_embedding, memory2.content_embedding))
        
        # Fallback to text similarity
        words1 = set(memory1.content.lower().split())
        words2 = set(memory2.content.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        overlap = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return overlap / union if union > 0 else 0.0
    
    async def _chunk_for_episodic_storage(self, content: str, doc_id: str) -> List[str]:
        """Chunk document for episodic memory storage."""
        # Simple sentence-based chunking for episodic memories
        sentences = content.split('. ')
        chunks = []
        
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Target chunk size for episodic memory (smaller than semantic)
            if len(current_chunk + sentence) < 200 or not current_chunk:
                current_chunk = current_chunk + ". " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks or [content]  # Fallback to full content
    
    async def _consolidation_worker(self):
        """Background worker for memory consolidation."""
        while True:
            try:
                await asyncio.sleep(self.consolidation_interval)
                
                if self.consolidation_queue:
                    await self._consolidate_memories()
                    self.stats["consolidation_cycles"] += 1
                    
            except Exception as e:
                logger.error(f"Consolidation worker error: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _consolidate_memories(self):
        """Consolidate eligible memories to semantic storage."""
        logger.debug(f"Starting consolidation of {len(self.consolidation_queue)} memories")
        
        consolidated_count = 0
        
        while self.consolidation_queue and consolidated_count < 10:  # Limit per cycle
            memory_id = self.consolidation_queue.pop(0)
            
            if memory_id in self.episodic_memories:
                memory = self.episodic_memories[memory_id]
                
                if memory.encoding_strength >= self.consolidation_threshold:
                    # Move to semantic memory
                    semantic_memory = {
                        "content": memory.content,
                        "context": memory.context,
                        "strength": memory.encoding_strength,
                        "consolidation_date": datetime.now(),
                        "source_episodic_id": memory_id,
                        "retrieval_count": memory.retrieval_count
                    }
                    
                    self.semantic_memories[memory_id] = semantic_memory
                    memory.consolidation_state = ConsolidationState.CONSOLIDATED
                    
                    # Store consolidated memory in MCP
                    if self.mcp_coordinator:
                        await self.mcp_coordinator.store_memory(
                            f"semantic_{memory_id}",
                            semantic_memory,
                            "semantic"
                        )
                    
                    consolidated_count += 1
                    self.stats["memories_consolidated"] += 1
        
        if consolidated_count > 0:
            logger.info(f"Consolidated {consolidated_count} memories to semantic storage")
    
    async def _memory_maintenance(self):
        """Background memory maintenance and cleanup."""
        while True:
            try:
                await asyncio.sleep(3600)  # Every hour
                
                # Apply forgetting curves
                for memory in self.episodic_memories.values():
                    memory.apply_forgetting_curve()
                
                # Remove very weak memories
                weak_memories = [
                    mid for mid, mem in self.episodic_memories.items()
                    if mem.forgetting_curve < 0.01 and mem.retrieval_count == 0
                ]
                
                for memory_id in weak_memories:
                    del self.episodic_memories[memory_id]
                
                if weak_memories:
                    logger.info(f"Removed {len(weak_memories)} weak memories")
                
            except Exception as e:
                logger.error(f"Memory maintenance error: {e}")
                await asyncio.sleep(3600)
    
    async def _manage_memory_capacity(self):
        """Manage episodic memory capacity."""
        if len(self.episodic_memories) > self.max_episodic_memories:
            # Remove oldest, weakest memories
            memories_by_strength = sorted(
                self.episodic_memories.items(),
                key=lambda x: (x[1].encoding_strength, x[1].encoded_at),
                reverse=False  # Weakest and oldest first
            )
            
            to_remove = len(self.episodic_memories) - self.max_episodic_memories + 100  # Remove extra buffer
            
            for i in range(min(to_remove, len(memories_by_strength))):
                memory_id, _ = memories_by_strength[i]
                del self.episodic_memories[memory_id]
            
            logger.info(f"Removed {to_remove} memories to manage capacity")
    
    async def _load_persistent_memories(self):
        """Load persistent memories from MCP."""
        try:
            if not self.mcp_coordinator:
                return
            
            # Load episodic memories
            episodic_data = await self.mcp_coordinator.retrieve_memory("episodic_memories", "episodic")
            if episodic_data and isinstance(episodic_data, dict):
                logger.info(f"Loaded {len(episodic_data)} persistent episodic memories")
            
            # Load semantic memories
            semantic_data = await self.mcp_coordinator.retrieve_memory("semantic_memories", "semantic")
            if semantic_data and isinstance(semantic_data, dict):
                self.semantic_memories.update(semantic_data)
                logger.info(f"Loaded {len(semantic_data)} persistent semantic memories")
            
        except Exception as e:
            logger.warning(f"Failed to load persistent memories: {e}")
    
    async def _store_memory_in_mcp(self, memory: EpisodicMemory):
        """Store memory in MCP for persistence."""
        try:
            if not self.mcp_coordinator:
                return
            
            memory_data = {
                "memory_id": memory.memory_id,
                "content": memory.content,
                "context": memory.context,
                "encoding_strength": memory.encoding_strength,
                "encoded_at": memory.encoded_at.isoformat(),
                "source_document": memory.source_document,
                "metadata": memory.metadata
            }
            
            await self.mcp_coordinator.store_memory(
                f"episodic_{memory.memory_id}",
                memory_data,
                "episodic"
            )
            
        except Exception as e:
            logger.debug(f"Failed to store memory in MCP: {e}")
    
    async def _initialize_pattern_networks(self):
        """Initialize neural pattern networks."""
        # Initialize empty association matrix
        self.association_matrix = np.zeros((0, 0), dtype=np.float32)
        logger.debug("Initialized pattern networks")
    
    async def get_memory_status(self) -> Dict[str, Any]:
        """Get comprehensive memory system status."""
        return {
            "initialized": self.initialized,
            "memory_counts": {
                "episodic": len(self.episodic_memories),
                "semantic": len(self.semantic_memories),
                "working": len(self.working_memory),
                "consolidation_queue": len(self.consolidation_queue)
            },
            "statistics": self.stats.copy(),
            "configuration": {
                "consolidation_threshold": self.consolidation_threshold,
                "consolidation_interval": self.consolidation_interval,
                "max_episodic_memories": self.max_episodic_memories,
                "mcp_integration": self.mcp_coordinator is not None
            },
            "memory_health": {
                "avg_encoding_strength": np.mean([m.encoding_strength for m in self.episodic_memories.values()]) if self.episodic_memories else 0.0,
                "consolidation_rate": self.stats["memories_consolidated"] / max(1, self.stats["memories_encoded"]),
                "pattern_network_size": len(self.pattern_index)
            }
        }
    
    async def close(self):
        """Close the hippocampal memory system."""
        logger.info("Shutting down HippoRAG Memory System...")
        
        # Save persistent state to MCP if available
        if self.mcp_coordinator:
            try:
                await self.mcp_coordinator.store_memory(
                    "memory_system_state",
                    {
                        "stats": self.stats,
                        "episodic_count": len(self.episodic_memories),
                        "semantic_count": len(self.semantic_memories),
                        "shutdown_time": datetime.now().isoformat()
                    },
                    "system"
                )
            except Exception as e:
                logger.warning(f"Failed to save system state: {e}")
        
        # Clear memory structures
        self.episodic_memories.clear()
        self.semantic_memories.clear()
        self.working_memory.clear()
        self.consolidation_queue.clear()
        self.pattern_index.clear()
        
        self.initialized = False
        logger.info("HippoRAG Memory System shutdown complete")