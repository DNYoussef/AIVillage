"""
Episodic Storage System for HippoRAG Memory
Implements hippocampus-inspired episodic memory storage and retrieval
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

logger = logging.getLogger(__name__)

@dataclass
class EpisodicMemory:
    """Individual episodic memory record"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 0.5
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    
    def update_access(self):
        """Update access tracking"""
        self.access_count += 1
        self.last_accessed = datetime.now()

@dataclass
class EpisodeCluster:
    """Cluster of related episodic memories"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    memories: List[EpisodicMemory] = field(default_factory=list)
    centroid: Optional[List[float]] = None
    importance: float = 0.5
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_memory(self, memory: EpisodicMemory):
        """Add memory to cluster"""
        self.memories.append(memory)
        self._update_importance()
    
    def _update_importance(self):
        """Update cluster importance based on memories"""
        if self.memories:
            self.importance = sum(m.importance for m in self.memories) / len(self.memories)

class EpisodicStorage:
    """
    Episodic storage system implementing hippocampus-inspired memory patterns
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/episodic_storage")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.memories: Dict[str, EpisodicMemory] = {}
        self.clusters: Dict[str, EpisodeCluster] = {}
        self.memory_index: Dict[str, List[str]] = {}  # content -> memory_ids
        
        self._load_storage()
    
    def _load_storage(self):
        """Load existing memories from storage"""
        try:
            memories_file = self.storage_path / "memories.json"
            clusters_file = self.storage_path / "clusters.json"
            
            if memories_file.exists():
                with open(memories_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for mem_data in data:
                        memory = EpisodicMemory(**mem_data)
                        self.memories[memory.id] = memory
            
            if clusters_file.exists():
                with open(clusters_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for cluster_data in data:
                        cluster = EpisodeCluster(**cluster_data)
                        self.clusters[cluster.id] = cluster
                        
            logger.info(f"Loaded {len(self.memories)} memories and {len(self.clusters)} clusters")
            
        except Exception as e:
            logger.warning(f"Failed to load episodic storage: {e}")
    
    def _save_storage(self):
        """Save memories to persistent storage"""
        try:
            memories_file = self.storage_path / "memories.json"
            clusters_file = self.storage_path / "clusters.json"
            
            # Save memories
            memories_data = []
            for memory in self.memories.values():
                mem_dict = {
                    'id': memory.id,
                    'content': memory.content,
                    'context': memory.context,
                    'timestamp': memory.timestamp.isoformat(),
                    'importance': memory.importance,
                    'access_count': memory.access_count,
                    'last_accessed': memory.last_accessed.isoformat() if memory.last_accessed else None,
                    'tags': memory.tags,
                    'embedding': memory.embedding
                }
                memories_data.append(mem_dict)
            
            with open(memories_file, 'w', encoding='utf-8') as f:
                json.dump(memories_data, f, indent=2, ensure_ascii=False)
            
            # Save clusters
            clusters_data = []
            for cluster in self.clusters.values():
                cluster_dict = {
                    'id': cluster.id,
                    'memories': [m.id for m in cluster.memories],
                    'centroid': cluster.centroid,
                    'importance': cluster.importance,
                    'created_at': cluster.created_at.isoformat()
                }
                clusters_data.append(cluster_dict)
            
            with open(clusters_file, 'w', encoding='utf-8') as f:
                json.dump(clusters_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save episodic storage: {e}")
    
    async def store_memory(
        self,
        content: str,
        context: Dict[str, Any] = None,
        importance: float = 0.5,
        tags: List[str] = None,
        embedding: Optional[List[float]] = None
    ) -> str:
        """Store new episodic memory"""
        memory = EpisodicMemory(
            content=content,
            context=context or {},
            importance=importance,
            tags=tags or [],
            embedding=embedding
        )
        
        self.memories[memory.id] = memory
        
        # Update search index
        content_lower = content.lower()
        if content_lower not in self.memory_index:
            self.memory_index[content_lower] = []
        self.memory_index[content_lower].append(memory.id)
        
        # Auto-cluster if we have embeddings
        if embedding:
            await self._auto_cluster_memory(memory)
        
        self._save_storage()
        return memory.id
    
    async def retrieve_memories(
        self,
        query: str,
        limit: int = 10,
        importance_threshold: float = 0.0
    ) -> List[EpisodicMemory]:
        """Retrieve relevant episodic memories"""
        matching_memories = []
        
        query_lower = query.lower()
        
        # Search through memories
        for memory in self.memories.values():
            if memory.importance < importance_threshold:
                continue
            
            # Simple text matching (can be enhanced with embeddings)
            if query_lower in memory.content.lower():
                memory.update_access()
                matching_memories.append(memory)
            
            # Tag matching
            elif any(query_lower in tag.lower() for tag in memory.tags):
                memory.update_access()
                matching_memories.append(memory)
        
        # Sort by importance and recency
        matching_memories.sort(
            key=lambda m: (m.importance, m.timestamp),
            reverse=True
        )
        
        return matching_memories[:limit]
    
    async def _auto_cluster_memory(self, memory: EpisodicMemory):
        """Automatically cluster memory based on similarity"""
        if not memory.embedding:
            return
        
        # Find similar cluster or create new one
        best_cluster = None
        best_similarity = 0.0
        
        for cluster in self.clusters.values():
            if cluster.centroid:
                similarity = self._cosine_similarity(memory.embedding, cluster.centroid)
                if similarity > best_similarity and similarity > 0.7:  # threshold
                    best_similarity = similarity
                    best_cluster = cluster
        
        if best_cluster:
            best_cluster.add_memory(memory)
        else:
            # Create new cluster
            new_cluster = EpisodeCluster(
                memories=[memory],
                centroid=memory.embedding.copy(),
                importance=memory.importance
            )
            self.clusters[new_cluster.id] = new_cluster
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            import numpy as np
            a_arr = np.array(a)
            b_arr = np.array(b)
            return np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
        except:
            return 0.0
    
    async def consolidate_memories(self, max_age_days: int = 30):
        """Consolidate old memories (hippocampus-like forgetting)"""
        cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
        
        memories_to_remove = []
        for memory in self.memories.values():
            if (memory.timestamp.timestamp() < cutoff_time and 
                memory.importance < 0.3 and 
                memory.access_count < 2):
                memories_to_remove.append(memory.id)
        
        for memory_id in memories_to_remove:
            del self.memories[memory_id]
        
        if memories_to_remove:
            logger.info(f"Consolidated {len(memories_to_remove)} old memories")
            self._save_storage()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories"""
        return {
            'total_memories': len(self.memories),
            'total_clusters': len(self.clusters),
            'avg_importance': sum(m.importance for m in self.memories.values()) / len(self.memories) if self.memories else 0,
            'storage_path': str(self.storage_path)
        }