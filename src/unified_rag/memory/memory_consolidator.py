"""
Memory Consolidator for HippoRAG System
Implements sleep-like memory consolidation patterns inspired by hippocampus-neocortex interactions
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .episodic_storage import EpisodicMemory, EpisodeCluster, EpisodicStorage

logger = logging.getLogger(__name__)

@dataclass
class ConsolidationRule:
    """Rule for memory consolidation"""
    name: str
    condition: callable
    action: callable
    priority: int = 0

class MemoryConsolidator:
    """
    Handles memory consolidation processes similar to hippocampus-neocortex transfer
    during sleep and rest periods
    """
    
    def __init__(self, episodic_storage: EpisodicStorage):
        self.episodic_storage = episodic_storage
        self.consolidation_rules: List[ConsolidationRule] = []
        self.last_consolidation: Optional[datetime] = None
        self.consolidation_interval = timedelta(hours=6)  # Consolidate every 6 hours
        
        # Initialize default consolidation rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default memory consolidation rules"""
        
        # Rule 1: Strengthen frequently accessed memories
        def frequent_access_condition(memory: EpisodicMemory) -> bool:
            return memory.access_count >= 5
        
        def strengthen_memory(memory: EpisodicMemory):
            memory.importance = min(1.0, memory.importance + 0.1)
            logger.debug(f"Strengthened memory {memory.id} importance to {memory.importance}")
        
        self.consolidation_rules.append(ConsolidationRule(
            name="strengthen_frequent",
            condition=frequent_access_condition,
            action=strengthen_memory,
            priority=1
        ))
        
        # Rule 2: Fade unused memories
        def unused_condition(memory: EpisodicMemory) -> bool:
            if not memory.last_accessed:
                return False
            days_since_access = (datetime.now() - memory.last_accessed).days
            return days_since_access >= 7 and memory.access_count < 2
        
        def fade_memory(memory: EpisodicMemory):
            memory.importance = max(0.0, memory.importance - 0.05)
            logger.debug(f"Faded memory {memory.id} importance to {memory.importance}")
        
        self.consolidation_rules.append(ConsolidationRule(
            name="fade_unused",
            condition=unused_condition,
            action=fade_memory,
            priority=2
        ))
        
        # Rule 3: Merge similar memories
        def similar_memories_condition(memory: EpisodicMemory) -> bool:
            return len(memory.content) > 50  # Only merge substantial memories
        
        def merge_similar(memory: EpisodicMemory):
            # This would be implemented with similarity matching
            pass
        
        self.consolidation_rules.append(ConsolidationRule(
            name="merge_similar",
            condition=similar_memories_condition,
            action=merge_similar,
            priority=3
        ))
    
    def add_consolidation_rule(self, rule: ConsolidationRule):
        """Add custom consolidation rule"""
        self.consolidation_rules.append(rule)
        self.consolidation_rules.sort(key=lambda r: r.priority)
    
    async def should_consolidate(self) -> bool:
        """Check if consolidation should be triggered"""
        if not self.last_consolidation:
            return True
        
        time_since_last = datetime.now() - self.last_consolidation
        return time_since_last >= self.consolidation_interval
    
    async def consolidate_memories(self) -> Dict[str, Any]:
        """
        Perform memory consolidation process
        Returns statistics about the consolidation
        """
        start_time = datetime.now()
        stats = {
            'total_memories': len(self.episodic_storage.memories),
            'rules_applied': 0,
            'memories_modified': 0,
            'memories_removed': 0,
            'duration_ms': 0
        }
        
        logger.info("Starting memory consolidation process")
        
        try:
            # Apply consolidation rules
            for rule in self.consolidation_rules:
                applied_count = await self._apply_rule(rule)
                if applied_count > 0:
                    stats['rules_applied'] += 1
                    stats['memories_modified'] += applied_count
            
            # Remove very low importance memories
            removed_count = await self._remove_forgotten_memories()
            stats['memories_removed'] = removed_count
            
            # Update cluster relationships
            await self._update_cluster_relationships()
            
            # Strengthen cross-references
            await self._strengthen_cross_references()
            
            self.last_consolidation = datetime.now()
            
        except Exception as e:
            logger.error(f"Error during memory consolidation: {e}")
            raise
        
        finally:
            duration = datetime.now() - start_time
            stats['duration_ms'] = int(duration.total_seconds() * 1000)
            
            logger.info(f"Memory consolidation completed: {stats}")
            
        return stats
    
    async def _apply_rule(self, rule: ConsolidationRule) -> int:
        """Apply a single consolidation rule"""
        applied_count = 0
        
        for memory in list(self.episodic_storage.memories.values()):
            try:
                if rule.condition(memory):
                    rule.action(memory)
                    applied_count += 1
            except Exception as e:
                logger.warning(f"Error applying rule {rule.name} to memory {memory.id}: {e}")
        
        return applied_count
    
    async def _remove_forgotten_memories(self) -> int:
        """Remove memories with very low importance (forgotten)"""
        memories_to_remove = []
        
        for memory_id, memory in self.episodic_storage.memories.items():
            if memory.importance <= 0.05:  # Very low importance threshold
                memories_to_remove.append(memory_id)
        
        for memory_id in memories_to_remove:
            del self.episodic_storage.memories[memory_id]
        
        if memories_to_remove:
            logger.info(f"Removed {len(memories_to_remove)} forgotten memories")
        
        return len(memories_to_remove)
    
    async def _update_cluster_relationships(self):
        """Update memory cluster relationships"""
        # Recalculate cluster centroids and importance
        for cluster in self.episodic_storage.clusters.values():
            if cluster.memories:
                # Update cluster importance
                cluster.importance = np.mean([m.importance for m in cluster.memories])
                
                # Update centroid if we have embeddings
                embeddings = [m.embedding for m in cluster.memories if m.embedding]
                if embeddings:
                    cluster.centroid = np.mean(embeddings, axis=0).tolist()
    
    async def _strengthen_cross_references(self):
        """Strengthen memories that reference each other"""
        # Simple implementation - strengthen memories with common tags
        tag_groups = {}
        
        for memory in self.episodic_storage.memories.values():
            for tag in memory.tags:
                if tag not in tag_groups:
                    tag_groups[tag] = []
                tag_groups[tag].append(memory)
        
        # Strengthen memories in groups with multiple items
        for tag, memories in tag_groups.items():
            if len(memories) >= 3:  # Groups of 3 or more
                boost = 0.02 * len(memories)  # Small boost based on group size
                for memory in memories:
                    memory.importance = min(1.0, memory.importance + boost)
    
    async def force_consolidation(self) -> Dict[str, Any]:
        """Force immediate consolidation regardless of timing"""
        logger.info("Forcing immediate memory consolidation")
        return await self.consolidate_memories()
    
    def get_consolidation_status(self) -> Dict[str, Any]:
        """Get current consolidation status"""
        return {
            'last_consolidation': self.last_consolidation.isoformat() if self.last_consolidation else None,
            'next_consolidation_due': self.should_consolidate(),
            'consolidation_interval_hours': self.consolidation_interval.total_seconds() / 3600,
            'active_rules': len(self.consolidation_rules),
            'rules': [{'name': r.name, 'priority': r.priority} for r in self.consolidation_rules]
        }