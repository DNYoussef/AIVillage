import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import networkx as nx
from rag_system.utils.embedding import get_embedding
from rag_system.utils.named_entity_recognition import extract_entities
from rag_system.utils.relation_extraction import extract_relations

logger = logging.getLogger(__name__)

@dataclass
class KnowledgeChange:
    entity: str
    relation: str
    old_value: Any
    new_value: Any
    timestamp: datetime
    source: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class UnifiedKnowledgeTracker:
    def __init__(self, vector_store, graph_store):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.knowledge_changes: List[KnowledgeChange] = []
        self.knowledge_graph: Dict[str, Dict[str, Any]] = {}
        self.entity_embeddings: Dict[str, List[float]] = {}
        self.relation_patterns: Set[str] = set()
        self.confidence_thresholds: Dict[str, float] = {
            'entity_merge': 0.85,
            'relation_inference': 0.75,
            'knowledge_update': 0.9
        }

    async def record_change(self, change: KnowledgeChange):
        """Record a knowledge change and update all relevant stores."""
        try:
            # Validate change
            if not self._validate_change(change):
                logger.warning(f"Invalid change detected: {change}")
                return

            # Record the change
            self.knowledge_changes.append(change)
            
            # Update knowledge graph
            await self._update_knowledge_graph(change)
            
            # Update vector store
            await self.update_vector_store(change)
            
            # Update graph store
            await self.update_graph_store(change)
            
            # Update entity embeddings
            await self._update_entity_embeddings(change.entity)
            
            logger.info(f"Successfully recorded change for entity {change.entity}")
        except Exception as e:
            logger.exception(f"Error recording change: {str(e)}")
            raise

    async def track_changes(self, result: Dict[str, Any], timestamp: datetime):
        """Track changes from a result and update knowledge stores."""
        try:
            # Extract entities and relations
            entities = await extract_entities(result)
            relations = await extract_relations(result)
            
            # Process each entity and its relations
            for entity in entities:
                # Get current state
                current_state = self.get_current_knowledge(entity)
                
                # Process each relation
                for relation in relations:
                    if relation['subject'] == entity:
                        change = KnowledgeChange(
                            entity=entity,
                            relation=relation['predicate'],
                            old_value=current_state.get(relation['predicate']),
                            new_value=relation['object'],
                            timestamp=timestamp,
                            source="result_processing",
                            confidence=relation.get('confidence', 1.0),
                            metadata={'context': result.get('context')}
                        )
                        await self.record_change(change)
            
            # Perform knowledge integration
            await self._integrate_new_knowledge(result)
            
            logger.info(f"Successfully tracked changes from result")
        except Exception as e:
            logger.exception(f"Error tracking changes: {str(e)}")
            raise

    async def update_vector_store(self, change: Optional[KnowledgeChange] = None):
        """Update the vector store with new knowledge."""
        try:
            if change:
                # Update specific entity
                embedding = await get_embedding(f"{change.entity} {change.relation} {change.new_value}")
                await self.vector_store.update(
                    id=change.entity,
                    vector=embedding,
                    metadata={
                        'relation': change.relation,
                        'value': change.new_value,
                        'timestamp': change.timestamp.isoformat(),
                        'confidence': change.confidence
                    }
                )
            else:
                # Bulk update all entities
                for entity, knowledge in self.knowledge_graph.items():
                    text = f"{entity} " + " ".join(f"{k} {v}" for k, v in knowledge.items())
                    embedding = await get_embedding(text)
                    await self.vector_store.update(
                        id=entity,
                        vector=embedding,
                        metadata=knowledge
                    )
            
            logger.info(f"Successfully updated vector store")
        except Exception as e:
            logger.exception(f"Error updating vector store: {str(e)}")
            raise

    async def update_graph_store(self, change: Optional[KnowledgeChange] = None):
        """Update the graph store with new knowledge."""
        try:
            if change:
                # Update specific relation
                self.graph_store.add_edge(
                    change.entity,
                    change.new_value,
                    key=change.relation,
                    weight=change.confidence,
                    timestamp=change.timestamp.isoformat(),
                    metadata=change.metadata
                )
            else:
                # Rebuild entire graph
                self.graph_store.clear()
                for change in self.knowledge_changes:
                    self.graph_store.add_edge(
                        change.entity,
                        change.new_value,
                        key=change.relation,
                        weight=change.confidence,
                        timestamp=change.timestamp.isoformat(),
                        metadata=change.metadata
                    )
            
            logger.info(f"Successfully updated graph store")
        except Exception as e:
            logger.exception(f"Error updating graph store: {str(e)}")
            raise

    async def _update_knowledge_graph(self, change: KnowledgeChange):
        """Update the internal knowledge graph with new information."""
        try:
            if change.entity not in self.knowledge_graph:
                self.knowledge_graph[change.entity] = {}
            
            # Update the value
            self.knowledge_graph[change.entity][change.relation] = {
                'value': change.new_value,
                'confidence': change.confidence,
                'timestamp': change.timestamp,
                'source': change.source,
                'metadata': change.metadata
            }
            
            # Infer new relations
            await self._infer_new_relations(change)
            
            logger.info(f"Successfully updated knowledge graph for entity {change.entity}")
        except Exception as e:
            logger.exception(f"Error updating knowledge graph: {str(e)}")
            raise

    async def _infer_new_relations(self, change: KnowledgeChange):
        """Infer new relations based on existing knowledge."""
        try:
            # Get related entities
            related_entities = self.graph_store.get_neighbors(change.entity)
            
            for related_entity in related_entities:
                # Get common relations
                common_relations = self._find_common_relations(change.entity, related_entity)
                
                # Infer new relations based on patterns
                for relation in common_relations:
                    confidence = self._calculate_inference_confidence(
                        change.entity,
                        related_entity,
                        relation
                    )
                    
                    if confidence >= self.confidence_thresholds['relation_inference']:
                        inferred_change = KnowledgeChange(
                            entity=related_entity,
                            relation=f"inferred_{relation}",
                            old_value=None,
                            new_value=change.new_value,
                            timestamp=datetime.now(),
                            source="inference",
                            confidence=confidence
                        )
                        await self.record_change(inferred_change)
            
            logger.info(f"Successfully inferred new relations for entity {change.entity}")
        except Exception as e:
            logger.exception(f"Error inferring new relations: {str(e)}")
            raise

    def _find_common_relations(self, entity1: str, entity2: str) -> Set[str]:
        """Find common relations between two entities."""
        relations1 = set(self.knowledge_graph.get(entity1, {}).keys())
        relations2 = set(self.knowledge_graph.get(entity2, {}).keys())
        return relations1.intersection(relations2)

    def _calculate_inference_confidence(
        self,
        entity1: str,
        entity2: str,
        relation: str
    ) -> float:
        """Calculate confidence score for an inferred relation."""
        try:
            # Get embeddings
            emb1 = self.entity_embeddings.get(entity1)
            emb2 = self.entity_embeddings.get(entity2)
            
            if not (emb1 and emb2):
                return 0.0
            
            # Calculate similarity
            similarity = self._calculate_embedding_similarity(emb1, emb2)
            
            # Get relation confidence
            relation_conf = min(
                self.knowledge_graph[entity1][relation]['confidence'],
                self.knowledge_graph[entity2][relation]['confidence']
            )
            
            # Combine scores
            return 0.7 * similarity + 0.3 * relation_conf
        except Exception:
            return 0.0

    async def _update_entity_embeddings(self, entity: str):
        """Update embeddings for an entity."""
        try:
            # Get entity context
            context = self._get_entity_context(entity)
            
            # Generate new embedding
            embedding = await get_embedding(context)
            
            # Update embedding
            self.entity_embeddings[entity] = embedding
            
            logger.info(f"Successfully updated embeddings for entity {entity}")
        except Exception as e:
            logger.exception(f"Error updating entity embeddings: {str(e)}")
            raise

    def _get_entity_context(self, entity: str) -> str:
        """Get textual context for an entity."""
        knowledge = self.knowledge_graph.get(entity, {})
        return f"{entity} " + " ".join(
            f"{relation} {value['value']}"
            for relation, value in knowledge.items()
        )

    async def _integrate_new_knowledge(self, result: Dict[str, Any]):
        """Integrate new knowledge with existing knowledge."""
        try:
            # Extract new knowledge
            new_entities = await extract_entities(result)
            new_relations = await extract_relations(result)
            
            # Find potential entity matches
            for new_entity in new_entities:
                matches = await self._find_matching_entities(new_entity)
                
                # Merge or create entities
                if matches:
                    await self._merge_entities(new_entity, matches)
                else:
                    await self._create_new_entity(new_entity, new_relations)
            
            logger.info("Successfully integrated new knowledge")
        except Exception as e:
            logger.exception(f"Error integrating new knowledge: {str(e)}")
            raise

    async def _find_matching_entities(self, entity: str) -> List[Tuple[str, float]]:
        """Find existing entities that might match the new entity."""
        try:
            # Get embedding for new entity
            embedding = await get_embedding(entity)
            
            # Find similar entities in vector store
            similar_entities = await self.vector_store.search(
                embedding,
                limit=5,
                min_score=self.confidence_thresholds['entity_merge']
            )
            
            return [(hit.id, hit.score) for hit in similar_entities]
        except Exception as e:
            logger.exception(f"Error finding matching entities: {str(e)}")
            return []

    async def _merge_entities(self, new_entity: str, matches: List[Tuple[str, float]]):
        """Merge a new entity with existing entities."""
        try:
            # Sort matches by similarity score
            matches.sort(key=lambda x: x[1], reverse=True)
            best_match = matches[0][0]
            
            # Merge knowledge
            if best_match in self.knowledge_graph:
                for relation, value in self.knowledge_graph[best_match].items():
                    change = KnowledgeChange(
                        entity=new_entity,
                        relation=relation,
                        old_value=None,
                        new_value=value['value'],
                        timestamp=datetime.now(),
                        source="entity_merge",
                        confidence=value['confidence']
                    )
                    await self.record_change(change)
            
            logger.info(f"Successfully merged entity {new_entity} with {best_match}")
        except Exception as e:
            logger.exception(f"Error merging entities: {str(e)}")
            raise

    async def _create_new_entity(self, entity: str, relations: List[Dict[str, Any]]):
        """Create a new entity with its relations."""
        try:
            # Record initial knowledge
            for relation in relations:
                if relation['subject'] == entity:
                    change = KnowledgeChange(
                        entity=entity,
                        relation=relation['predicate'],
                        old_value=None,
                        new_value=relation['object'],
                        timestamp=datetime.now(),
                        source="new_entity",
                        confidence=relation.get('confidence', 1.0)
                    )
                    await self.record_change(change)
            
            logger.info(f"Successfully created new entity {entity}")
        except Exception as e:
            logger.exception(f"Error creating new entity: {str(e)}")
            raise

    def _validate_change(self, change: KnowledgeChange) -> bool:
        """Validate a knowledge change."""
        return (
            change.confidence >= self.confidence_thresholds['knowledge_update']
            and change.entity
            and change.relation
            and change.new_value is not None
        )

    def get_entity_history(self, entity: str) -> List[KnowledgeChange]:
        """Get the history of changes for an entity."""
        return [
            change for change in self.knowledge_changes
            if change.entity == entity
        ]

    def get_current_knowledge(self, entity: str) -> Dict[str, Any]:
        """Get current knowledge about an entity."""
        return self.knowledge_graph.get(entity, {})

    def analyze_knowledge_evolution(self) -> Dict[str, Any]:
        """Analyze how knowledge has evolved over time."""
        try:
            total_changes = len(self.knowledge_changes)
            unique_entities = len(set(change.entity for change in self.knowledge_changes))
            unique_relations = len(set(change.relation for change in self.knowledge_changes))
            
            # Calculate change frequency over time
            timestamps = [change.timestamp for change in self.knowledge_changes]
            if timestamps:
                time_range = max(timestamps) - min(timestamps)
                changes_per_day = total_changes / (time_range.days + 1)
            else:
                changes_per_day = 0
            
            # Analyze confidence trends
            confidences = [change.confidence for change in self.knowledge_changes]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                "total_changes": total_changes,
                "unique_entities": unique_entities,
                "unique_relations": unique_relations,
                "changes_per_day": changes_per_day,
                "average_confidence": avg_confidence,
                "last_update": max(timestamps) if timestamps else None
            }
        except Exception as e:
            logger.exception(f"Error analyzing knowledge evolution: {str(e)}")
            return {}
