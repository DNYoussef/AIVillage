"""
Creative Graph Search for Brainstorming and Innovation
Implements creative reasoning patterns for discovering novel connections and insights
"""

import asyncio
import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Set
import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)

class CreativityPattern(Enum):
    """Different patterns of creative thinking"""
    ANALOGICAL = "analogical"           # Find analogies between distant concepts
    COMBINATORIAL = "combinatorial"     # Combine unrelated concepts
    DIVERGENT = "divergent"            # Explore multiple directions
    ASSOCIATIVE = "associative"        # Follow loose associations
    METAPHORICAL = "metaphorical"      # Create metaphorical connections
    CONTRARIAN = "contrarian"          # Explore opposite/contrasting ideas

@dataclass
class CreativeConnection:
    """A creative connection between concepts"""
    source_concept: str
    target_concept: str
    connection_type: CreativityPattern
    novelty_score: float
    confidence: float
    explanation: str
    supporting_evidence: List[str] = field(default_factory=list)
    
@dataclass
class CreativeInsight:
    """Creative insight generated from graph exploration"""
    insight_text: str
    creativity_score: float
    novelty_score: float
    connections: List[CreativeConnection]
    concepts_involved: List[str]
    pattern_type: CreativityPattern

@dataclass
class BrainstormingSession:
    """Session for creative brainstorming"""
    session_id: str
    prompt: str
    generated_insights: List[CreativeInsight] = field(default_factory=list)
    explored_concepts: Set[str] = field(default_factory=set)
    connection_map: Dict[str, List[str]] = field(default_factory=dict)

class CreativeGraphSearch:
    """
    Creative search engine for brainstorming and innovative thinking
    Uses graph traversal patterns that mimic creative cognitive processes
    """
    
    def __init__(self, knowledge_graph: nx.Graph):
        self.knowledge_graph = knowledge_graph
        self.creativity_weights = {
            CreativityPattern.ANALOGICAL: 0.9,
            CreativityPattern.COMBINATORIAL: 0.8,
            CreativityPattern.DIVERGENT: 0.7,
            CreativityPattern.ASSOCIATIVE: 0.6,
            CreativityPattern.METAPHORICAL: 0.85,
            CreativityPattern.CONTRARIAN: 0.75
        }
        
        # Parameters for creative exploration
        self.randomness_factor = 0.3      # How much randomness in exploration
        self.novelty_threshold = 0.4      # Minimum novelty for insights
        self.max_connection_distance = 5   # Maximum hops for connections
        self.diversity_penalty = 0.1      # Penalty for similar insights
    
    async def creative_brainstorm(self,
                                prompt: str,
                                num_insights: int = 10,
                                creativity_patterns: Optional[List[CreativityPattern]] = None,
                                exploration_depth: int = 3) -> BrainstormingSession:
        """
        Generate creative insights through graph-based brainstorming
        
        Args:
            prompt: Initial brainstorming prompt
            num_insights: Number of insights to generate
            creativity_patterns: Specific patterns to use
            exploration_depth: How deep to explore connections
            
        Returns:
            BrainstormingSession with generated insights
        """
        session = BrainstormingSession(
            session_id=f"session_{random.randint(1000, 9999)}",
            prompt=prompt
        )
        
        # Use all patterns if none specified
        if creativity_patterns is None:
            creativity_patterns = list(CreativityPattern)
        
        # Find seed concepts from prompt
        seed_concepts = await self._extract_concepts_from_prompt(prompt)
        session.explored_concepts.update(seed_concepts)
        
        # Generate insights using different creativity patterns
        for pattern in creativity_patterns:
            pattern_insights = await self._generate_pattern_insights(
                seed_concepts, pattern, exploration_depth
            )
            session.generated_insights.extend(pattern_insights)
        
        # Rank and filter insights
        session.generated_insights = await self._rank_creative_insights(
            session.generated_insights, num_insights
        )
        
        return session
    
    async def _extract_concepts_from_prompt(self, prompt: str) -> List[str]:
        """Extract key concepts from brainstorming prompt"""
        # Simple keyword extraction (can be enhanced with NLP)
        prompt_lower = prompt.lower()
        seed_concepts = []
        
        # Find nodes that match words in the prompt
        for node_id in self.knowledge_graph.nodes():
            node_data = self.knowledge_graph.nodes[node_id]
            content = node_data.get('content', '').lower()
            
            # Check if prompt words appear in node content
            prompt_words = prompt_lower.split()
            for word in prompt_words:
                if len(word) > 3 and word in content:
                    seed_concepts.append(node_id)
                    break
        
        # If no direct matches, pick random starting points
        if not seed_concepts:
            all_nodes = list(self.knowledge_graph.nodes())
            seed_concepts = random.sample(all_nodes, min(3, len(all_nodes)))
        
        return seed_concepts[:5]  # Limit initial concepts
    
    async def _generate_pattern_insights(self,
                                       seed_concepts: List[str],
                                       pattern: CreativityPattern,
                                       depth: int) -> List[CreativeInsight]:
        """Generate insights using a specific creativity pattern"""
        insights = []
        
        if pattern == CreativityPattern.ANALOGICAL:
            insights = await self._analogical_reasoning(seed_concepts, depth)
        elif pattern == CreativityPattern.COMBINATORIAL:
            insights = await self._combinatorial_exploration(seed_concepts, depth)
        elif pattern == CreativityPattern.DIVERGENT:
            insights = await self._divergent_exploration(seed_concepts, depth)
        elif pattern == CreativityPattern.ASSOCIATIVE:
            insights = await self._associative_exploration(seed_concepts, depth)
        elif pattern == CreativityPattern.METAPHORICAL:
            insights = await self._metaphorical_connections(seed_concepts, depth)
        elif pattern == CreativityPattern.CONTRARIAN:
            insights = await self._contrarian_exploration(seed_concepts, depth)
        
        return insights
    
    async def _analogical_reasoning(self, seed_concepts: List[str], depth: int) -> List[CreativeInsight]:
        """Find analogical connections between distant concepts"""
        insights = []
        
        for seed in seed_concepts:
            # Find concepts at medium distance (not too close, not too far)
            distant_concepts = []
            
            for node in self.knowledge_graph.nodes():
                if node != seed and self.knowledge_graph.has_node(node):
                    try:
                        distance = nx.shortest_path_length(self.knowledge_graph, seed, node)
                        if 2 <= distance <= 4:  # Sweet spot for analogies
                            distant_concepts.append((node, distance))
                    except nx.NetworkXNoPath:
                        continue
            
            # Create analogical insights
            for target, distance in distant_concepts[:5]:
                # Find structural similarities
                seed_neighbors = set(self.knowledge_graph.neighbors(seed))
                target_neighbors = set(self.knowledge_graph.neighbors(target))
                
                # If they have similar connection patterns, it's a good analogy
                if len(seed_neighbors & target_neighbors) > 0:
                    novelty_score = 0.8 + (distance * 0.05)  # More distant = more novel
                    creativity_score = 0.7 + random.uniform(0.0, 0.2)
                    
                    connection = CreativeConnection(
                        source_concept=seed,
                        target_concept=target,
                        connection_type=CreativityPattern.ANALOGICAL,
                        novelty_score=novelty_score,
                        confidence=0.6,
                        explanation=f"Analogical connection: {seed} is like {target} in structural patterns"
                    )
                    
                    insight = CreativeInsight(
                        insight_text=f"Consider how {seed} might work like {target} - they share similar relationship patterns",
                        creativity_score=creativity_score,
                        novelty_score=novelty_score,
                        connections=[connection],
                        concepts_involved=[seed, target],
                        pattern_type=CreativityPattern.ANALOGICAL
                    )
                    insights.append(insight)
        
        return insights[:3]  # Limit insights per pattern
    
    async def _combinatorial_exploration(self, seed_concepts: List[str], depth: int) -> List[CreativeInsight]:
        """Combine unrelated concepts for creative synthesis"""
        insights = []
        
        # Create combinations of seed concepts with distant concepts
        for seed in seed_concepts:
            # Find unrelated concepts (no direct connection)
            unrelated_concepts = []
            
            for node in self.knowledge_graph.nodes():
                if (node != seed and 
                    not self.knowledge_graph.has_edge(seed, node) and
                    not self.knowledge_graph.has_edge(node, seed)):
                    unrelated_concepts.append(node)
            
            # Create combinatorial insights
            for target in random.sample(unrelated_concepts, min(3, len(unrelated_concepts))):
                novelty_score = 0.9  # High novelty for unrelated combinations
                creativity_score = 0.8 + random.uniform(0.0, 0.15)
                
                connection = CreativeConnection(
                    source_concept=seed,
                    target_concept=target,
                    connection_type=CreativityPattern.COMBINATORIAL,
                    novelty_score=novelty_score,
                    confidence=0.5,  # Lower confidence for bold combinations
                    explanation=f"Creative synthesis: combining {seed} with {target}"
                )
                
                insight = CreativeInsight(
                    insight_text=f"What if we combined {seed} with {target}? This unexpected pairing might reveal new possibilities",
                    creativity_score=creativity_score,
                    novelty_score=novelty_score,
                    connections=[connection],
                    concepts_involved=[seed, target],
                    pattern_type=CreativityPattern.COMBINATORIAL
                )
                insights.append(insight)
        
        return insights[:3]
    
    async def _divergent_exploration(self, seed_concepts: List[str], depth: int) -> List[CreativeInsight]:
        """Explore multiple divergent paths from each concept"""
        insights = []
        
        for seed in seed_concepts:
            # Get multiple neighbor paths
            all_paths = []
            
            # Do random walks from seed concept
            for _ in range(5):  # 5 different explorations
                current = seed
                path = [current]
                
                for step in range(depth):
                    neighbors = list(self.knowledge_graph.neighbors(current))
                    if neighbors:
                        # Add randomness to exploration
                        if random.random() < self.randomness_factor:
                            next_node = random.choice(neighbors)
                        else:
                            # Choose based on some criteria (could be enhanced)
                            next_node = neighbors[0]
                        
                        path.append(next_node)
                        current = next_node
                    else:
                        break
                
                if len(path) > 1:
                    all_paths.append(path)
            
            # Create insights from divergent paths
            for path in all_paths:
                if len(path) >= 3:
                    novelty_score = 0.6 + (len(path) * 0.05)
                    creativity_score = 0.65 + random.uniform(0.0, 0.2)
                    
                    connections = []
                    for i in range(len(path) - 1):
                        connections.append(CreativeConnection(
                            source_concept=path[i],
                            target_concept=path[i + 1],
                            connection_type=CreativityPattern.DIVERGENT,
                            novelty_score=novelty_score,
                            confidence=0.6,
                            explanation=f"Divergent path step: {path[i]} -> {path[i + 1]}"
                        ))
                    
                    insight = CreativeInsight(
                        insight_text=f"Following the path {' -> '.join(path[:4])} reveals unexpected connections",
                        creativity_score=creativity_score,
                        novelty_score=novelty_score,
                        connections=connections,
                        concepts_involved=path,
                        pattern_type=CreativityPattern.DIVERGENT
                    )
                    insights.append(insight)
        
        return insights[:4]
    
    async def _associative_exploration(self, seed_concepts: List[str], depth: int) -> List[CreativeInsight]:
        """Follow loose associative connections"""
        insights = []
        
        for seed in seed_concepts:
            # Follow loose associations (weak connections)
            weak_connections = []
            
            # Find nodes with indirect connections
            for neighbor in self.knowledge_graph.neighbors(seed):
                for second_neighbor in self.knowledge_graph.neighbors(neighbor):
                    if second_neighbor != seed and second_neighbor not in self.knowledge_graph.neighbors(seed):
                        weak_connections.append((neighbor, second_neighbor))
            
            # Create associative insights
            for intermediate, target in weak_connections[:3]:
                novelty_score = 0.55 + random.uniform(0.0, 0.15)
                creativity_score = 0.6 + random.uniform(0.0, 0.15)
                
                connection = CreativeConnection(
                    source_concept=seed,
                    target_concept=target,
                    connection_type=CreativityPattern.ASSOCIATIVE,
                    novelty_score=novelty_score,
                    confidence=0.4,  # Lower confidence for loose associations
                    explanation=f"Associative link: {seed} connects to {target} through {intermediate}"
                )
                
                insight = CreativeInsight(
                    insight_text=f"There's an interesting association between {seed} and {target} (through {intermediate})",
                    creativity_score=creativity_score,
                    novelty_score=novelty_score,
                    connections=[connection],
                    concepts_involved=[seed, intermediate, target],
                    pattern_type=CreativityPattern.ASSOCIATIVE
                )
                insights.append(insight)
        
        return insights[:3]
    
    async def _metaphorical_connections(self, seed_concepts: List[str], depth: int) -> List[CreativeInsight]:
        """Create metaphorical connections between concepts"""
        insights = []
        
        # This would be enhanced with semantic analysis
        # For now, we create metaphorical connections based on structural patterns
        
        for seed in seed_concepts:
            # Find concepts with similar structural properties
            seed_degree = self.knowledge_graph.degree(seed)
            
            metaphor_candidates = []
            for node in self.knowledge_graph.nodes():
                if node != seed:
                    node_degree = self.knowledge_graph.degree(node)
                    # Similar degree = potentially good metaphor
                    if abs(seed_degree - node_degree) <= 2:
                        try:
                            distance = nx.shortest_path_length(self.knowledge_graph, seed, node)
                            if distance >= 3:  # Distant enough to be metaphorical
                                metaphor_candidates.append((node, distance))
                        except nx.NetworkXNoPath:
                            pass
            
            # Create metaphorical insights
            for target, distance in metaphor_candidates[:2]:
                novelty_score = 0.75 + (distance * 0.03)
                creativity_score = 0.8 + random.uniform(0.0, 0.15)
                
                connection = CreativeConnection(
                    source_concept=seed,
                    target_concept=target,
                    connection_type=CreativityPattern.METAPHORICAL,
                    novelty_score=novelty_score,
                    confidence=0.5,
                    explanation=f"Metaphorical connection: {seed} as a metaphor for {target}"
                )
                
                insight = CreativeInsight(
                    insight_text=f"Think of {seed} as a metaphor for {target} - what new perspectives does this reveal?",
                    creativity_score=creativity_score,
                    novelty_score=novelty_score,
                    connections=[connection],
                    concepts_involved=[seed, target],
                    pattern_type=CreativityPattern.METAPHORICAL
                )
                insights.append(insight)
        
        return insights[:2]
    
    async def _contrarian_exploration(self, seed_concepts: List[str], depth: int) -> List[CreativeInsight]:
        """Explore contrarian or opposite perspectives"""
        insights = []
        
        # This would be enhanced with semantic opposition detection
        # For now, we simulate by finding concepts with minimal connections
        
        for seed in seed_concepts:
            # Find concepts that are least connected to seed (potentially contrarian)
            contrarian_candidates = []
            
            for node in self.knowledge_graph.nodes():
                if node != seed:
                    try:
                        distance = nx.shortest_path_length(self.knowledge_graph, seed, node)
                        if distance >= 4:  # Very distant = potentially contrarian
                            contrarian_candidates.append(node)
                    except nx.NetworkXNoPath:
                        # No path = maximally contrarian
                        contrarian_candidates.append(node)
            
            # Create contrarian insights
            for target in random.sample(contrarian_candidates, min(2, len(contrarian_candidates))):
                novelty_score = 0.85  # High novelty for contrarian views
                creativity_score = 0.75 + random.uniform(0.0, 0.2)
                
                connection = CreativeConnection(
                    source_concept=seed,
                    target_concept=target,
                    connection_type=CreativityPattern.CONTRARIAN,
                    novelty_score=novelty_score,
                    confidence=0.45,  # Lower confidence for contrarian ideas
                    explanation=f"Contrarian perspective: {seed} vs {target}"
                )
                
                insight = CreativeInsight(
                    insight_text=f"What if we challenged {seed} by considering {target} instead? Sometimes opposite views spark innovation",
                    creativity_score=creativity_score,
                    novelty_score=novelty_score,
                    connections=[connection],
                    concepts_involved=[seed, target],
                    pattern_type=CreativityPattern.CONTRARIAN
                )
                insights.append(insight)
        
        return insights[:2]
    
    async def _rank_creative_insights(self, insights: List[CreativeInsight], max_insights: int) -> List[CreativeInsight]:
        """Rank and filter creative insights by novelty and creativity"""
        
        # Apply diversity penalty for similar insights
        for i, insight1 in enumerate(insights):
            for j, insight2 in enumerate(insights[i+1:], i+1):
                # Check overlap in concepts
                overlap = len(set(insight1.concepts_involved) & set(insight2.concepts_involved))
                if overlap > 0:
                    # Apply penalty to lower-scoring insight
                    if insight1.creativity_score > insight2.creativity_score:
                        insight2.creativity_score -= (overlap * self.diversity_penalty)
                    else:
                        insight1.creativity_score -= (overlap * self.diversity_penalty)
        
        # Filter by novelty threshold
        filtered_insights = [i for i in insights if i.novelty_score >= self.novelty_threshold]
        
        # Sort by combined creativity and novelty score
        filtered_insights.sort(
            key=lambda x: (x.creativity_score + x.novelty_score) / 2,
            reverse=True
        )
        
        return filtered_insights[:max_insights]
    
    def get_creativity_stats(self) -> Dict[str, Any]:
        """Get statistics about creative exploration capabilities"""
        return {
            'graph_size': self.knowledge_graph.number_of_nodes(),
            'graph_edges': self.knowledge_graph.number_of_edges(),
            'creativity_patterns': len(CreativityPattern),
            'randomness_factor': self.randomness_factor,
            'novelty_threshold': self.novelty_threshold,
            'max_connection_distance': self.max_connection_distance
        }