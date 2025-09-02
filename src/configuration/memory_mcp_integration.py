"""
Memory MCP Integration for Configuration Management Pattern Storage
Stores configuration patterns, decisions, and learned optimizations
"""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigurationPatternType(Enum):
    """Types of configuration patterns"""
    HIERARCHY_STRUCTURE = "hierarchy_structure"
    CONFLICT_RESOLUTION = "conflict_resolution"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    SECURITY_HARDENING = "security_hardening"
    DEPLOYMENT_PATTERN = "deployment_pattern"
    SERVICE_INTEGRATION = "service_integration"
    CACHING_STRATEGY = "caching_strategy"

@dataclass
class ConfigurationPattern:
    """Represents a learned configuration pattern"""
    pattern_id: str
    pattern_type: ConfigurationPatternType
    name: str
    description: str
    pattern_data: Dict[str, Any]
    success_rate: float
    usage_count: int
    created_at: datetime
    last_used: datetime
    confidence_score: float
    tags: List[str]
    source_analysis: Optional[str] = None

@dataclass
class ConfigurationDecision:
    """Represents a configuration management decision"""
    decision_id: str
    decision_type: str
    context: Dict[str, Any]
    alternatives: List[Dict[str, Any]]
    chosen_alternative: Dict[str, Any]
    reasoning: str
    outcome: Optional[str] = None
    success: Optional[bool] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class PerformanceMetrics:
    """Configuration performance metrics"""
    metric_id: str
    config_pattern: str
    startup_time_ms: float
    memory_usage_mb: float
    config_load_time_ms: float
    cache_hit_rate: float
    error_rate: float
    throughput_rps: float
    measured_at: datetime

class ConfigurationMemoryManager:
    """Manages configuration patterns and decisions in memory"""
    
    def __init__(self, memory_db_path: str = "data/config_memory.db"):
        self.memory_db_path = Path(memory_db_path)
        self.memory_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._patterns: Dict[str, ConfigurationPattern] = {}
        self._decisions: Dict[str, ConfigurationDecision] = {}
        self._metrics: Dict[str, PerformanceMetrics] = {}
        
        # Learning parameters
        self._min_confidence_threshold = 0.7
        self._pattern_decay_days = 30
        self._success_rate_weight = 0.6
        self._usage_count_weight = 0.4
        
    async def initialize(self):
        """Initialize memory management system"""
        logger.info("Initializing Configuration Memory Manager")
        
        # Create database schema
        await self._create_schema()
        
        # Load existing patterns and decisions
        await self._load_from_storage()
        
        # Reference implementation: Connect to Memory MCP
        # await memory_mcp.initialize_category("config-management-patterns")
        
        logger.info(f"Memory manager initialized with {len(self._patterns)} patterns")
        
    async def store_configuration_pattern(self, pattern: ConfigurationPattern):
        """Store a configuration pattern with learning"""
        
        # Check if similar pattern already exists
        similar_pattern = await self._find_similar_pattern(pattern)
        
        if similar_pattern:
            # Update existing pattern
            await self._update_pattern(similar_pattern, pattern)
            logger.info(f"Updated similar pattern: {similar_pattern.pattern_id}")
        else:
            # Store new pattern
            self._patterns[pattern.pattern_id] = pattern
            await self._save_pattern_to_storage(pattern)
            logger.info(f"Stored new configuration pattern: {pattern.pattern_id}")
            
        # Reference implementation: Store in Memory MCP
        # await memory_mcp.store(
        #     f"config-patterns/{pattern.pattern_id}",
        #     asdict(pattern),
        #     category="config-management-patterns"
        # )
        
    async def store_configuration_decision(self, decision: ConfigurationDecision):
        """Store a configuration management decision"""
        
        self._decisions[decision.decision_id] = decision
        await self._save_decision_to_storage(decision)
        
        # Learn from decision outcome
        if decision.success is not None:
            await self._learn_from_decision(decision)
            
        logger.info(f"Stored configuration decision: {decision.decision_id}")
        
        # Reference implementation: Store in Memory MCP
        # await memory_mcp.store(
        #     f"config-decisions/{decision.decision_id}",
        #     asdict(decision),
        #     category="config-management-decisions"
        # )
        
    async def get_pattern_recommendations(self, 
                                        context: Dict[str, Any],
                                        pattern_type: Optional[ConfigurationPatternType] = None) -> List[ConfigurationPattern]:
        """Get pattern recommendations based on context"""
        
        recommendations = []
        
        for pattern in self._patterns.values():
            if pattern_type and pattern.pattern_type != pattern_type:
                continue
                
            # Calculate relevance score
            relevance_score = await self._calculate_pattern_relevance(pattern, context)
            
            if relevance_score > self._min_confidence_threshold:
                # Adjust confidence based on success rate and usage
                adjusted_confidence = (
                    pattern.confidence_score * 0.4 +
                    pattern.success_rate * self._success_rate_weight +
                    min(pattern.usage_count / 10, 1.0) * self._usage_count_weight
                )
                
                pattern.confidence_score = adjusted_confidence
                recommendations.append(pattern)
                
        # Sort by confidence score
        recommendations.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return recommendations[:5]  # Top 5 recommendations
        
    async def record_pattern_usage(self, pattern_id: str, success: bool, metrics: Optional[PerformanceMetrics] = None):
        """Record pattern usage and outcome"""
        
        if pattern_id not in self._patterns:
            logger.warning(f"Pattern not found: {pattern_id}")
            return
            
        pattern = self._patterns[pattern_id]
        pattern.usage_count += 1
        pattern.last_used = datetime.now()
        
        # Update success rate using exponential moving average
        alpha = 0.1  # Learning rate
        if success:
            pattern.success_rate = pattern.success_rate * (1 - alpha) + alpha
        else:
            pattern.success_rate = pattern.success_rate * (1 - alpha)
            
        # Store performance metrics if provided
        if metrics:
            self._metrics[metrics.metric_id] = metrics
            await self._save_metrics_to_storage(metrics)
            
        await self._save_pattern_to_storage(pattern)
        logger.info(f"Recorded usage for pattern {pattern_id}: success={success}")
        
    async def analyze_configuration_trends(self) -> Dict[str, Any]:
        """Analyze trends in configuration patterns and decisions"""
        
        now = datetime.now()
        last_30_days = now - timedelta(days=30)
        
        # Recent pattern usage
        recent_patterns = [p for p in self._patterns.values() 
                          if p.last_used and p.last_used > last_30_days]
        
        # Recent decisions
        recent_decisions = [d for d in self._decisions.values() 
                          if d.timestamp > last_30_days]
        
        # Success rates by pattern type
        success_by_type = {}
        for pattern_type in ConfigurationPatternType:
            type_patterns = [p for p in self._patterns.values() 
                           if p.pattern_type == pattern_type]
            if type_patterns:
                avg_success = sum(p.success_rate for p in type_patterns) / len(type_patterns)
                success_by_type[pattern_type.value] = avg_success
                
        # Most successful patterns
        top_patterns = sorted(self._patterns.values(), 
                            key=lambda x: x.success_rate * x.usage_count, 
                            reverse=True)[:10]
        
        # Performance trends
        performance_trends = {}
        if self._metrics:
            recent_metrics = [m for m in self._metrics.values() 
                            if m.measured_at > last_30_days]
            if recent_metrics:
                performance_trends = {
                    "avg_startup_time": sum(m.startup_time_ms for m in recent_metrics) / len(recent_metrics),
                    "avg_memory_usage": sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics),
                    "avg_cache_hit_rate": sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics)
                }
                
        return {
            "analysis_timestamp": now.isoformat(),
            "total_patterns": len(self._patterns),
            "recent_pattern_usage": len(recent_patterns),
            "recent_decisions": len(recent_decisions),
            "success_rates_by_type": success_by_type,
            "top_performing_patterns": [
                {"id": p.pattern_id, "name": p.name, "success_rate": p.success_rate}
                for p in top_patterns
            ],
            "performance_trends": performance_trends,
            "learning_insights": await self._generate_learning_insights()
        }
        
    async def optimize_configuration_hierarchy(self, current_hierarchy: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize configuration hierarchy based on learned patterns"""
        
        # Get hierarchy optimization patterns
        hierarchy_patterns = await self.get_pattern_recommendations(
            context={"hierarchy": current_hierarchy},
            pattern_type=ConfigurationPatternType.HIERARCHY_STRUCTURE
        )
        
        optimized_hierarchy = current_hierarchy.copy()
        
        for pattern in hierarchy_patterns:
            if pattern.confidence_score > 0.8:
                # Apply high-confidence optimizations
                optimizations = pattern.pattern_data.get("optimizations", [])
                
                for optimization in optimizations:
                    opt_type = optimization.get("type")
                    
                    if opt_type == "layer_reordering":
                        # Reorder hierarchy layers based on learned priorities
                        optimized_hierarchy = self._apply_layer_reordering(
                            optimized_hierarchy, optimization["config"]
                        )
                    elif opt_type == "consolidation":
                        # Apply learned consolidation patterns
                        optimized_hierarchy = self._apply_consolidation(
                            optimized_hierarchy, optimization["config"]
                        )
                    elif opt_type == "caching_strategy":
                        # Apply caching optimizations
                        optimized_hierarchy = self._apply_caching_strategy(
                            optimized_hierarchy, optimization["config"]
                        )
                        
        return optimized_hierarchy
        
    async def predict_configuration_conflicts(self, proposed_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predict potential conflicts based on historical patterns"""
        
        # Get conflict resolution patterns
        conflict_patterns = await self.get_pattern_recommendations(
            context={"proposed_config": proposed_config},
            pattern_type=ConfigurationPatternType.CONFLICT_RESOLUTION
        )
        
        potential_conflicts = []
        
        for pattern in conflict_patterns:
            conflict_indicators = pattern.pattern_data.get("conflict_indicators", [])
            
            for indicator in conflict_indicators:
                if self._matches_conflict_pattern(proposed_config, indicator):
                    potential_conflicts.append({
                        "type": indicator.get("conflict_type"),
                        "description": indicator.get("description"),
                        "confidence": pattern.confidence_score,
                        "suggested_resolution": indicator.get("resolution"),
                        "historical_success_rate": pattern.success_rate
                    })
                    
        return potential_conflicts
        
    async def learn_from_deployment_outcome(self, 
                                          deployment_config: Dict[str, Any],
                                          outcome_metrics: PerformanceMetrics,
                                          success: bool):
        """Learn from deployment outcomes to improve future recommendations"""
        
        # Create pattern from successful deployment
        if success and outcome_metrics.error_rate < 0.01:
            pattern = ConfigurationPattern(
                pattern_id=f"deploy_{hashlib.md5(str(deployment_config).encode()).hexdigest()[:8]}",
                pattern_type=ConfigurationPatternType.DEPLOYMENT_PATTERN,
                name=f"Successful Deployment Pattern {datetime.now().strftime('%Y%m%d')}",
                description="Learned from successful deployment",
                pattern_data={
                    "deployment_config": deployment_config,
                    "performance_characteristics": asdict(outcome_metrics)
                },
                success_rate=1.0,
                usage_count=1,
                created_at=datetime.now(),
                last_used=datetime.now(),
                confidence_score=0.9,
                tags=["deployment", "success", "learned"]
            )
            
            await self.store_configuration_pattern(pattern)
            
        # Record metrics
        await self.record_pattern_usage(
            f"deploy_pattern_{deployment_config.get('environment', 'unknown')}",
            success,
            outcome_metrics
        )
        
    async def _find_similar_pattern(self, pattern: ConfigurationPattern) -> Optional[ConfigurationPattern]:
        """Find similar existing pattern"""
        
        for existing_pattern in self._patterns.values():
            if existing_pattern.pattern_type == pattern.pattern_type:
                # Calculate similarity based on pattern data
                similarity = self._calculate_pattern_similarity(
                    existing_pattern.pattern_data, 
                    pattern.pattern_data
                )
                
                if similarity > 0.8:  # 80% similarity threshold
                    return existing_pattern
                    
        return None
        
    async def _update_pattern(self, existing: ConfigurationPattern, new_pattern: ConfigurationPattern):
        """Update existing pattern with new information"""
        
        # Merge pattern data
        merged_data = existing.pattern_data.copy()
        merged_data.update(new_pattern.pattern_data)
        
        existing.pattern_data = merged_data
        existing.usage_count += 1
        existing.last_used = datetime.now()
        
        # Update confidence using exponential moving average
        alpha = 0.2
        existing.confidence_score = (
            existing.confidence_score * (1 - alpha) + 
            new_pattern.confidence_score * alpha
        )
        
        await self._save_pattern_to_storage(existing)
        
    async def _calculate_pattern_relevance(self, pattern: ConfigurationPattern, context: Dict[str, Any]) -> float:
        """Calculate how relevant a pattern is to the current context"""
        
        relevance_score = 0.0
        
        # Tag matching
        context_tags = context.get("tags", [])
        if context_tags and pattern.tags:
            tag_overlap = len(set(context_tags) & set(pattern.tags))
            tag_relevance = tag_overlap / max(len(context_tags), len(pattern.tags))
            relevance_score += tag_relevance * 0.3
            
        # Pattern type matching
        if context.get("pattern_type") == pattern.pattern_type.value:
            relevance_score += 0.3
            
        # Configuration similarity
        context_config = context.get("config", {})
        if context_config and pattern.pattern_data:
            config_similarity = self._calculate_pattern_similarity(
                context_config, pattern.pattern_data
            )
            relevance_score += config_similarity * 0.4
            
        return min(relevance_score, 1.0)
        
    def _calculate_pattern_similarity(self, data1: Dict[str, Any], data2: Dict[str, Any]) -> float:
        """Calculate similarity between two pattern data structures"""
        
        # Simple Jaccard similarity on keys
        keys1 = set(self._flatten_dict(data1).keys())
        keys2 = set(self._flatten_dict(data2).keys())
        
        if not keys1 and not keys2:
            return 1.0
        if not keys1 or not keys2:
            return 0.0
            
        intersection = len(keys1 & keys2)
        union = len(keys1 | keys2)
        
        return intersection / union
        
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
        
    async def _learn_from_decision(self, decision: ConfigurationDecision):
        """Learn from decision outcomes"""
        
        if decision.success:
            # Extract successful decision pattern
            pattern_data = {
                "decision_context": decision.context,
                "successful_choice": decision.chosen_alternative,
                "reasoning": decision.reasoning
            }
            
            # Determine pattern type based on decision type
            pattern_type_map = {
                "hierarchy": ConfigurationPatternType.HIERARCHY_STRUCTURE,
                "security": ConfigurationPatternType.SECURITY_HARDENING,
                "performance": ConfigurationPatternType.PERFORMANCE_OPTIMIZATION,
                "caching": ConfigurationPatternType.CACHING_STRATEGY
            }
            
            pattern_type = pattern_type_map.get(
                decision.decision_type, 
                ConfigurationPatternType.SERVICE_INTEGRATION
            )
            
            pattern = ConfigurationPattern(
                pattern_id=f"decision_{decision.decision_id}",
                pattern_type=pattern_type,
                name=f"Learned from decision {decision.decision_type}",
                description=f"Pattern learned from successful {decision.decision_type} decision",
                pattern_data=pattern_data,
                success_rate=1.0,
                usage_count=1,
                created_at=datetime.now(),
                last_used=datetime.now(),
                confidence_score=0.8,
                tags=[decision.decision_type, "learned", "successful"]
            )
            
            await self.store_configuration_pattern(pattern)
            
    async def _generate_learning_insights(self) -> List[str]:
        """Generate insights from learned patterns"""
        
        insights = []
        
        # Most successful pattern types
        type_success = {}
        for pattern in self._patterns.values():
            if pattern.pattern_type not in type_success:
                type_success[pattern.pattern_type] = []
            type_success[pattern.pattern_type].append(pattern.success_rate)
            
        for pattern_type, success_rates in type_success.items():
            if success_rates:
                avg_success = sum(success_rates) / len(success_rates)
                if avg_success > 0.8:
                    insights.append(f"{pattern_type.value} patterns show high success rate ({avg_success:.1%})")
                elif avg_success < 0.5:
                    insights.append(f"{pattern_type.value} patterns need improvement ({avg_success:.1%} success)")
                    
        # Usage patterns
        high_usage_patterns = [p for p in self._patterns.values() if p.usage_count > 5]
        if high_usage_patterns:
            insights.append(f"{len(high_usage_patterns)} patterns are frequently reused (>5 times)")
            
        # Performance insights
        if self._metrics:
            recent_metrics = [m for m in self._metrics.values() 
                            if (datetime.now() - m.measured_at).days < 7]
            if recent_metrics:
                avg_startup = sum(m.startup_time_ms for m in recent_metrics) / len(recent_metrics)
                if avg_startup > 5000:
                    insights.append(f"Configuration load time is high ({avg_startup:.0f}ms)")
                elif avg_startup < 1000:
                    insights.append(f"Configuration load time is optimized ({avg_startup:.0f}ms)")
                    
        return insights
        
    def _matches_conflict_pattern(self, config: Dict[str, Any], indicator: Dict[str, Any]) -> bool:
        """Check if configuration matches a conflict pattern"""
        
        pattern_keys = indicator.get("keys", [])
        config_keys = set(self._flatten_dict(config).keys())
        
        # Simple pattern matching - check if indicator keys exist in config
        return any(key in config_keys for key in pattern_keys)
        
    def _apply_layer_reordering(self, hierarchy: Dict[str, Any], optimization: Dict[str, Any]) -> Dict[str, Any]:
        """Apply layer reordering optimization"""
        # Implementation would depend on specific hierarchy structure
        return hierarchy
        
    def _apply_consolidation(self, hierarchy: Dict[str, Any], optimization: Dict[str, Any]) -> Dict[str, Any]:
        """Apply consolidation optimization"""
        # Implementation would depend on specific hierarchy structure
        return hierarchy
        
    def _apply_caching_strategy(self, hierarchy: Dict[str, Any], optimization: Dict[str, Any]) -> Dict[str, Any]:
        """Apply caching strategy optimization"""
        # Implementation would depend on specific hierarchy structure
        return hierarchy
        
    async def _create_schema(self):
        """Create database schema for pattern storage"""
        
        async with self._get_db_connection() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS configuration_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    pattern_data TEXT NOT NULL,
                    success_rate REAL NOT NULL,
                    usage_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    last_used TEXT,
                    confidence_score REAL NOT NULL,
                    tags TEXT
                )
            ''')
            
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS configuration_decisions (
                    decision_id TEXT PRIMARY KEY,
                    decision_type TEXT NOT NULL,
                    context TEXT NOT NULL,
                    alternatives TEXT NOT NULL,
                    chosen_alternative TEXT NOT NULL,
                    reasoning TEXT,
                    outcome TEXT,
                    success INTEGER,
                    timestamp TEXT NOT NULL
                )
            ''')
            
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    metric_id TEXT PRIMARY KEY,
                    config_pattern TEXT NOT NULL,
                    startup_time_ms REAL NOT NULL,
                    memory_usage_mb REAL NOT NULL,
                    config_load_time_ms REAL NOT NULL,
                    cache_hit_rate REAL NOT NULL,
                    error_rate REAL NOT NULL,
                    throughput_rps REAL NOT NULL,
                    measured_at TEXT NOT NULL
                )
            ''')
            
            await conn.commit()
            
    async def _get_db_connection(self):
        """Get database connection (async wrapper for SQLite)"""
        # For simplicity, using synchronous SQLite
        # In production, would use aiosqlite
        return sqlite3.connect(self.memory_db_path)
        
    async def _load_from_storage(self):
        """Load patterns and decisions from storage"""
        # Implementation would load from SQLite database
        pass
        
    async def _save_pattern_to_storage(self, pattern: ConfigurationPattern):
        """Save pattern to storage"""
        # Implementation would save to SQLite database
        pass
        
    async def _save_decision_to_storage(self, decision: ConfigurationDecision):
        """Save decision to storage"""
        # Implementation would save to SQLite database
        pass
        
    async def _save_metrics_to_storage(self, metrics: PerformanceMetrics):
        """Save metrics to storage"""
        # Implementation would save to SQLite database
        pass

# Factory function
async def create_memory_manager(memory_db_path: str = "data/config_memory.db") -> ConfigurationMemoryManager:
    """Create and initialize configuration memory manager"""
    
    manager = ConfigurationMemoryManager(memory_db_path)
    await manager.initialize()
    return manager

if __name__ == "__main__":
    async def test_memory_manager():
        manager = await create_memory_manager()
        
        # Create test pattern
        pattern = ConfigurationPattern(
            pattern_id="test_hierarchy_001",
            pattern_type=ConfigurationPatternType.HIERARCHY_STRUCTURE,
            name="Standard 4-Layer Hierarchy",
            description="Base -> Environment -> Service -> Runtime hierarchy",
            pattern_data={
                "layers": ["base", "environment", "service", "runtime"],
                "priorities": [10, 30, 40, 50]
            },
            success_rate=0.85,
            usage_count=1,
            created_at=datetime.now(),
            last_used=datetime.now(),
            confidence_score=0.9,
            tags=["hierarchy", "standard", "production"]
        )
        
        await manager.store_configuration_pattern(pattern)
        
        # Get recommendations
        recommendations = await manager.get_pattern_recommendations(
            context={"tags": ["hierarchy"], "pattern_type": "hierarchy_structure"}
        )
        
        print(f"Found {len(recommendations)} recommendations")
        
        # Analyze trends
        trends = await manager.analyze_configuration_trends()
        print(f"Analysis: {trends}")
        
    asyncio.run(test_memory_manager())