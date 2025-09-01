#!/usr/bin/env python3
"""DSPy Integration Module for Agent Prompt Optimization.

This module integrates DSPy (Declarative Self-improving Python) for automatic
prompt optimization based on agent performance patterns and outcomes.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import sqlite3

logger = logging.getLogger(__name__)


class DSPyAgentOptimizer:
    """DSPy integration for automatic agent prompt optimization."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.optimization_db_path = self.project_root / ".mcp" / "dspy_optimization.db"
        self.optimization_db_path.parent.mkdir(exist_ok=True)
        self._initialize_optimization_db()
        
    def _initialize_optimization_db(self):
        """Initialize the DSPy optimization database."""
        conn = sqlite3.connect(str(self.optimization_db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prompt_optimization (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_type TEXT NOT NULL,
                original_prompt TEXT NOT NULL,
                optimized_prompt TEXT,
                performance_metrics TEXT NOT NULL,
                optimization_iteration INTEGER DEFAULT 1,
                success_rate REAL DEFAULT 0.0,
                avg_completion_time REAL DEFAULT 0.0,
                quality_score REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_type TEXT NOT NULL,
                session_id TEXT NOT NULL,
                task_description TEXT NOT NULL,
                completion_status TEXT NOT NULL,
                completion_time REAL,
                quality_score REAL,
                error_count INTEGER DEFAULT 0,
                memory_usage REAL,
                reasoning_quality REAL,
                output_quality REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """)
        conn.commit()
        conn.close()

    def record_agent_performance(self, 
                                agent_type: str,
                                session_id: str,
                                task_description: str,
                                performance_data: Dict[str, Any]) -> str:
        """Record agent performance for DSPy optimization learning."""
        
        # Extract performance metrics
        completion_status = performance_data.get('completion_status', 'unknown')
        completion_time = performance_data.get('completion_time', 0.0)
        quality_score = performance_data.get('quality_score', 0.0)
        error_count = performance_data.get('error_count', 0)
        memory_usage = performance_data.get('memory_usage', 0.0)
        reasoning_quality = performance_data.get('reasoning_quality', 0.0)
        output_quality = performance_data.get('output_quality', 0.0)
        
        conn = sqlite3.connect(str(self.optimization_db_path))
        cursor = conn.execute("""
            INSERT INTO agent_performance 
            (agent_type, session_id, task_description, completion_status, 
             completion_time, quality_score, error_count, memory_usage, 
             reasoning_quality, output_quality, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            agent_type, session_id, task_description, completion_status,
            completion_time, quality_score, error_count, memory_usage,
            reasoning_quality, output_quality, json.dumps(performance_data)
        ))
        
        performance_id = str(cursor.lastrowid)
        conn.commit()
        conn.close()
        
        # Trigger optimization if we have enough data
        self._check_optimization_trigger(agent_type)
        
        return performance_id

    def _check_optimization_trigger(self, agent_type: str):
        """Check if we should trigger prompt optimization for an agent type."""
        conn = sqlite3.connect(str(self.optimization_db_path))
        cursor = conn.execute("""
            SELECT COUNT(*), AVG(quality_score), AVG(completion_time)
            FROM agent_performance 
            WHERE agent_type = ? AND created_at > datetime('now', '-7 days')
        """, (agent_type,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0] >= 5:  # Minimum 5 performance records
            performance_count, avg_quality, avg_time = result
            
            # Trigger optimization if quality is below threshold or time is high
            if avg_quality < 0.7 or avg_time > 300:  # 5 minutes
                self._optimize_agent_prompts(agent_type)

    def _optimize_agent_prompts(self, agent_type: str):
        """Optimize prompts for an agent type using DSPy patterns."""
        
        # Get recent performance data
        conn = sqlite3.connect(str(self.optimization_db_path))
        cursor = conn.execute("""
            SELECT task_description, completion_status, quality_score, 
                   reasoning_quality, output_quality, metadata
            FROM agent_performance 
            WHERE agent_type = ? AND created_at > datetime('now', '-7 days')
            ORDER BY quality_score DESC
        """, (agent_type,))
        
        performances = cursor.fetchall()
        conn.close()
        
        if not performances:
            return
        
        # Analyze patterns in successful vs failed tasks
        successful_tasks = [p for p in performances if p[1] == 'completed' and p[2] > 0.7]
        failed_tasks = [p for p in performances if p[1] != 'completed' or p[2] <= 0.5]
        
        optimization_insights = self._generate_optimization_insights(
            agent_type, successful_tasks, failed_tasks
        )
        
        # Generate optimized prompt
        optimized_prompt = self._generate_optimized_prompt(agent_type, optimization_insights)
        
        # Store optimization
        self._store_prompt_optimization(agent_type, optimized_prompt, optimization_insights)

    def _generate_optimization_insights(self, 
                                      agent_type: str, 
                                      successful_tasks: List[Tuple],
                                      failed_tasks: List[Tuple]) -> Dict[str, Any]:
        """Generate insights from performance patterns."""
        
        insights = {
            "agent_type": agent_type,
            "optimization_timestamp": datetime.now().isoformat(),
            "successful_patterns": [],
            "failure_patterns": [],
            "recommendations": []
        }
        
        # Analyze successful task patterns
        if successful_tasks:
            avg_quality = sum(task[2] for task in successful_tasks) / len(successful_tasks)
            avg_reasoning = sum(task[3] for task in successful_tasks) / len(successful_tasks)
            
            insights["successful_patterns"] = {
                "count": len(successful_tasks),
                "avg_quality": avg_quality,
                "avg_reasoning": avg_reasoning,
                "common_task_types": self._extract_common_patterns([task[0] for task in successful_tasks])
            }
        
        # Analyze failure patterns
        if failed_tasks:
            insights["failure_patterns"] = {
                "count": len(failed_tasks),
                "common_issues": self._extract_failure_patterns(failed_tasks),
                "avg_quality": sum(task[2] for task in failed_tasks) / len(failed_tasks)
            }
        
        # Generate recommendations
        insights["recommendations"] = self._generate_recommendations(insights)
        
        return insights

    def _extract_common_patterns(self, task_descriptions: List[str]) -> List[str]:
        """Extract common patterns from successful task descriptions."""
        # Simple keyword extraction - in practice, this would use NLP
        common_words = {}
        
        for desc in task_descriptions:
            words = desc.lower().split()
            for word in words:
                if len(word) > 4:  # Skip short words
                    common_words[word] = common_words.get(word, 0) + 1
        
        # Return most common words
        return sorted(common_words.keys(), key=lambda w: common_words[w], reverse=True)[:5]

    def _extract_failure_patterns(self, failed_tasks: List[Tuple]) -> List[str]:
        """Extract patterns from failed tasks."""
        failure_reasons = []
        
        for task in failed_tasks:
            metadata = json.loads(task[5]) if task[5] else {}
            errors = metadata.get('errors', [])
            failure_reasons.extend(errors)
        
        return list(set(failure_reasons))

    def _generate_recommendations(self, insights: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on insights."""
        recommendations = []
        
        successful_patterns = insights.get("successful_patterns", {})
        failure_patterns = insights.get("failure_patterns", {})
        
        if successful_patterns.get("avg_reasoning", 0) > 0.8:
            recommendations.append("Emphasize step-by-step reasoning in prompts")
        
        if failure_patterns.get("count", 0) > successful_patterns.get("count", 0):
            recommendations.append("Add more specific task guidance and examples")
        
        if "timeout" in str(failure_patterns.get("common_issues", [])):
            recommendations.append("Break down complex tasks into smaller steps")
        
        if successful_patterns.get("avg_quality", 0) > 0.85:
            recommendations.append("Replicate successful task structuring patterns")
        
        return recommendations

    def _generate_optimized_prompt(self, agent_type: str, insights: Dict[str, Any]) -> str:
        """Generate an optimized prompt based on insights."""
        
        base_prompt = f"""
OPTIMIZED {agent_type.upper()} AGENT PROMPT (DSPy Enhanced)

Based on performance analysis of {insights.get('successful_patterns', {}).get('count', 0)} successful tasks
and {insights.get('failure_patterns', {}).get('count', 0)} failed tasks.

OPTIMIZATION INSIGHTS:
{json.dumps(insights.get('recommendations', []), indent=2)}

ENHANCED INSTRUCTIONS:
"""
        
        # Add specific optimizations based on agent type
        if agent_type == "researcher":
            base_prompt += """
1. ALWAYS start with a clear research question breakdown
2. Use systematic information gathering from multiple sources
3. Provide structured summaries with confidence scores
4. Include source citations and reliability assessments
"""
        elif agent_type == "coder":
            base_prompt += """
1. ALWAYS analyze requirements before coding
2. Break down implementation into logical components
3. Include error handling and edge cases
4. Provide clear documentation and comments
"""
        elif agent_type == "tester":
            base_prompt += """
1. ALWAYS create comprehensive test plans before implementation
2. Include unit, integration, and edge case tests
3. Provide clear test documentation and coverage reports
4. Include performance and security test considerations
"""
        elif agent_type == "reviewer":
            base_prompt += """
1. ALWAYS use systematic review checklists
2. Focus on code quality, security, and maintainability
3. Provide constructive feedback with specific examples
4. Include recommendations for improvement
"""
        
        # Add common optimization patterns
        base_prompt += f"""

DSPY OPTIMIZATION PATTERNS:
- Success Rate Target: >85% (Current: {insights.get('successful_patterns', {}).get('count', 0) / (insights.get('successful_patterns', {}).get('count', 0) + insights.get('failure_patterns', {}).get('count', 0)) * 100:.1f}%)
- Quality Score Target: >0.8 (Current Avg: {insights.get('successful_patterns', {}).get('avg_quality', 0):.2f})
- Reasoning Quality Target: >0.8 (Current Avg: {insights.get('successful_patterns', {}).get('avg_reasoning', 0):.2f})

MEMORY INTEGRATION:
- Store intermediate results with structured keys
- Check memory for context and previous agent work
- Update shared coordination state regularly

SEQUENTIAL THINKING:
- Use step-by-step reasoning for complex tasks
- Document decision-making process
- Validate each step before proceeding

This prompt has been automatically optimized based on performance data.
Version: DSPy_Optimized_v{datetime.now().strftime('%Y%m%d_%H%M%S')}
"""
        
        return base_prompt

    def _store_prompt_optimization(self, 
                                 agent_type: str, 
                                 optimized_prompt: str, 
                                 insights: Dict[str, Any]):
        """Store the optimized prompt for future use."""
        
        # Calculate performance metrics
        successful_count = insights.get('successful_patterns', {}).get('count', 0)
        failed_count = insights.get('failure_patterns', {}).get('count', 0)
        total_tasks = successful_count + failed_count
        
        success_rate = successful_count / total_tasks if total_tasks > 0 else 0.0
        quality_score = insights.get('successful_patterns', {}).get('avg_quality', 0.0)
        
        conn = sqlite3.connect(str(self.optimization_db_path))
        
        # Get current iteration count
        cursor = conn.execute("""
            SELECT MAX(optimization_iteration) FROM prompt_optimization 
            WHERE agent_type = ?
        """, (agent_type,))
        
        result = cursor.fetchone()
        iteration = (result[0] or 0) + 1
        
        # Store new optimization
        conn.execute("""
            INSERT INTO prompt_optimization 
            (agent_type, original_prompt, optimized_prompt, performance_metrics, 
             optimization_iteration, success_rate, quality_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            agent_type,
            "",  # We don't store original prompts for now
            optimized_prompt,
            json.dumps(insights),
            iteration,
            success_rate,
            quality_score
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Stored DSPy optimization for {agent_type} - Iteration {iteration}, Success Rate: {success_rate:.2f}")

    def get_optimized_prompt(self, agent_type: str) -> Optional[str]:
        """Get the latest optimized prompt for an agent type."""
        
        conn = sqlite3.connect(str(self.optimization_db_path))
        cursor = conn.execute("""
            SELECT optimized_prompt, success_rate, quality_score, optimization_iteration
            FROM prompt_optimization 
            WHERE agent_type = ? 
            ORDER BY optimization_iteration DESC 
            LIMIT 1
        """, (agent_type,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            logger.info(f"Retrieved optimized prompt for {agent_type} - Iteration {result[3]}, Success Rate: {result[1]:.2f}")
            return result[0]
        
        return None

    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status for all agent types."""
        
        conn = sqlite3.connect(str(self.optimization_db_path))
        cursor = conn.execute("""
            SELECT agent_type, COUNT(*) as optimization_count, 
                   MAX(success_rate) as best_success_rate,
                   MAX(quality_score) as best_quality_score,
                   MAX(optimization_iteration) as latest_iteration
            FROM prompt_optimization 
            GROUP BY agent_type
        """)
        
        optimization_status = {}
        for row in cursor.fetchall():
            optimization_status[row[0]] = {
                "optimization_count": row[1],
                "best_success_rate": row[2],
                "best_quality_score": row[3],
                "latest_iteration": row[4]
            }
        
        # Get performance data
        cursor = conn.execute("""
            SELECT agent_type, COUNT(*) as total_tasks,
                   AVG(quality_score) as avg_quality,
                   AVG(completion_time) as avg_time
            FROM agent_performance
            WHERE created_at > datetime('now', '-7 days')
            GROUP BY agent_type
        """)
        
        for row in cursor.fetchall():
            agent_type = row[0]
            if agent_type not in optimization_status:
                optimization_status[agent_type] = {}
            
            optimization_status[agent_type].update({
                "recent_tasks": row[1],
                "avg_quality": row[2],
                "avg_completion_time": row[3]
            })
        
        conn.close()
        
        return {
            "optimization_db_path": str(self.optimization_db_path),
            "agent_optimizations": optimization_status,
            "last_updated": datetime.now().isoformat()
        }


if __name__ == "__main__":
    # Example usage
    optimizer = DSPyAgentOptimizer()
    
    # Example performance recording
    optimizer.record_agent_performance(
        agent_type="researcher",
        session_id="test_session_001",
        task_description="Research microservices architecture patterns",
        performance_data={
            "completion_status": "completed",
            "completion_time": 120.0,
            "quality_score": 0.85,
            "reasoning_quality": 0.9,
            "output_quality": 0.8,
            "error_count": 0,
            "memory_usage": 45.2
        }
    )
    
    print("DSPy Optimization Status:")
    print(json.dumps(optimizer.get_optimization_status(), indent=2))