#!/usr/bin/env python3
"""Sequential Thinking MCP Wrapper for Enhanced Agent Reasoning.

This module provides integration with the Sequential Thinking MCP server
to enable multi-step reasoning and intelligent planning for agents.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import subprocess
import asyncio

logger = logging.getLogger(__name__)


class SequentialThinkingWrapper:
    """Wrapper for Sequential Thinking MCP integration with agents."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.mcp_config_path = self.project_root / "config" / "mcp_config.json"
        self.reasoning_cache = {}
        self.active_reasoning_sessions = {}
        
    def create_reasoning_chain(self, 
                             agent_type: str,
                             task_description: str,
                             complexity_level: str = "medium",
                             context: Optional[Dict[str, Any]] = None) -> str:
        """Create a reasoning chain for an agent task."""
        
        # Define reasoning patterns based on agent type and complexity
        reasoning_patterns = {
            "researcher": {
                "low": ["Identify Topic", "Search Sources", "Extract Information", "Summarize"],
                "medium": ["Analyze Requirements", "Research Multiple Sources", "Cross-Reference", "Synthesize", "Validate"],
                "high": ["Decompose Question", "Strategic Source Planning", "Deep Research", "Critical Analysis", "Synthesis", "Peer Review", "Final Validation"]
            },
            "coder": {
                "low": ["Understand Task", "Plan Implementation", "Code", "Test"],
                "medium": ["Analyze Requirements", "Design Architecture", "Implement Components", "Unit Testing", "Integration", "Validation"],
                "high": ["Requirements Analysis", "System Design", "Technology Selection", "Implementation Planning", "Development", "Testing Strategy", "Code Review", "Integration Testing", "Performance Optimization"]
            },
            "system-architect": {
                "low": ["Understand Requirements", "Design System", "Document"],
                "medium": ["Requirements Analysis", "System Design", "Component Planning", "Interface Design", "Documentation", "Validation"],
                "high": ["Stakeholder Analysis", "Requirements Engineering", "System Modeling", "Architecture Patterns", "Component Design", "Interface Specifications", "Quality Attributes", "Risk Assessment", "Documentation"]
            },
            "tester": {
                "low": ["Understand Code", "Write Tests", "Execute Tests"],
                "medium": ["Analyze Requirements", "Test Planning", "Test Implementation", "Execution", "Reporting", "Validation"],
                "high": ["Requirements Analysis", "Test Strategy", "Test Planning", "Test Design", "Implementation", "Execution", "Defect Analysis", "Regression Testing", "Performance Testing", "Security Testing"]
            },
            "reviewer": {
                "low": ["Read Code", "Check Quality", "Provide Feedback"],
                "medium": ["Code Analysis", "Architecture Review", "Quality Assessment", "Security Review", "Documentation Review", "Recommendations"],
                "high": ["Comprehensive Analysis", "Architecture Assessment", "Code Quality Metrics", "Security Audit", "Performance Review", "Maintainability Analysis", "Best Practices Check", "Risk Assessment", "Strategic Recommendations"]
            }
        }
        
        # Get reasoning pattern for agent type and complexity
        pattern = reasoning_patterns.get(agent_type, reasoning_patterns["coder"])[complexity_level]
        
        # Create detailed reasoning chain
        reasoning_chain = self._build_detailed_reasoning_chain(
            agent_type, task_description, pattern, context
        )
        
        # Cache for future use
        cache_key = f"{agent_type}_{complexity_level}_{hash(task_description)}"
        self.reasoning_cache[cache_key] = reasoning_chain
        
        return reasoning_chain

    def _build_detailed_reasoning_chain(self,
                                      agent_type: str,
                                      task_description: str,
                                      pattern: List[str],
                                      context: Optional[Dict[str, Any]] = None) -> str:
        """Build a detailed reasoning chain with specific steps."""
        
        context = context or {}
        
        chain_parts = [
            f"SEQUENTIAL THINKING CHAIN FOR {agent_type.upper()}",
            f"Task: {task_description}",
            f"Complexity: {len(pattern)} steps",
            f"Context: {json.dumps(context) if context else 'None'}",
            "",
            "REASONING STEPS:"
        ]
        
        for i, step in enumerate(pattern, 1):
            detailed_step = self._get_detailed_step_instructions(agent_type, step, task_description)
            chain_parts.append(f"{i}. {step}: {detailed_step}")
            chain_parts.append("")
        
        chain_parts.extend([
            "MEMORY INTEGRATION:",
            "- Store reasoning results after each step",
            "- Check memory for related context before each step",
            "- Update coordination state with progress",
            "",
            "VALIDATION REQUIREMENTS:",
            "- Verify each step's output before proceeding",
            "- Document decision rationale",
            "- Include confidence scores for major decisions",
            "",
            "COORDINATION HOOKS:",
            "- Execute pre-step memory checks",
            "- Store intermediate results",
            "- Update other agents on progress"
        ])
        
        return "\n".join(chain_parts)

    def _get_detailed_step_instructions(self, agent_type: str, step: str, task_description: str) -> str:
        """Get detailed instructions for each reasoning step based on agent type."""
        
        step_instructions = {
            "researcher": {
                "Identify Topic": "Break down the research question into specific, searchable components",
                "Search Sources": "Use multiple search strategies and evaluate source credibility",
                "Extract Information": "Pull key facts, figures, and insights with proper citation",
                "Summarize": "Create structured summary with key findings and confidence levels",
                "Analyze Requirements": "Identify what information is needed and success criteria",
                "Research Multiple Sources": "Cross-reference information from diverse, credible sources",
                "Cross-Reference": "Validate information consistency across sources",
                "Synthesize": "Combine findings into coherent insights and recommendations",
                "Validate": "Check conclusions against evidence and identify gaps"
            },
            "coder": {
                "Understand Task": "Parse requirements and identify acceptance criteria",
                "Plan Implementation": "Design approach, identify dependencies, estimate effort",
                "Code": "Implement solution following best practices and patterns",
                "Test": "Write and execute tests to validate functionality",
                "Analyze Requirements": "Break down functional and non-functional requirements",
                "Design Architecture": "Create system design with clear component boundaries",
                "Implement Components": "Code individual components with proper interfaces",
                "Unit Testing": "Create comprehensive unit tests for each component",
                "Integration": "Combine components and test interactions",
                "Validation": "Verify solution meets all requirements"
            },
            "system-architect": {
                "Understand Requirements": "Analyze business and technical requirements thoroughly",
                "Design System": "Create high-level architecture with clear component relationships",
                "Document": "Create comprehensive architecture documentation and diagrams",
                "Requirements Analysis": "Decompose and prioritize functional and quality requirements",
                "System Design": "Design system structure, interfaces, and data flow",
                "Component Planning": "Define component responsibilities and interactions",
                "Interface Design": "Specify APIs and data contracts between components",
                "Documentation": "Create detailed architecture documentation with rationale",
                "Validation": "Review design against requirements and constraints"
            },
            "tester": {
                "Understand Code": "Analyze code structure, logic, and expected behavior",
                "Write Tests": "Create comprehensive test cases covering all scenarios",
                "Execute Tests": "Run tests and analyze results for failures and gaps",
                "Analyze Requirements": "Understand what needs to be tested and acceptance criteria",
                "Test Planning": "Create test strategy and identify test scenarios",
                "Test Implementation": "Write automated and manual test cases",
                "Execution": "Run tests systematically and collect results",
                "Reporting": "Document test results, defects, and coverage metrics",
                "Validation": "Ensure all requirements are adequately tested"
            },
            "reviewer": {
                "Read Code": "Thoroughly examine code for logic, structure, and clarity",
                "Check Quality": "Evaluate code quality against standards and best practices",
                "Provide Feedback": "Give constructive feedback with specific recommendations",
                "Code Analysis": "Deep dive into code structure, patterns, and implementation",
                "Architecture Review": "Evaluate system design and component interactions",
                "Quality Assessment": "Measure code quality metrics and adherence to standards",
                "Security Review": "Identify potential security vulnerabilities and risks",
                "Documentation Review": "Assess documentation completeness and clarity",
                "Recommendations": "Provide actionable improvement suggestions"
            }
        }
        
        agent_steps = step_instructions.get(agent_type, step_instructions["coder"])
        return agent_steps.get(step, f"Execute {step} with attention to {task_description}")

    def create_reasoning_prompt_enhancement(self, 
                                          base_prompt: str,
                                          reasoning_chain: str,
                                          agent_type: str) -> str:
        """Enhance a base prompt with sequential thinking integration."""
        
        enhanced_prompt = f"""
{base_prompt}

SEQUENTIAL THINKING ENHANCEMENT:

{reasoning_chain}

EXECUTION INSTRUCTIONS:
1. Follow the reasoning chain step-by-step
2. Do not skip steps or combine them inappropriately
3. Store intermediate results in memory after each step
4. Use the following memory keys pattern:
   - Step progress: {agent_type}_step_[number]_[session_id]
   - Reasoning notes: {agent_type}_reasoning_[session_id]
   - Final results: {agent_type}_results_[session_id]

5. Before each step, check memory for:
   - Previous step results
   - Related agent findings
   - Session context and constraints

6. After each step, validate:
   - Step completion criteria met
   - Results quality acceptable
   - Ready to proceed to next step

QUALITY GATES:
- Each step must have measurable output
- Include confidence scores (0-1) for major decisions
- Document any assumptions or risks identified
- Flag any blockers or dependencies on other agents

MEMORY COORDINATION:
- Check for updates from other agents before each step
- Share significant findings immediately
- Update coordination status after each major step
- Store reasoning process for DSPy learning

Begin execution following the sequential thinking chain above.
"""
        
        return enhanced_prompt

    def get_reasoning_status(self, session_id: str) -> Dict[str, Any]:
        """Get status of reasoning processes for a session."""
        
        session_reasoning = {}
        
        for key, chain in self.reasoning_cache.items():
            if session_id in key:
                parts = key.split('_')
                agent_type = parts[0]
                session_reasoning[agent_type] = {
                    "reasoning_chain": chain[:200] + "..." if len(chain) > 200 else chain,
                    "cached_at": "recently",  # In practice, store timestamp
                    "chain_length": len(chain.split('\n')),
                    "complexity": "high" if len(chain.split('\n')) > 20 else "medium" if len(chain.split('\n')) > 10 else "low"
                }
        
        return {
            "session_id": session_id,
            "active_reasoning_sessions": session_reasoning,
            "total_cached_chains": len(self.reasoning_cache),
            "sequential_thinking_status": "active"
        }

    def validate_mcp_server_available(self) -> bool:
        """Check if sequential thinking MCP server is available."""
        try:
            # Check MCP config
            if not self.mcp_config_path.exists():
                logger.warning("MCP config file not found")
                return False
            
            with open(self.mcp_config_path) as f:
                config = json.load(f)
            
            sequential_config = config.get("mcpServers", {}).get("sequential-thinking")
            if not sequential_config:
                logger.warning("Sequential thinking MCP server not configured")
                return False
            
            if sequential_config.get("disabled", False):
                logger.warning("Sequential thinking MCP server is disabled")
                return False
            
            logger.info("Sequential thinking MCP server is available and configured")
            return True
            
        except Exception as e:
            logger.error(f"Error checking MCP server availability: {e}")
            return False

    def generate_mcp_integration_instructions(self, agent_type: str, session_id: str) -> str:
        """Generate instructions for proper MCP server integration."""
        
        if not self.validate_mcp_server_available():
            return """
WARNING: Sequential Thinking MCP server not available.
Proceeding with built-in reasoning patterns.
Consider setting up the MCP server for enhanced capabilities.
"""
        
        return f"""
SEQUENTIAL THINKING MCP INTEGRATION INSTRUCTIONS:

1. The sequential-thinking MCP server is available and configured
2. Use it for complex multi-step reasoning tasks
3. Store reasoning results in shared memory for other agents

MCP Server Usage Pattern:
- Server: sequential-thinking  
- Session: {session_id}
- Agent: {agent_type}

Memory Integration:
- Store reasoning chains with key: seq_thinking_{agent_type}_{session_id}
- Share major insights with key: insights_{agent_type}_{session_id}
- Update progress with key: progress_{agent_type}_{session_id}

Quality Assurance:
- Validate each reasoning step before proceeding
- Include confidence scores and assumptions
- Document decision rationale for DSPy learning
"""


def create_enhanced_reasoning_example():
    """Example of enhanced reasoning integration."""
    
    wrapper = SequentialThinkingWrapper()
    
    # Create reasoning chain for a complex research task
    reasoning_chain = wrapper.create_reasoning_chain(
        agent_type="researcher",
        task_description="Research and analyze the security implications of microservices architecture",
        complexity_level="high",
        context={
            "domain": "software_architecture",
            "focus": "security",
            "depth": "comprehensive"
        }
    )
    
    # Create enhanced prompt
    base_prompt = "Research microservices security implications and provide comprehensive analysis."
    enhanced_prompt = wrapper.create_reasoning_prompt_enhancement(
        base_prompt, reasoning_chain, "researcher"
    )
    
    return enhanced_prompt, wrapper


if __name__ == "__main__":
    # Example execution
    prompt, wrapper = create_enhanced_reasoning_example()
    print("Enhanced Sequential Thinking Integration:")
    print("=" * 80)
    print(prompt)
    print("\nReasoning Status:")
    print(json.dumps(wrapper.get_reasoning_status("example_session_123"), indent=2))