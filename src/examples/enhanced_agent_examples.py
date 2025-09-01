#!/usr/bin/env python3
"""Enhanced Agent Coordination Examples.

This file demonstrates the correct way to spawn agents with:
1. Shared memory MCP integration
2. Sequential thinking for intelligent reasoning  
3. DSPy system for prompt optimization and learning

These examples show the mandatory pattern for all future agent coordination.
"""

import sys
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.coordination.enhanced_agent_coordinator import EnhancedAgentCoordinator
from src.coordination.sequential_thinking_wrapper import SequentialThinkingWrapper
from src.coordination.dspy_integration import DSPyAgentOptimizer


def example_full_stack_development():
    """Example: Full-stack development with enhanced agent coordination."""
    
    coordinator = EnhancedAgentCoordinator()
    
    # Define agent configuration with enhanced capabilities
    agent_configs = [
        {
            "agent_type": "researcher",
            "task": "Research modern full-stack development patterns, focusing on microservices, security, and scalability",
            "context_keys": ["tech_stack", "security_requirements", "scalability_patterns"],
            "reasoning_chain": "Analyze Requirements -> Research Technologies -> Compare Solutions -> Validate Security -> Synthesize Recommendations"
        },
        {
            "agent_type": "system-architect", 
            "task": "Design comprehensive system architecture based on research findings",
            "context_keys": ["researcher_results", "business_requirements", "technical_constraints"],
            "reasoning_chain": "Review Research -> Identify Architecture Patterns -> Design System Components -> Define Interfaces -> Validate Design -> Document Architecture"
        },
        {
            "agent_type": "backend-dev",
            "task": "Implement backend services with REST APIs, authentication, and database integration",
            "context_keys": ["system_architecture", "api_specifications", "database_schema"],
            "reasoning_chain": "Analyze Architecture -> Plan Implementation -> Setup Project Structure -> Implement APIs -> Add Authentication -> Integrate Database -> Write Tests"
        },
        {
            "agent_type": "mobile-dev",
            "task": "Create React Native mobile application with offline capabilities and real-time updates",
            "context_keys": ["backend_apis", "ui_requirements", "offline_strategies"],
            "reasoning_chain": "Review APIs -> Design UI/UX -> Setup React Native -> Implement Core Features -> Add Offline Support -> Integrate Real-time -> Test Cross-platform"
        },
        {
            "agent_type": "tester",
            "task": "Create comprehensive test suite including unit, integration, and end-to-end tests",
            "context_keys": ["backend_implementation", "mobile_app", "system_architecture"],
            "reasoning_chain": "Analyze System -> Plan Test Strategy -> Write Unit Tests -> Create Integration Tests -> Build E2E Tests -> Setup CI/CD -> Generate Coverage Reports"
        },
        {
            "agent_type": "reviewer",
            "task": "Conduct comprehensive code review focusing on security, performance, and maintainability",
            "context_keys": ["all_implementations", "test_results", "architecture_docs"],
            "reasoning_chain": "Review Code Quality -> Security Audit -> Performance Analysis -> Maintainability Check -> Documentation Review -> Generate Recommendations"
        }
    ]
    
    # Generate Task tool instructions
    task_instructions = coordinator.generate_task_instructions(agent_configs)
    
    return f"""
ENHANCED FULL-STACK DEVELOPMENT EXAMPLE

Execute this in a single message to spawn all agents with enhanced coordination:

{task_instructions}

// Additional coordination setup
TodoWrite({{
  todos: [
    {{content: "Research modern tech stack", status: "in_progress", activeForm: "Researching modern tech stack"}},
    {{content: "Design system architecture", status: "pending", activeForm: "Designing system architecture"}},
    {{content: "Implement backend services", status: "pending", activeForm: "Implementing backend services"}},
    {{content: "Build mobile application", status: "pending", activeForm: "Building mobile application"}},
    {{content: "Create comprehensive tests", status: "pending", activeForm: "Creating comprehensive tests"}},
    {{content: "Conduct code review", status: "pending", activeForm: "Conducting code review"}},
    {{content: "Deploy to production", status: "pending", activeForm: "Deploying to production"}},
    {{content: "Document final system", status: "pending", activeForm: "Documenting final system"}}
  ]
}})

// Memory initialization
Bash("mkdir -p .mcp")

// File structure creation  
Bash("mkdir -p {{backend,mobile,tests,docs}}")
Write("backend/package.json")
Write("mobile/package.json") 
Write("tests/test-plan.md")
Write("docs/architecture.md")
"""


def example_ai_research_project():
    """Example: AI/ML research project with enhanced coordination."""
    
    coordinator = EnhancedAgentCoordinator()
    
    agent_configs = [
        {
            "agent_type": "researcher",
            "task": "Research state-of-the-art transformer architectures for multimodal AI systems",
            "context_keys": ["transformer_variants", "multimodal_approaches", "performance_benchmarks"],
            "reasoning_chain": "Literature Review -> Architecture Analysis -> Performance Comparison -> Innovation Identification -> Research Synthesis"
        },
        {
            "agent_type": "ml-developer",
            "task": "Implement and experiment with novel transformer architecture variants",
            "context_keys": ["research_findings", "architecture_specifications", "dataset_requirements"],
            "reasoning_chain": "Review Research -> Design Experiments -> Implement Models -> Training Pipeline -> Evaluation Metrics -> Results Analysis"
        },
        {
            "agent_type": "performance-benchmarker",
            "task": "Create comprehensive benchmarking suite for model evaluation",
            "context_keys": ["model_implementations", "evaluation_criteria", "baseline_comparisons"],
            "reasoning_chain": "Define Benchmarks -> Create Test Suite -> Performance Testing -> Memory Analysis -> Speed Optimization -> Comparative Analysis"
        },
        {
            "agent_type": "reviewer",
            "task": "Review research methodology, implementation quality, and experimental rigor",
            "context_keys": ["research_methodology", "code_implementation", "experimental_results"],
            "reasoning_chain": "Methodology Review -> Code Quality Analysis -> Experimental Validation -> Statistical Analysis -> Reproducibility Check -> Publication Preparation"
        }
    ]
    
    task_instructions = coordinator.generate_task_instructions(agent_configs)
    
    return f"""
ENHANCED AI/ML RESEARCH PROJECT EXAMPLE

{task_instructions}

// Research-specific coordination
TodoWrite({{
  todos: [
    {{content: "Literature review on transformer architectures", status: "in_progress", activeForm: "Conducting literature review on transformer architectures"}},
    {{content: "Implement novel architecture variants", status: "pending", activeForm: "Implementing novel architecture variants"}},
    {{content: "Create benchmarking framework", status: "pending", activeForm: "Creating benchmarking framework"}},
    {{content: "Run comprehensive experiments", status: "pending", activeForm: "Running comprehensive experiments"}},
    {{content: "Analyze experimental results", status: "pending", activeForm: "Analyzing experimental results"}},
    {{content: "Prepare research publication", status: "pending", activeForm: "Preparing research publication"}}
  ]
}})

// Research environment setup
Bash("mkdir -p {{research,models,experiments,data,results}}")
Write("research/literature-review.md")
Write("models/transformer_variants.py")
Write("experiments/benchmark_suite.py")
Write("results/analysis.ipynb")
"""


def example_security_audit_project():
    """Example: Security audit with enhanced coordination."""
    
    coordinator = EnhancedAgentCoordinator()
    
    agent_configs = [
        {
            "agent_type": "security-manager",
            "task": "Conduct comprehensive security audit of distributed system architecture",
            "context_keys": ["system_architecture", "security_standards", "threat_models"],
            "reasoning_chain": "Architecture Analysis -> Threat Modeling -> Vulnerability Assessment -> Risk Analysis -> Mitigation Planning -> Security Recommendations"
        },
        {
            "agent_type": "code-analyzer",
            "task": "Perform static and dynamic code analysis for security vulnerabilities",
            "context_keys": ["codebase_structure", "security_patterns", "vulnerability_database"],
            "reasoning_chain": "Code Structure Analysis -> Static Analysis -> Dynamic Testing -> Vulnerability Detection -> Risk Assessment -> Remediation Recommendations"
        },
        {
            "agent_type": "tester",
            "task": "Execute penetration testing and security validation",
            "context_keys": ["security_vulnerabilities", "system_endpoints", "attack_vectors"],
            "reasoning_chain": "Test Planning -> Environment Setup -> Automated Testing -> Manual Penetration -> Exploit Verification -> Impact Assessment -> Report Generation"
        },
        {
            "agent_type": "reviewer",
            "task": "Review security findings and provide strategic security recommendations",
            "context_keys": ["security_assessment", "vulnerability_reports", "business_impact"],
            "reasoning_chain": "Findings Review -> Risk Prioritization -> Business Impact Analysis -> Remediation Strategy -> Compliance Check -> Final Recommendations"
        }
    ]
    
    task_instructions = coordinator.generate_task_instructions(agent_configs)
    
    return f"""
ENHANCED SECURITY AUDIT PROJECT EXAMPLE

{task_instructions}

// Security-specific coordination
TodoWrite({{
  todos: [
    {{content: "Analyze system architecture for security", status: "in_progress", activeForm: "Analyzing system architecture for security"}},
    {{content: "Perform static code analysis", status: "pending", activeForm: "Performing static code analysis"}},
    {{content: "Execute dynamic security testing", status: "pending", activeForm: "Executing dynamic security testing"}},
    {{content: "Conduct penetration testing", status: "pending", activeForm: "Conducting penetration testing"}},
    {{content: "Generate security assessment report", status: "pending", activeForm: "Generating security assessment report"}},
    {{content: "Develop remediation strategy", status: "pending", activeForm: "Developing remediation strategy"}}
  ]
}})

// Security audit environment
Bash("mkdir -p {{security-audit,static-analysis,pen-testing,reports}}")
Write("security-audit/threat-model.md")
Write("static-analysis/scan-results.json")
Write("pen-testing/test-plan.md")  
Write("reports/security-assessment.md")
"""


def demonstrate_memory_and_sequential_thinking():
    """Demonstrate memory and sequential thinking integration."""
    
    # Initialize components
    coordinator = EnhancedAgentCoordinator()
    thinking_wrapper = SequentialThinkingWrapper()
    dspy_optimizer = DSPyAgentOptimizer()
    
    # Create reasoning chain for complex task
    reasoning_chain = thinking_wrapper.create_reasoning_chain(
        agent_type="system-architect",
        task_description="Design a fault-tolerant microservices architecture with eventual consistency",
        complexity_level="high",
        context={
            "domain": "distributed_systems",
            "requirements": ["fault_tolerance", "eventual_consistency", "scalability"],
            "constraints": ["budget", "timeline", "team_expertise"]
        }
    )
    
    # Store sample performance data for DSPy learning
    dspy_optimizer.record_agent_performance(
        agent_type="system-architect",
        session_id="demo_session_001", 
        task_description="Design microservices architecture",
        performance_data={
            "completion_status": "completed",
            "completion_time": 180.0,
            "quality_score": 0.92,
            "reasoning_quality": 0.89,
            "output_quality": 0.95,
            "error_count": 0,
            "memory_usage": 67.3
        }
    )
    
    # Get optimized prompt if available
    optimized_prompt = dspy_optimizer.get_optimized_prompt("system-architect")
    
    return f"""
MEMORY AND SEQUENTIAL THINKING DEMONSTRATION

Sequential Thinking Chain Generated:
{reasoning_chain[:500]}...

DSPy Optimization Status:
{dspy_optimizer.get_optimization_status()}

Enhanced Agent Prompt Pattern:
{optimized_prompt[:300] + "..." if optimized_prompt else "No optimized prompt available yet"}

Memory Integration:
- Shared memory database: .mcp/memory.db
- Coordination state tracking
- Cross-agent communication
- Performance learning storage
"""


if __name__ == "__main__":
    print("ENHANCED AGENT COORDINATION EXAMPLES")
    print("=" * 80)
    
    print("\n1. FULL-STACK DEVELOPMENT:")
    print(example_full_stack_development())
    
    print("\n2. AI/ML RESEARCH PROJECT:")
    print(example_ai_research_project())
    
    print("\n3. SECURITY AUDIT PROJECT:")  
    print(example_security_audit_project())
    
    print("\n4. MEMORY & SEQUENTIAL THINKING:")
    print(demonstrate_memory_and_sequential_thinking())