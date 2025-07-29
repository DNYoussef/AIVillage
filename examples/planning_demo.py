#!/usr/bin/env python3
"""HypeRAG Planning Engine Demo

Demonstrates the strategic query planning system with various query types
and reasoning strategies.
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mcp_servers.hyperag.planning import (
    QueryClassifier,
    QueryPlan,
    QueryPlanner,
    QueryType,
    ReasoningStrategy,
    RetrievalConstraints,
    StrategySelector,
)
from mcp_servers.hyperag.planning.query_planner import AgentReasoningModel


async def demo_query_classification():
    """Demonstrate query classification capabilities"""
    print("🔍 Query Classification Demo")
    print("=" * 50)

    classifier = QueryClassifier()

    test_queries = [
        "What is the capital of France?",
        "Why did the Roman Empire fall?",
        "Compare Python and Java programming languages",
        "What happened during the Industrial Revolution?",
        "How many countries are in the European Union?",
        "What if artificial intelligence becomes sentient?",
        "Explain the causes of climate change and their effects on weather patterns",
        "What do you know about quantum computing?",
        "How are social media algorithms related to political polarization through echo chambers?"
    ]

    for query in test_queries:
        query_type, confidence, analysis = classifier.classify_query(query)

        print(f"\n📝 Query: {query}")
        print(f"   Type: {query_type.value}")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Complexity: {analysis['complexity_score']:.3f}")

        if analysis["pattern_matches"]:
            patterns = ", ".join(f"{k}({v:.2f})" for k, v in analysis["pattern_matches"].items())
            print(f"   Patterns: {patterns}")

        strategy = classifier.suggest_strategy(query_type, analysis["complexity_score"])
        print(f"   Suggested Strategy: {strategy.value}")


async def demo_strategic_planning():
    """Demonstrate strategic plan creation"""
    print("\n\n🎯 Strategic Planning Demo")
    print("=" * 50)

    # Create different agent models
    agents = [
        AgentReasoningModel(
            model_name="BasicAgent",
            capabilities=["retrieval", "basic_reasoning"],
            performance_profile={
                "max_complexity": 0.6,
                "reasoning_speed": 1.2,
                "accuracy": 0.8,
                "memory_limit_mb": 500
            }
        ),
        AgentReasoningModel(
            model_name="AdvancedAgent",
            capabilities=["retrieval", "reasoning", "causal_analysis", "temporal_reasoning"],
            performance_profile={
                "max_complexity": 0.9,
                "reasoning_speed": 0.8,
                "accuracy": 0.95,
                "memory_limit_mb": 2000
            }
        )
    ]

    planner = QueryPlanner()

    complex_query = ("How did the invention of the printing press influence the Scientific Revolution, "
                    "and what were the cascading effects on modern democratic institutions?")

    for agent in agents:
        print(f"\n🤖 Planning for {agent.model_name}")
        print(f"   Capabilities: {', '.join(agent.capabilities)}")
        print(f"   Max Complexity: {agent.max_complexity}")

        try:
            plan = await planner.create_plan(complex_query, agent)

            print("   ✅ Plan Created:")
            print(f"      Strategy: {plan.reasoning_strategy.value}")
            print(f"      Steps: {len(plan.execution_steps)}")
            print(f"      Complexity: {plan.complexity_score:.3f}")
            print(f"      Confidence: {plan.overall_confidence:.3f}")

            # Show execution steps
            for i, step in enumerate(plan.execution_steps[:3], 1):  # Show first 3 steps
                print(f"      Step {i}: {step.description}")

        except Exception as e:
            print(f"   ❌ Planning failed: {e}")


async def demo_plan_adaptation():
    """Demonstrate plan adaptation and replanning"""
    print("\n\n🔄 Plan Adaptation Demo")
    print("=" * 50)

    planner = QueryPlanner()
    agent = AgentReasoningModel(
        model_name="AdaptiveAgent",
        capabilities=["retrieval", "reasoning"],
        performance_profile={"max_complexity": 0.8, "reasoning_speed": 1.0, "accuracy": 0.85}
    )

    query = "Analyze the causal relationships between economic inequality and social unrest"

    # Create initial plan
    print("📋 Creating initial plan...")
    original_plan = await planner.create_plan(query, agent)
    print(f"   Initial Strategy: {original_plan.reasoning_strategy.value}")
    print(f"   Steps: {len(original_plan.execution_steps)}")

    # Simulate execution failure requiring replan
    print("\n⚠️  Simulating execution failure...")
    print("   Reason: Low confidence in causal analysis")

    intermediate_results = {
        "retrieved_entities": ["economic_inequality", "social_unrest", "protests"],
        "partial_relationships": ["inequality -> frustration", "frustration -> protests"]
    }

    # Trigger replanning
    new_plan = await planner.replan(
        original_plan,
        intermediate_results,
        current_confidence=0.3,
        failure_reason="causal_analysis_low_confidence"
    )

    print(f"   🔄 Replanned Strategy: {new_plan.reasoning_strategy.value}")
    print(f"   New Steps: {len(new_plan.execution_steps)}")
    print(f"   Replan Count: {new_plan.replan_count}")
    print(f"   Adaptation Reason: {new_plan.adaptation_reason}")


async def demo_strategy_comparison():
    """Demonstrate different strategies for the same query"""
    print("\n\n⚖️  Strategy Comparison Demo")
    print("=" * 50)

    selector = StrategySelector()
    query = "How do climate patterns affect agricultural productivity over time?"

    # Test different complexity levels
    complexity_levels = [0.3, 0.6, 0.9]

    for complexity in complexity_levels:
        print(f"\n🌡️  Complexity Level: {complexity:.1f}")

        # Test different query types
        query_types = [QueryType.TEMPORAL_ANALYSIS, QueryType.CAUSAL_CHAIN, QueryType.COMPARATIVE]

        for query_type in query_types:
            strategy = selector.select_strategy(
                query_type,
                complexity,
                RetrievalConstraints(),
                {"prefer_accurate": True}
            )

            requirements = selector.get_strategy_requirements(strategy)

            print(f"   {query_type.value:20} -> {strategy.value:20} "
                  f"(~{requirements['estimated_time_ms']}ms)")


async def demo_plan_dsl():
    """Demonstrate Plan DSL serialization"""
    print("\n\n📄 Plan DSL Demo")
    print("=" * 50)

    from mcp_servers.hyperag.planning.plan_structures import ExecutionStep, PlanDSL

    # Create a sample plan
    plan = QueryPlan(
        plan_id="demo-plan-001",
        original_query="What are the environmental impacts of renewable energy adoption?",
        query_type=QueryType.COMPARATIVE,
        reasoning_strategy=ReasoningStrategy.COMPARATIVE_ANALYSIS,
        complexity_score=0.6,
        retrieval_constraints=RetrievalConstraints(
            max_depth=4,
            max_nodes=150,
            confidence_threshold=0.75,
            time_budget_ms=8000
        )
    )

    # Add execution steps
    steps = [
        ExecutionStep(
            description="Extract renewable energy types",
            operation="entity_extraction",
            parameters={"focus": "renewable_energy"},
            confidence_threshold=0.8
        ),
        ExecutionStep(
            description="Retrieve environmental impact data",
            operation="impact_data_retrieval",
            parameters={"impact_types": ["carbon", "land_use", "water"]},
            dependencies=[],
            confidence_threshold=0.75
        ),
        ExecutionStep(
            description="Compare environmental benefits",
            operation="comparative_analysis",
            parameters={"analysis_type": "environmental_comparison"},
            dependencies=[],
            confidence_threshold=0.7
        )
    ]

    for step in steps:
        plan.add_step(step)

    # Serialize to DSL
    dsl_text = PlanDSL.serialize_plan(plan)

    print("📝 Generated Plan DSL:")
    print("-" * 40)
    print(dsl_text)
    print("-" * 40)


async def demo_learning_system():
    """Demonstrate the learning system capabilities"""
    print("\n\n🧠 Learning System Demo")
    print("=" * 50)


    from mcp_servers.hyperag.planning.learning import PlanLearner, StrategyFeedback

    learner = PlanLearner()

    # Simulate some feedback
    feedback_examples = [
        StrategyFeedback(
            strategy=ReasoningStrategy.CAUSAL_REASONING,
            query_type=QueryType.CAUSAL_CHAIN,
            complexity_score=0.7,
            success=True,
            confidence_achieved=0.85,
            execution_time_ms=2500,
            steps_completed=4,
            steps_failed=0,
            result_quality=0.9
        ),
        StrategyFeedback(
            strategy=ReasoningStrategy.DIRECT_RETRIEVAL,
            query_type=QueryType.SIMPLE_FACT,
            complexity_score=0.2,
            success=True,
            confidence_achieved=0.95,
            execution_time_ms=800,
            steps_completed=1,
            steps_failed=0,
            result_quality=0.85
        ),
        StrategyFeedback(
            strategy=ReasoningStrategy.STEP_BY_STEP,
            query_type=QueryType.MULTI_HOP,
            complexity_score=0.9,
            success=False,
            confidence_achieved=0.3,
            execution_time_ms=5000,
            steps_completed=2,
            steps_failed=3,
            result_quality=0.4
        )
    ]

    # Record feedback
    for feedback in feedback_examples:
        # Create dummy plan for feedback
        plan = QueryPlan(
            query_type=feedback.query_type,
            reasoning_strategy=feedback.strategy
        )

        learner.record_execution_feedback(plan, feedback)

    # Get strategy recommendations
    print("🎯 Strategy Recommendations:")

    test_scenarios = [
        (QueryType.CAUSAL_CHAIN, 0.6),
        (QueryType.SIMPLE_FACT, 0.3),
        (QueryType.MULTI_HOP, 0.8)
    ]

    for query_type, complexity in test_scenarios:
        strategy, confidence = learner.get_strategy_recommendation(query_type, complexity)
        print(f"   {query_type.value:15} (complexity {complexity:.1f}) -> "
              f"{strategy.value:20} (confidence: {confidence:.3f})")

    # Get learning insights
    insights = learner.get_learning_insights()
    print("\n📊 Learning Statistics:")
    print(f"   Total Feedback: {insights['learning_stats']['total_feedback_received']}")
    print(f"   Successful Adaptations: {insights['learning_stats']['successful_adaptations']}")

    if insights["top_performing_strategies"]:
        print("\n🏆 Top Performing Strategies:")
        for strategy_info in insights["top_performing_strategies"][:3]:
            print(f"   {strategy_info['strategy']:20} - "
                  f"Success: {strategy_info['success_rate']:.2f}, "
                  f"Confidence: {strategy_info['avg_confidence']:.2f}")


async def main():
    """Main demo function"""
    print("🚀 HypeRAG Planning Engine Demo")
    print("=" * 60)

    try:
        await demo_query_classification()
        await demo_strategic_planning()
        await demo_plan_adaptation()
        await demo_strategy_comparison()
        await demo_plan_dsl()
        await demo_learning_system()

        print("\n\n✅ Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("  • Intelligent query classification")
        print("  • Strategic plan generation")
        print("  • Adaptive replanning")
        print("  • Strategy selection and comparison")
        print("  • Plan serialization (DSL)")
        print("  • Learning from execution feedback")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    asyncio.run(main())
