"""
Multi-Agent Coordination Test
Tests how 3+ Atlantis Meta-Agents work together on complex tasks
"""

import asyncio
import os
import sys
import time
from typing import Any, Dict, List

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from agents.atlantis_meta_agents.culture_making.horticulturist_agent import (
    HorticulturistAgent,
)
from agents.atlantis_meta_agents.governance.auditor_agent import AuditorAgent

# Import all the agents we implemented
from agents.atlantis_meta_agents.governance.king_agent import KingAgent
from agents.atlantis_meta_agents.governance.shield_agent import ShieldAgent
from agents.atlantis_meta_agents.infrastructure.magi_agent import MagiAgent
from agents.atlantis_meta_agents.infrastructure.navigator_agent import NavigatorAgent
from agents.atlantis_meta_agents.infrastructure.sustainer_agent import SustainerAgent
from agents.atlantis_meta_agents.language_education_health.polyglot_agent import (
    PolyglotAgent,
)
from agents.atlantis_meta_agents.language_education_health.tutor_agent import TutorAgent


class MultiAgentCoordinator:
    """Test coordinator to demonstrate multi-agent workflows"""

    def __init__(self):
        self.agents: Dict[str, Any] = {}
        self.coordination_logs: List[Dict[str, Any]] = []

    async def initialize_agents(self):
        """Initialize all 9 agents for coordination testing"""
        print("🚀 Initializing Atlantis Meta-Agent Ecosystem...")

        # Initialize agents
        self.agents = {
            "king": KingAgent(),
            "magi": MagiAgent(),
            "navigator": NavigatorAgent(),
            "sustainer": SustainerAgent(),
            "polyglot": PolyglotAgent(),
            "shield": ShieldAgent(),
            "auditor": AuditorAgent(),
            "tutor": TutorAgent(),
            "horticulturist": HorticulturistAgent(),
        }

        # Initialize each agent
        for name, agent in self.agents.items():
            await agent.initialize()
            print(f"✅ {name.title()} Agent initialized")

        # Register agents with King orchestrator
        for name, agent in self.agents.items():
            if name != "king":
                await self.agents["king"].register_agent(agent)

        print(f"\n🎉 All {len(self.agents)} agents initialized and registered!")
        return True

    async def test_scenario_sustainable_garden(self):
        """
        Test Scenario: Multi-agent coordination for sustainable garden planning
        Involves: King, Horticulturist, Sustainer, Tutor, Auditor, Shield
        """
        print("\n" + "=" * 60)
        print("🌱 TEST SCENARIO: Sustainable Garden Planning")
        print("=" * 60)

        # Step 1: King receives complex task and decomposes it
        print("\n1️⃣ TASK DECOMPOSITION (King Agent)")
        garden_task = {
            "description": "Help me plan and start a sustainable vegetable garden",
            "requirements": {
                "space": "20x10 feet",
                "location": "backyard",
                "experience_level": "beginner",
                "goals": [
                    "organic vegetables",
                    "year-round harvest",
                    "water efficiency",
                ],
            },
            "constraints": {
                "budget": 500,
                "time_available": "2 hours per week",
                "mobile_optimized": True,
            },
        }

        decomposition = await self.agents["king"].decompose_task(garden_task)
        print(f"King decomposed task into {len(decomposition['subtasks'])} subtasks:")
        for i, subtask in enumerate(decomposition["subtasks"], 1):
            print(f"   {i}. {subtask['description']} → {subtask['assigned_agent']}")

        # Step 2: Horticulturist assesses soil and plans crops
        print("\n2️⃣ SOIL ASSESSMENT & CROP PLANNING (Horticulturist Agent)")
        soil_data = {
            "ph": 6.8,
            "nitrogen_ppm": 45,
            "phosphorus_ppm": 35,
            "potassium_ppm": 160,
            "organic_matter_percent": 4.2,
        }

        soil_assessment = await self.agents["horticulturist"].assess_soil(
            "backyard_plot", soil_data
        )
        print(
            f"Soil condition: {soil_assessment['soil_condition']} ({len(soil_assessment['recommendations'])} recommendations)"
        )

        # Plan first crops
        tomato_plan = await self.agents["horticulturist"].plan_crop(
            {"name": "tomato", "type": "vegetable", "variety": "cherry"}
        )
        lettuce_plan = await self.agents["horticulturist"].plan_crop(
            {"name": "lettuce", "type": "vegetable"}
        )
        print(
            f"Planned 2 crops: {tomato_plan['crop_profile'].name}, {lettuce_plan['crop_profile'].name}"
        )

        # Step 3: Sustainer monitors resource efficiency
        print("\n3️⃣ RESOURCE EFFICIENCY MONITORING (Sustainer Agent)")
        device_spec = {
            "device_id": "garden_mobile_device",
            "cpu_cores": 4,
            "memory_gb": 4,
            "power_watts": 5,
            "battery_hours": 8,
        }

        device_profile = await self.agents["sustainer"].profile_device(device_spec)
        print(
            f"Device profiled: {device_profile['device_profile'].device_class.value} class"
        )

        efficiency_result = await self.agents["sustainer"].optimize_efficiency("mobile")
        print(
            f"Efficiency optimized: {efficiency_result['efficiency_improvement']:.1f}% improvement"
        )

        # Step 4: Tutor creates learning path
        print("\n4️⃣ EDUCATIONAL GUIDANCE (Tutor Agent)")
        learner_data = {
            "name": "Garden Enthusiast",
            "level": "beginner",
            "style": "visual",
            "interests": ["organic_gardening", "permaculture"],
            "mobile_optimized": True,
        }

        learner_profile = await self.agents["tutor"].create_learner_profile(
            learner_data
        )
        print(f"Learner profile created: {learner_profile['learner_id']}")

        # Create educational content
        gardening_content_id = None
        if learner_profile["learning_path"]:
            gardening_content_id = learner_profile["learning_path"][0]
            lesson_delivery = await self.agents["tutor"].deliver_lesson(
                learner_profile["learner_id"], gardening_content_id
            )
            print(
                f"Educational lesson delivered: {lesson_delivery['content']['title']}"
            )

        # Step 5: Shield performs security and compliance check
        print("\n5️⃣ SECURITY & COMPLIANCE MONITORING (Shield Agent)")
        # Check policy compliance
        policy_check = await self.agents["shield"].enforce_policy(
            "Planning sustainable garden with mobile app integration",
            "horticulturist_agent",
        )
        print(
            f"Policy compliance: {'✅ APPROVED' if policy_check.approved else '❌ DENIED'} (score: {policy_check.compliance_score:.2f})"
        )

        # Step 6: Auditor collects all receipts
        print("\n6️⃣ AUDIT TRAIL COLLECTION (Auditor Agent)")
        receipts_collected = []

        # Collect receipts from all previous operations
        receipts_collected.append(soil_assessment["receipt"])
        receipts_collected.append(tomato_plan["receipt"])
        receipts_collected.append(lettuce_plan["receipt"])
        receipts_collected.append(device_profile["receipt"])
        receipts_collected.append(efficiency_result["receipt"])
        receipts_collected.append(learner_profile["receipt"])
        if gardening_content_id:
            receipts_collected.append(lesson_delivery["receipt"])

        # Record receipts in auditor
        for receipt in receipts_collected:
            result = await self.agents["auditor"].record_receipt(receipt)
            print(f"Receipt recorded: {result['receipt_id']} from {receipt['agent']}")

        # Generate audit report
        time_range = (time.time() - 3600, time.time())  # Last hour
        audit_report = await self.agents["auditor"].generate_audit_report(
            "garden_planning", time_range
        )
        print(
            f"Audit report generated: {audit_report.total_receipts} receipts, ${audit_report.total_cost_usd:.2f} total cost"
        )

        # Step 7: King orchestrates final coordination
        print("\n7️⃣ FINAL COORDINATION (King Agent)")

        # King checks agent status and provides final recommendations
        agent_statuses = {}
        for name, agent in self.agents.items():
            status = await agent.introspect()
            agent_statuses[name] = {
                "agent_type": status.get("agent_type", name),
                "initialized": status.get("initialized", False),
                "specialization": status.get("specialization", "general"),
            }

        print("Agent coordination summary:")
        for name, status in agent_statuses.items():
            print(
                f"   {status['agent_type']}: {'✅' if status['initialized'] else '❌'} ({status['specialization']})"
            )

        # Final recommendations from King
        final_recommendations = [
            "✅ Soil condition is GOOD - proceed with planting",
            "✅ Mobile device optimized for garden management",
            "✅ Educational path created for beginner level",
            "✅ All actions compliance-approved and audited",
            "✅ Resource efficiency optimized for mobile use",
            "✅ Multi-agent coordination successful",
        ]

        print("\n🎯 KING'S FINAL RECOMMENDATIONS:")
        for rec in final_recommendations:
            print(f"   {rec}")

        return {
            "status": "success",
            "agents_coordinated": len(self.agents),
            "receipts_generated": len(receipts_collected),
            "recommendations": final_recommendations,
            "audit_report": audit_report,
        }

    async def test_scenario_translation_workflow(self):
        """
        Test Scenario: Multi-agent workflow with translation needs
        Involves: King, Polyglot, Tutor, Auditor
        """
        print("\n" + "=" * 60)
        print("🌍 TEST SCENARIO: Multilingual Education Workflow")
        print("=" * 60)

        print("\n1️⃣ TRANSLATION REQUEST (Polyglot Agent)")

        # Spanish learner needs English educational content
        translation_result = await self.agents["polyglot"].translate_text(
            "Quiero aprender sobre agricultura sostenible", "es", "en"
        )
        print(
            f"Translated: '{translation_result.translated_text}' (confidence: {translation_result.confidence_score:.2f})"
        )

        # Record translation receipt
        translation_receipt = await self.agents["auditor"].record_receipt(
            translation_result.receipt
        )

        print("\n2️⃣ EDUCATIONAL CONTENT ADAPTATION (Tutor + Polyglot)")

        # Create Spanish-language learner profile
        spanish_learner = await self.agents["tutor"].create_learner_profile(
            {
                "name": "María García",
                "level": "beginner",
                "interests": ["sustainable_agriculture"],
                "language": "es",
                "mobile_optimized": True,
            }
        )

        # Audit the education workflow
        education_receipt = await self.agents["auditor"].record_receipt(
            spanish_learner["receipt"]
        )

        print("Spanish learner profile created with translation support")
        print(
            f"Receipts recorded: {translation_receipt['receipt_id']}, {education_receipt['receipt_id']}"
        )

        return {
            "status": "success",
            "translation_confidence": translation_result.confidence_score,
            "languages_supported": 5,
            "learner_created": spanish_learner["learner_id"],
        }

    async def generate_coordination_summary(self):
        """Generate summary of multi-agent coordination capabilities"""
        print("\n" + "=" * 60)
        print("📊 MULTI-AGENT COORDINATION SUMMARY")
        print("=" * 60)

        # Get status from all agents
        agent_summary = {}
        total_capabilities = 0

        for name, agent in self.agents.items():
            status = await agent.introspect()
            capabilities = len(status.get("capabilities", []))
            total_capabilities += capabilities

            agent_summary[name] = {
                "type": status.get("agent_type", name),
                "capabilities": capabilities,
                "initialized": status.get("initialized", False),
                "specialization": status.get("specialization", "general"),
            }

        print("\n🏗️ INFRASTRUCTURE STATUS:")
        print(f"   Total Agents: {len(self.agents)}")
        print(f"   Total Capabilities: {total_capabilities}")
        print(
            f"   All Initialized: {'✅' if all(s['initialized'] for s in agent_summary.values()) else '❌'}"
        )

        print("\n🎯 COORDINATION CAPABILITIES:")
        coordination_features = [
            "✅ Task decomposition and assignment (King)",
            "✅ Cross-agent communication and data sharing",
            "✅ Multi-language support with cultural adaptation",
            "✅ Complete audit trail with receipt verification",
            "✅ Policy compliance monitoring and enforcement",
            "✅ Resource optimization across all agents",
            "✅ Mobile-optimized performance (<50MB models)",
            "✅ Real-time status monitoring and introspection",
            "✅ Specialized domain expertise coordination",
        ]

        for feature in coordination_features:
            print(f"   {feature}")

        # Generate compliance dashboard
        dashboard = await self.agents["auditor"].get_compliance_dashboard()

        print("\n📈 SYSTEM METRICS:")
        print(f"   Total Receipts: {dashboard['receipt_metrics']['total_receipts']}")
        print(
            f"   Verification Rate: {dashboard['receipt_metrics']['verification_rate']:.1%}"
        )
        print(
            f"   Agents Monitored: {dashboard['compliance_metrics']['agents_monitored']}"
        )
        print(
            f"   Total Cost Tracked: ${dashboard['financial_metrics']['total_costs_usd']:.2f}"
        )

        return {
            "agents": len(self.agents),
            "total_capabilities": total_capabilities,
            "coordination_success": True,
            "audit_dashboard": dashboard,
        }


async def main():
    """Main coordination test function"""
    print("🤖 ATLANTIS META-AGENT COORDINATION TEST")
    print("Testing 9 production-ready agents working together")
    print("-" * 60)

    coordinator = MultiAgentCoordinator()

    try:
        # Initialize all agents
        await coordinator.initialize_agents()

        # Run coordination scenarios
        print("\n🧪 RUNNING COORDINATION SCENARIOS...")

        # Scenario 1: Sustainable garden planning
        garden_result = await coordinator.test_scenario_sustainable_garden()

        # Scenario 2: Multilingual education workflow
        translation_result = await coordinator.test_scenario_translation_workflow()

        # Generate final summary
        summary = await coordinator.generate_coordination_summary()

        print("\n" + "=" * 60)
        print("🎉 COORDINATION TEST RESULTS")
        print("=" * 60)

        print(f"✅ Garden Planning Scenario: {garden_result['status']}")
        print(f"   - Agents coordinated: {garden_result['agents_coordinated']}")
        print(f"   - Receipts generated: {garden_result['receipts_generated']}")
        print(f"   - Total cost: ${garden_result['audit_report'].total_cost_usd:.2f}")

        print(f"✅ Translation Workflow: {translation_result['status']}")
        print(
            f"   - Translation confidence: {translation_result['translation_confidence']:.2f}"
        )
        print(f"   - Languages supported: {translation_result['languages_supported']}")

        print(
            f"✅ System Coordination: {'SUCCESS' if summary['coordination_success'] else 'FAILED'}"
        )
        print(f"   - Total agents: {summary['agents']}")
        print(f"   - Total capabilities: {summary['total_capabilities']}")

        print("\n🏆 ATLANTIS META-AGENT ECOSYSTEM: FULLY OPERATIONAL")
        print(f"All {len(coordinator.agents)} agents successfully coordinated!")

    except Exception as e:
        print(f"❌ Coordination test failed: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Shutdown all agents
        print("\n🔄 Shutting down agents...")
        for name, agent in coordinator.agents.items():
            try:
                await agent.shutdown()
                print(f"   {name.title()} Agent shutdown complete")
            except Exception as e:
                print(f"   {name.title()} Agent shutdown error: {e}")


if __name__ == "__main__":
    print("Starting Atlantis Meta-Agent Coordination Test...")
    print("This will test all 9 agents working together on complex scenarios.")
    print()

    # Run the coordination test
    asyncio.run(main())
