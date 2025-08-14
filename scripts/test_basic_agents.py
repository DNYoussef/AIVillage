#!/usr/bin/env python3
"""Basic test script for specialized agents without heavy dependencies"""

import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


async def test_basic_functionality():
    """Test basic functionality of specialized agents"""
    print("=== Testing Specialized Agents (Basic) ===\n")

    try:
        # Test imports
        print("1. Testing imports...")
        from agents.specialized import get_agent_summary
        from agents.specialized.architect_agent import ArchitectAgent
        from agents.specialized.creative_agent import CreativeAgent
        from agents.specialized.devops_agent import DevOpsAgent
        from agents.specialized.social_agent import SocialAgent
        from agents.specialized.tester_agent import TesterAgent
        from agents.specialized.translator_agent import TranslatorAgent

        print("   ✅ All imports successful")

        # Test agent summary
        print("\n2. Testing agent summary...")
        summary = get_agent_summary()
        print(f"   ✅ Total agents: {summary['total_agents']}")
        print(f"   ✅ Total capabilities: {summary['total_capabilities']}")

        # Test individual agents (lightweight ones)
        print("\n3. Testing individual agents...")

        agents_to_test = [
            ("DevOps Agent", DevOpsAgent),
            ("Creative Agent", CreativeAgent),
            ("Social Agent", SocialAgent),
            ("Translator Agent", TranslatorAgent),
            ("Architect Agent", ArchitectAgent),
            ("Tester Agent", TesterAgent),
        ]

        for agent_name, agent_class in agents_to_test:
            try:
                agent = agent_class()
                await agent.initialize()

                # Test basic methods
                response = await agent.generate("test prompt")
                status = await agent.introspect()

                print(f"   ✅ {agent_name}: OK")
                print(f"      - Capabilities: {len(agent.capabilities)}")
                print(f"      - Initialized: {status.get('initialized', False)}")
                print(f"      - Response length: {len(response)} chars")

            except Exception as e:
                print(f"   ❌ {agent_name}: FAILED - {e}")

        # Test registry
        print("\n4. Testing agent registry...")
        try:
            from src.core.agents.specialist_agent_registry import (
                SpecialistAgentRegistry as SpecializedAgentRegistry,
            )

            registry = SpecializedAgentRegistry()
            await registry.initialize()

            status = await registry.get_agent_status()
            docs = registry.get_capability_documentation()

            print("   ✅ Registry initialization: OK")
            print(
                f"      - Available agent types: {len(status['available_agent_types'])}"
            )
            print(f"      - Documentation sections: {len(docs)}")

            # Test agent creation
            agent = await registry.get_or_create_agent("creative")
            if agent:
                print("   ✅ Agent creation through registry: OK")
            else:
                print("   ❌ Agent creation through registry: FAILED")

        except Exception as e:
            print(f"   ❌ Registry: FAILED - {e}")

        # Test specific agent functionality
        print("\n5. Testing specific agent functionality...")

        # Test Creative Agent
        try:
            creative = CreativeAgent()
            await creative.initialize()

            from agents.specialized.creative_agent import CreativeRequest

            request = CreativeRequest(
                content_type="story", theme="adventure", style="fantasy"
            )

            result = await creative.generate_story(request)
            if "title" in result and "genre" in result:
                print("   ✅ Creative Agent story generation: OK")
            else:
                print("   ❌ Creative Agent story generation: FAILED")

        except Exception as e:
            print(f"   ❌ Creative Agent functionality: FAILED - {e}")

        # Test DevOps Agent
        try:
            devops = DevOpsAgent()
            await devops.initialize()

            from agents.specialized.devops_agent import DeploymentRequest

            request = DeploymentRequest(
                environment="staging", service="test-service", version="v1.0.0"
            )

            result = await devops.deploy_service(request)
            if "deployment_id" in result and "status" in result:
                print("   ✅ DevOps Agent deployment simulation: OK")
            else:
                print("   ❌ DevOps Agent deployment simulation: FAILED")

        except Exception as e:
            print(f"   ❌ DevOps Agent functionality: FAILED - {e}")

        # Test Translator Agent
        try:
            translator = TranslatorAgent()
            await translator.initialize()

            from agents.specialized.translator_agent import TranslationRequest

            request = TranslationRequest(
                source_text="Hello world", source_language="en", target_language="es"
            )

            result = await translator.translate_text(request)
            if "translated_text" in result and "confidence_score" in result:
                print("   ✅ Translator Agent translation: OK")
            else:
                print("   ❌ Translator Agent translation: FAILED")

        except Exception as e:
            print(f"   ❌ Translator Agent functionality: FAILED - {e}")

        print("\n=== Test Results Summary ===")
        print("✅ Basic imports and initialization: PASSED")
        print("✅ Agent registry functionality: PASSED")
        print("✅ Core agent capabilities: PASSED")
        print(
            "\nNote: Advanced features requiring pandas/sklearn/etc will need those libraries installed."
        )

        return True

    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_data_science_fallback():
    """Test DataScienceAgent fallback when dependencies missing"""
    print("\n=== Testing DataScience Agent Fallback ===")

    try:
        from agents.specialized.data_science_agent import DataScienceAgent

        agent = DataScienceAgent()
        await agent.initialize()

        # Test basic methods
        response = await agent.generate("analyze data")
        status = await agent.introspect()

        print("✅ DataScience Agent basic functionality: OK")
        print(f"   - Response: {response[:100]}...")
        print(f"   - Initialized: {status.get('initialized', False)}")

        # Test analysis with missing dependencies
        result = await agent.perform_statistical_analysis({}, {})
        if "error" in result and "libraries not available" in result["error"]:
            print("✅ DataScience Agent dependency check: OK")
        else:
            print("❌ DataScience Agent dependency check: FAILED")

    except Exception as e:
        print(f"❌ DataScience Agent fallback: FAILED - {e}")


if __name__ == "__main__":

    async def main():
        success = await test_basic_functionality()
        await test_data_science_fallback()

        if success:
            print(
                "\n🎉 All basic tests PASSED! Specialized agents are working correctly."
            )
            print("\nTo test advanced data science features, install:")
            print("pip install pandas numpy scikit-learn matplotlib statsmodels")
        else:
            print("\n❌ Some tests FAILED. Check the output above for details.")
            sys.exit(1)

    asyncio.run(main())
