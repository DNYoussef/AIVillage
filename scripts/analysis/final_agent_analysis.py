#!/usr/bin/env python3
"""Final comprehensive analysis of AIVillage agent system."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd()))


class AgentAnalyzer:
    """Comprehensive agent system analyzer."""

    def __init__(self) -> None:
        self.results = {}

    def analyze_templates(self):
        """Analyze agent templates."""
        print("=" * 60)
        print("TEMPLATE ANALYSIS")
        print("=" * 60)

        try:
            from src.production.agent_forge.agent_factory import AgentFactory

            # Use correct path
            template_path = (
                Path.cwd() / "src" / "production" / "agent_forge" / "templates"
            )
            factory = AgentFactory(str(template_path))

            templates = factory.list_available_agents()

            print(f"Found {len(templates)} agent templates:")
            for agent in templates:
                print(f"  {agent['id']}: {agent['name']}")
                print(f"    Role: {agent['role']}")

            self.results["templates"] = {"count": len(templates), "agents": templates}

            return factory

        except Exception as e:
            print(f"Error analyzing templates: {e}")
            self.results["templates"] = {"error": str(e)}
            return None

    def analyze_implementations(self, factory):
        """Analyze agent implementations (real vs stub)."""
        print("\n" + "=" * 60)
        print("IMPLEMENTATION ANALYSIS")
        print("=" * 60)

        if not factory:
            print("No factory available")
            return None

        real_implementations = 0
        stub_implementations = 0
        failed_creations = 0

        agent_details = {}

        for template in factory.templates:
            try:
                agent = factory.create_agent(template)
                class_name = agent.__class__.__name__

                is_real = class_name != "GenericAgent"

                if is_real:
                    real_implementations += 1
                    status = "REAL"
                else:
                    stub_implementations += 1
                    status = "STUB"

                print(f"  {template}: {status} ({class_name})")

                agent_details[template] = {
                    "created": True,
                    "class": class_name,
                    "is_real": is_real,
                    "has_name": hasattr(agent, "name"),
                    "has_config": hasattr(agent, "config"),
                    "has_specialization": hasattr(agent, "specialization"),
                }

            except Exception as e:
                failed_creations += 1
                print(f"  {template}: FAILED - {str(e)[:50]}")
                agent_details[template] = {"created": False, "error": str(e)}

        print("\nSummary:")
        print(f"  Real implementations: {real_implementations}")
        print(f"  Stub implementations: {stub_implementations}")
        print(f"  Failed creations: {failed_creations}")

        self.results["implementations"] = {
            "real": real_implementations,
            "stubs": stub_implementations,
            "failed": failed_creations,
            "details": agent_details,
        }

        return factory

    def test_behavioral_differences(self, factory) -> None:
        """Test if agents show different behaviors."""
        print("\n" + "=" * 60)
        print("BEHAVIORAL ANALYSIS")
        print("=" * 60)

        if not factory:
            print("No factory available")
            return

        # Test with different types of tasks
        test_tasks = [
            {"task": "coordinate", "data": "strategy meeting"},
            {"task": "research", "data": "machine learning trends"},
            {"task": "create", "data": "user interface design"},
            {"task": "analyze", "data": "system performance"},
        ]

        test_agents = ["king", "sage", "magi", "maker", "auditor"]

        behavioral_differences = 0

        for task in test_tasks:
            print(f"\nTask: {task['task']} - {task['data']}")
            responses = {}

            for agent_id in test_agents:
                if agent_id in factory.templates:
                    try:
                        agent = factory.create_agent(agent_id)
                        result = agent.process(task)
                        responses[agent_id] = result
                        print(f"  {agent_id}: {result.get('result', 'no result')[:50]}")
                    except Exception as e:
                        print(f"  {agent_id}: ERROR - {str(e)[:30]}")

            # Check if responses are different
            unique_responses = {str(r) for r in responses.values()}
            if len(unique_responses) > 1:
                behavioral_differences += 1
                print(f"    -> {len(unique_responses)} different responses")
            else:
                print("    -> All responses identical")

        print(
            f"\nBehavioral differences found: {behavioral_differences}/{len(test_tasks)} tasks"
        )

        self.results["behavior"] = {
            "tasks_tested": len(test_tasks),
            "different_behaviors": behavioral_differences,
            "agents_tested": test_agents,
        }

    def test_kpi_system(self, factory) -> None:
        """Test KPI tracking system."""
        print("\n" + "=" * 60)
        print("KPI SYSTEM ANALYSIS")
        print("=" * 60)

        if not factory:
            print("No factory available")
            return

        try:
            agent = factory.create_agent("king")

            # Test initial KPI
            initial_kpi = agent.evaluate_kpi()
            print(f"Initial KPI: {initial_kpi}")

            # Update performance
            agent.update_performance(
                {
                    "timestamp": "2024-01-01T12:00:00",
                    "success": True,
                    "metrics": {"accuracy": 0.85},
                }
            )

            # Test updated KPI
            updated_kpi = agent.evaluate_kpi()
            print(f"Updated KPI: {updated_kpi}")

            # Check performance history
            print(f"Performance history entries: {len(agent.performance_history)}")

            kpi_working = bool(updated_kpi) and bool(agent.performance_history)
            print(f"KPI system working: {kpi_working}")

            self.results["kpi"] = {
                "working": kpi_working,
                "initial_kpi": initial_kpi,
                "updated_kpi": updated_kpi,
                "history_length": len(agent.performance_history),
            }

        except Exception as e:
            print(f"KPI test failed: {e}")
            self.results["kpi"] = {"working": False, "error": str(e)}

    def test_communication_system(self) -> None:
        """Test communication system."""
        print("\n" + "=" * 60)
        print("COMMUNICATION SYSTEM ANALYSIS")
        print("=" * 60)

        try:
            from src.communications.message import Message, MessageType
            from src.communications.protocol import CommunicationsProtocol

            # Test protocol creation
            CommunicationsProtocol("test_agent")
            print("+ Communication protocol created")

            # Test message creation
            message = Message(
                type=MessageType.TASK,
                sender="agent_a",
                receiver="agent_b",
                content={"task": "collaborate", "data": "test"},
            )
            print("+ Message created")
            print(f"  Message type: {message.type}")

            self.results["communication"] = {
                "protocol_works": True,
                "message_works": True,
            }

        except Exception as e:
            print(f"- Communication test failed: {e}")
            self.results["communication"] = {
                "protocol_works": False,
                "message_works": False,
                "error": str(e),
            }

    def check_experimental_agents(self) -> None:
        """Check experimental agent implementations."""
        print("\n" + "=" * 60)
        print("EXPERIMENTAL AGENTS ANALYSIS")
        print("=" * 60)

        exp_path = Path("experimental/agents/agents")
        if not exp_path.exists():
            print("No experimental agents directory")
            return

        experimental_agents = {}

        # Check known agent directories
        known_agents = ["king", "magi", "sage"]

        for agent_name in known_agents:
            agent_dir = exp_path / agent_name
            if agent_dir.exists():
                # Find main agent file
                agent_files = list(agent_dir.glob("*_agent.py"))

                if agent_files:
                    main_file = agent_files[0]
                    file_size = main_file.stat().st_size

                    # Try to analyze complexity
                    try:
                        with open(main_file, encoding="utf-8") as f:
                            content = f.read()

                        lines = content.count("\n")
                        classes = content.count("class ")
                        functions = content.count("def ")
                        imports = content.count("import ") + content.count("from ")

                        complexity_score = (
                            (classes * 10) + (functions * 2) + (imports * 1)
                        )

                        print(f"{agent_name}:")
                        print(f"  File: {main_file.name} ({file_size} bytes)")
                        print(f"  Lines: {lines}")
                        print(f"  Classes: {classes}")
                        print(f"  Functions: {functions}")
                        print(f"  Complexity score: {complexity_score}")

                        experimental_agents[agent_name] = {
                            "has_implementation": True,
                            "file_size": file_size,
                            "lines": lines,
                            "classes": classes,
                            "functions": functions,
                            "complexity": complexity_score,
                        }

                    except Exception as e:
                        print(f"{agent_name}: Error analyzing - {e}")
                        experimental_agents[agent_name] = {
                            "has_implementation": True,
                            "error": str(e),
                        }
                else:
                    print(f"{agent_name}: No main agent file")
                    experimental_agents[agent_name] = {"has_implementation": False}
            else:
                print(f"{agent_name}: Directory not found")
                experimental_agents[agent_name] = {"has_implementation": False}

        self.results["experimental"] = experimental_agents

    def generate_report(self) -> None:
        """Generate comprehensive final report."""
        print("\n" + "=" * 60)
        print("FINAL COMPREHENSIVE REPORT")
        print("=" * 60)

        # Template analysis
        template_count = self.results.get("templates", {}).get("count", 0)
        print(f"Agent Templates: {template_count}/18 expected")

        # Implementation analysis
        impl = self.results.get("implementations", {})
        real_count = impl.get("real", 0)
        stub_count = impl.get("stubs", 0)

        print(f"Real Implementations: {real_count}/{template_count}")
        print(f"Stub Implementations: {stub_count}/{template_count}")

        # Behavioral analysis
        behavior = self.results.get("behavior", {})
        behavioral_diff = behavior.get("different_behaviors", 0)
        tasks_tested = behavior.get("tasks_tested", 0)

        print(f"Behavioral Differences: {behavioral_diff}/{tasks_tested} tasks")

        # KPI system
        kpi = self.results.get("kpi", {})
        kpi_working = kpi.get("working", False)
        print(f"KPI System: {'Working' if kpi_working else 'Not Working'}")

        # Communication system
        comm = self.results.get("communication", {})
        comm_working = comm.get("protocol_works", False) and comm.get(
            "message_works", False
        )
        print(f"Communication System: {'Working' if comm_working else 'Not Working'}")

        # Experimental agents
        exp = self.results.get("experimental", {})
        exp_count = sum(1 for v in exp.values() if v.get("has_implementation", False))
        print(f"Experimental Implementations: {exp_count}/3 checked")

        # Overall assessment
        print(f"\n{'=' * 20} VERDICT {'=' * 20}")

        if real_count > 0:
            print("VERDICT: Some real agent implementations exist")
        else:
            print("VERDICT: All agents are generic stubs")

        if behavioral_diff > 0:
            print("BEHAVIOR: Agents show some different behaviors")
        else:
            print("BEHAVIOR: All agents behave identically")

        if comm_working and kpi_working:
            print("SYSTEMS: Core systems (KPI, Communication) functional")
        else:
            print("SYSTEMS: Core systems have issues")

        # Inter-agent coordination assessment
        can_coordinate = (
            template_count >= 18
            and comm_working
            and kpi_working
            and (real_count > 0 or behavioral_diff > 0)
        )

        print(f"\nCAN COORDINATE 18 AGENTS: {'YES' if can_coordinate else 'NO'}")

        if not can_coordinate:
            issues = []
            if template_count < 18:
                issues.append("Missing agent templates")
            if not comm_working:
                issues.append("Communication system broken")
            if not kpi_working:
                issues.append("KPI system broken")
            if real_count == 0 and behavioral_diff == 0:
                issues.append("No differentiated behaviors")

            print(f"ISSUES: {', '.join(issues)}")


def main() -> None:
    """Run comprehensive agent analysis."""
    print("AIVillage Agent System - COMPREHENSIVE ANALYSIS")

    analyzer = AgentAnalyzer()

    # Run all analyses
    factory = analyzer.analyze_templates()
    analyzer.analyze_implementations(factory)
    analyzer.test_behavioral_differences(factory)
    analyzer.test_kpi_system(factory)
    analyzer.test_communication_system()
    analyzer.check_experimental_agents()

    # Generate final report
    analyzer.generate_report()


if __name__ == "__main__":
    main()
