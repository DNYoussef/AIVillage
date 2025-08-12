"""Comprehensive integration readiness tests."""

import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


class IntegrationAuditor:
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "sections": {},
            "scores": {},
            "issues": {"P0": [], "P1": [], "P2": [], "P3": []},
            "summary": {},
        }

    def test_hyperrag(self):
        """Test HyperRAG vector+graph integration."""
        section = {"tests": [], "score": 0}

        try:
            from src.production.rag.rag_system.core.pipeline import RAGPipeline

            section["tests"].append({"name": "RAG import", "status": "PASS"})

            # Test instantiation
            try:
                pipeline = RAGPipeline()
                section["tests"].append({"name": "Pipeline creation", "status": "PASS"})
                section["score"] += 50
            except Exception as e:
                section["tests"].append(
                    {"name": "Pipeline creation", "status": "FAIL", "error": str(e)}
                )
                self.results["issues"]["P1"].append("RAG pipeline instantiation fails")

        except ImportError as e:
            section["tests"].append(
                {"name": "RAG import", "status": "FAIL", "error": str(e)}
            )
            self.results["issues"]["P0"].append("RAG system cannot be imported")

        self.results["sections"]["hyperrag"] = section
        return section["score"]

    def test_agent_forge(self):
        """Test Agent Forge and EvoMerge."""
        section = {"tests": [], "score": 0}

        try:
            from agent_forge.core import AgentForge

            section["tests"].append({"name": "Agent Forge import", "status": "PASS"})
            section["score"] += 30

            # Test evolution
            try:
                from agent_forge.evolution import EvoMerge

                section["tests"].append({"name": "EvoMerge import", "status": "PASS"})
                section["score"] += 30
            except ImportError:
                section["tests"].append({"name": "EvoMerge import", "status": "WARN"})
                self.results["issues"]["P2"].append("EvoMerge not available")

        except ImportError as e:
            section["tests"].append(
                {"name": "Agent Forge import", "status": "FAIL", "error": str(e)}
            )
            self.results["issues"]["P1"].append("Agent Forge system missing")

        self.results["sections"]["agent_forge"] = section
        return section["score"]

    def test_tokenomics(self):
        """Test tokenomics/DAO functionality."""
        section = {"tests": [], "score": 0}

        try:
            from src.token_economy.credit_system import VILLAGECreditSystem

            section["tests"].append({"name": "Credit system import", "status": "PASS"})

            # Test basic functionality
            try:
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
                    system = VILLAGECreditSystem(f.name)
                    system.update_balance("test_user", 100)
                    balance = system.get_balance("test_user")

                    if balance == 100:
                        section["tests"].append(
                            {"name": "Balance operations", "status": "PASS"}
                        )
                        section["score"] += 60
                    else:
                        section["tests"].append(
                            {"name": "Balance operations", "status": "FAIL"}
                        )
                        self.results["issues"]["P1"].append(
                            "Credit system balance tracking broken"
                        )

                    system.close()
                    os.unlink(f.name)

            except Exception as e:
                section["tests"].append(
                    {"name": "Credit system ops", "status": "FAIL", "error": str(e)}
                )
                self.results["issues"]["P1"].append(f"Credit system error: {e}")

        except ImportError as e:
            section["tests"].append(
                {"name": "Tokenomics import", "status": "FAIL", "error": str(e)}
            )
            self.results["issues"]["P2"].append("Tokenomics system not found")

        self.results["sections"]["tokenomics"] = section
        return section["score"]

    def test_dual_path_comms(self):
        """Test BitChat/Betanet dual-path communications."""
        section = {"tests": [], "score": 0}

        # Test BitChat
        try:
            from core.p2p.bitchat_transport import BitChatTransport

            section["tests"].append({"name": "BitChat import", "status": "PASS"})
            section["score"] += 30
        except ImportError:
            section["tests"].append({"name": "BitChat import", "status": "FAIL"})
            self.results["issues"]["P0"].append("BitChat transport missing")

        # Test Betanet
        try:
            from core.p2p.betanet_transport import BetanetTransport

            section["tests"].append({"name": "Betanet import", "status": "PASS"})
            section["score"] += 30
        except ImportError:
            section["tests"].append({"name": "Betanet import", "status": "FAIL"})
            self.results["issues"]["P0"].append("Betanet transport missing")

        # Test integration
        try:
            from core.p2p.dual_path_transport import DualPathTransport

            transport = DualPathTransport()
            section["tests"].append({"name": "Dual-path integration", "status": "PASS"})
            section["score"] += 40
        except Exception as e:
            section["tests"].append(
                {"name": "Dual-path integration", "status": "FAIL", "error": str(e)}
            )
            self.results["issues"]["P0"].append("Dual-path transport broken")

        self.results["sections"]["comms"] = section
        return section["score"]

    def test_agents(self):
        """Test 18 specialist agents."""
        section = {"tests": [], "score": 0, "agents": []}

        # Check experimental agents
        import os

        agent_dir = "experimental/agents"
        if os.path.exists(agent_dir):
            agents = [
                d
                for d in os.listdir(agent_dir)
                if os.path.isdir(os.path.join(agent_dir, d)) and not d.startswith("_")
            ]

            section["agents"] = agents
            agent_count = len(agents)

            if agent_count >= 18:
                section["tests"].append(
                    {"name": "Agent count", "status": "PASS", "count": agent_count}
                )
                section["score"] += 80
            elif agent_count >= 9:
                section["tests"].append(
                    {"name": "Agent count", "status": "WARN", "count": agent_count}
                )
                section["score"] += 40
                self.results["issues"]["P2"].append(
                    f"Only {agent_count}/18 agents found"
                )
            else:
                section["tests"].append(
                    {"name": "Agent count", "status": "FAIL", "count": agent_count}
                )
                self.results["issues"]["P1"].append(
                    f"Critical: Only {agent_count}/18 agents"
                )
        else:
            section["tests"].append({"name": "Agent directory", "status": "FAIL"})
            self.results["issues"]["P1"].append("Agent directory not found")

        self.results["sections"]["agents"] = section
        return section["score"]

    def test_mobile_readiness(self):
        """Test mobile/on-device support."""
        section = {"tests": [], "score": 0}

        try:
            from production.monitoring.mobile.resource_management import (
                BatteryThermalResourceManager,
            )

            section["tests"].append({"name": "Mobile resource mgmt", "status": "PASS"})
            section["score"] += 70
        except ImportError:
            # Check if module exists but has import issues
            import os

            if os.path.exists(
                "src/production/monitoring/mobile/resource_management.py"
            ):
                section["tests"].append(
                    {
                        "name": "Mobile resource mgmt",
                        "status": "WARN",
                        "note": "Module exists but has import issues",
                    }
                )
                section["score"] += 40
                self.results["issues"]["P2"].append(
                    "Mobile resource management has import issues"
                )
            else:
                section["tests"].append(
                    {"name": "Mobile resource mgmt", "status": "FAIL"}
                )
                self.results["issues"]["P3"].append("Mobile support not implemented")

        self.results["sections"]["mobile"] = section
        return section["score"]

    def test_security(self):
        """Test security improvements."""
        section = {"tests": [], "score": 0}

        # Check for pickle usage
        import subprocess

        result = subprocess.run(
            ["git", "grep", "-l", "pickle.load", "src/"],
            check=False,
            capture_output=True,
            text=True,
        )

        pickle_files = [
            f for f in result.stdout.strip().split("\n") if f and "test" not in f
        ]

        if not pickle_files:
            section["tests"].append({"name": "No unsafe pickle", "status": "PASS"})
            section["score"] += 40
        else:
            section["tests"].append(
                {"name": "Pickle usage", "status": "FAIL", "files": len(pickle_files)}
            )
            self.results["issues"]["P0"].append(
                f"Unsafe pickle in {len(pickle_files)} files"
            )

        # Check HTTP in production
        result = subprocess.run(
            ["git", "grep", "http://", "src/production"],
            check=False,
            capture_output=True,
            text=True,
        )

        http_lines = [
            l
            for l in result.stdout.strip().split("\n")
            if l and "localhost" not in l and "AIVILLAGE_ENV" not in l
        ]

        if not http_lines:
            section["tests"].append({"name": "No HTTP in prod", "status": "PASS"})
            section["score"] += 60
        else:
            section["tests"].append(
                {
                    "name": "HTTP in production",
                    "status": "FAIL",
                    "count": len(http_lines),
                }
            )
            self.results["issues"]["P0"].append(
                f"HTTP endpoints in {len(http_lines)} places"
            )

        self.results["sections"]["security"] = section
        return section["score"]

    def calculate_final_score(self):
        """Calculate weighted final score."""
        weights = {
            "security": 0.25,
            "comms": 0.15,
            "hyperrag": 0.15,
            "agent_forge": 0.15,
            "agents": 0.10,
            "tokenomics": 0.10,
            "mobile": 0.10,
        }

        total_score = 0
        for section, weight in weights.items():
            if section in self.results["sections"]:
                score = self.results["sections"][section].get("score", 0)
                total_score += score * weight

        self.results["scores"]["total"] = round(total_score, 1)
        self.results["scores"]["breakdown"] = {
            s: self.results["sections"][s].get("score", 0)
            for s in self.results["sections"]
        }

        return total_score

    def generate_summary(self):
        """Generate audit summary."""
        p0_count = len(self.results["issues"]["P0"])
        p1_count = len(self.results["issues"]["P1"])
        total_score = self.results["scores"]["total"]

        self.results["summary"] = {
            "p0_issues": p0_count,
            "p1_issues": p1_count,
            "total_score": total_score,
            "readiness": "BLOCKED"
            if p0_count > 0
            else ("READY" if total_score >= 80 else "NOT_READY"),
            "top_risks": self.results["issues"]["P0"][:3]
            + self.results["issues"]["P1"][:2],
        }

        return self.results["summary"]

    def run_audit(self):
        """Run complete audit."""
        print("Running integration readiness audit...")

        self.test_security()
        self.test_dual_path_comms()
        self.test_hyperrag()
        self.test_agent_forge()
        self.test_agents()
        self.test_tokenomics()
        self.test_mobile_readiness()

        self.calculate_final_score()
        self.generate_summary()

        return self.results


def main():
    auditor = IntegrationAuditor()
    results = auditor.run_audit()

    # Save JSON
    with open("../../ai_village_integration_readiness.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("INTEGRATION READINESS SUMMARY")
    print("=" * 60)
    print(f"Total Score: {results['scores']['total']}/100")
    print(f"P0 Issues: {results['summary']['p0_issues']}")
    print(f"P1 Issues: {results['summary']['p1_issues']}")
    print(f"Status: {results['summary']['readiness']}")

    if results["summary"]["top_risks"]:
        print("\nTop Risks:")
        for risk in results["summary"]["top_risks"]:
            print(f"  - {risk}")

    print("\nDetailed results saved to ai_village_integration_readiness.json")

    return results["summary"]["readiness"] == "READY"


if __name__ == "__main__":
    ready = main()
    exit(0 if ready else 1)
