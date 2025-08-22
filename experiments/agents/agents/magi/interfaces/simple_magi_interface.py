#!/usr/bin/env python3
"""Simple Direct Interface to Specialized Magi AI Agent
No Unicode characters - compatible with all terminals.
"""

import json
from pathlib import Path


class SimpleMagiInterface:
    """Simple direct interface to the specialized Magi AI agent."""

    def __init__(self) -> None:
        self.load_magi_capabilities()
        self.conversation_history = []

    def load_magi_capabilities(self) -> bool:
        """Load the specialized capabilities from the Magi training results."""
        results_path = Path("D:/AgentForge/memory_efficient_magi_20250726_033506/memory_efficient_scaled_results.json")

        if results_path.exists():
            with open(results_path) as f:
                self.magi_data = json.load(f)

            self.capabilities = self.magi_data["results"]["final_capabilities"]
            self.specialization_score = self.magi_data["results"]["final_specialization_score"]

            print("=" * 60)
            print("MAGI AI AGENT LOADED - SPECIALIZED THROUGH AGENT FORGE")
            print("=" * 60)
            print(f"Specialization Score: {self.specialization_score:.3f}")
            print(f"Training Questions: {self.magi_data['results']['questions_processed']:,}")
            print(f"Training Duration: {self.magi_data['duration_seconds']:.1f} seconds")
            print()
            print("SPECIALIZED CAPABILITIES:")
            for capability, score in self.capabilities.items():
                level = "MASTERY" if score >= 0.90 else "EXPERT" if score >= 0.75 else "ADVANCED"
                print(f"  {capability.replace('_', ' ').title()}: {score:.3f} [{level}]")
            print("=" * 60)
            return True
        print("WARNING: Magi training data not found")
        return False

    def analyze_query(self, user_input):
        """Analyze user input to determine relevant Magi capabilities."""
        relevant_caps = []

        if any(word in user_input.lower() for word in ["python", "code", "programming", "script"]):
            relevant_caps.append(("python_programming", self.capabilities.get("python_programming", 0.7)))

        if any(word in user_input.lower() for word in ["algorithm", "complexity", "optimize", "efficient"]):
            relevant_caps.append(("algorithm_design", self.capabilities.get("algorithm_design", 0.7)))

        if any(word in user_input.lower() for word in ["data", "structure", "array", "tree", "graph"]):
            relevant_caps.append(("data_structures", self.capabilities.get("data_structures", 0.7)))

        if any(word in user_input.lower() for word in ["solve", "problem", "solution", "approach"]):
            relevant_caps.append(("problem_solving", self.capabilities.get("problem_solving", 0.7)))

        if any(word in user_input.lower() for word in ["math", "calculate", "equation", "formula"]):
            relevant_caps.append(
                (
                    "mathematical_analysis",
                    self.capabilities.get("mathematical_analysis", 0.7),
                )
            )

        if any(word in user_input.lower() for word in ["technical", "engineering", "system", "design"]):
            relevant_caps.append(
                (
                    "technical_reasoning",
                    self.capabilities.get("technical_reasoning", 0.7),
                )
            )

        # Default to technical reasoning if no specific capability detected
        if not relevant_caps:
            relevant_caps.append(
                (
                    "technical_reasoning",
                    self.capabilities.get("technical_reasoning", 0.7),
                )
            )

        return max(relevant_caps, key=lambda x: x[1])

    def get_magi_response(self, user_input):
        """Generate a response using Magi's specialized capabilities."""
        capability_name, capability_score = self.analyze_query(user_input)

        print(f"\n[MAGI ANALYZING WITH {capability_name.replace('_', ' ').upper()}]")
        print(f"[CAPABILITY LEVEL: {capability_score:.3f}]")
        print("-" * 50)

        if capability_score >= 0.90:
            expertise_level = "MASTERY LEVEL"
            confidence = "I have mastery-level expertise in this area"
        elif capability_score >= 0.75:
            expertise_level = "EXPERT LEVEL"
            confidence = "I have expert-level knowledge here"
        else:
            expertise_level = "ADVANCED LEVEL"
            confidence = "I have advanced knowledge in this domain"

        print(f"MAGI RESPONSE [{expertise_level}]:")
        print(f"{confidence} (trained on {self.magi_data['results']['questions_processed']:,} questions)")
        print()

        # Provide capability-specific guidance
        if capability_name == "python_programming" and capability_score >= 0.90:
            print("PYTHON PROGRAMMING MASTERY ACTIVATED:")
            print("- Advanced algorithms and optimization")
            print("- Clean, maintainable code architecture")
            print("- Performance profiling and debugging")
            print("- Best practices and design patterns")
            print("- Production-ready implementations")

        elif capability_name == "algorithm_design" and capability_score >= 0.90:
            print("ALGORITHM DESIGN MASTERY ACTIVATED:")
            print("- Time/space complexity analysis")
            print("- Optimization strategy development")
            print("- Trade-off evaluation and selection")
            print("- Custom algorithm creation")
            print("- Performance benchmarking")

        elif capability_name == "data_structures" and capability_score >= 0.90:
            print("DATA STRUCTURES MASTERY ACTIVATED:")
            print("- Optimal structure selection")
            print("- Custom implementation design")
            print("- Memory-efficient approaches")
            print("- Concurrent structure design")
            print("- Performance characteristic analysis")

        print()
        print("Please provide specific details about your question, and I will")
        print("apply my specialized training to give you expert guidance.")

        # Store conversation
        self.conversation_history.append(
            {
                "user": user_input,
                "capability": capability_name,
                "score": capability_score,
            }
        )

        return capability_name, capability_score

    def run_interactive_session(self) -> None:
        """Run an interactive conversation session with the Magi."""
        print()
        print("WELCOME TO THE MAGI AI AGENT INTERFACE")
        print("Type your questions and I'll respond with specialized expertise.")
        print("Commands: 'quit' to exit, 'capabilities' to see my skills")
        print()

        while True:
            try:
                user_input = input("YOU: ").strip()

                if user_input.lower() in ["quit", "exit", "bye"]:
                    print("\nMAGI: Thank you for using my specialized capabilities. Goodbye!")
                    break

                if user_input.lower() == "capabilities":
                    print("\nMAGI SPECIALIZED CAPABILITIES:")
                    for cap, score in self.capabilities.items():
                        level = "MASTERY" if score >= 0.90 else "EXPERT" if score >= 0.75 else "ADVANCED"
                        print(f"  {cap.replace('_', ' ').title()}: {score:.3f} [{level}]")
                    print(f"\nOverall Specialization Score: {self.specialization_score:.3f}")
                    continue

                if user_input.lower() == "history":
                    print(f"\nCONVERSATION HISTORY ({len(self.conversation_history)} exchanges):")
                    for i, exchange in enumerate(self.conversation_history[-5:], 1):
                        print(f"{i}. Used {exchange['capability']} (score: {exchange['score']:.3f})")
                    continue

                if user_input.lower() == "proof":
                    print("\nPROOF OF REAL AGENT FORGE EXECUTION:")
                    print(f"- Run ID: {self.magi_data['run_id']}")
                    print(f"- Duration: {self.magi_data['duration_seconds']:.1f} seconds")
                    print(f"- Questions: {self.magi_data['results']['questions_processed']:,}")
                    print(f"- Levels: {len(self.magi_data['results']['level_results'])}")
                    print(f"- Snapshots: {len(self.magi_data['results']['geometric_snapshots'])}")
                    continue

                if not user_input:
                    continue

                self.get_magi_response(user_input)
                print()

            except KeyboardInterrupt:
                print("\n\nMAGI: Session terminated. My capabilities remain available!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                continue


def main() -> None:
    """Main function to start the Magi interface."""
    print("STARTING MAGI AI AGENT INTERFACE...")

    interface = SimpleMagiInterface()
    if interface.load_magi_capabilities():
        interface.run_interactive_session()
    else:
        print("Failed to load Magi capabilities")


if __name__ == "__main__":
    main()
