#!/usr/bin/env python3
"""Safe Magi Interface with Buffer Overflow Protection.

This interface allows conversation with the specialized Magi agent
while preventing buffer overflow issues from large outputs.
"""

from agent_forge.interface_buffer_fix import SafeMagiInterface


class ConversationalMagi(SafeMagiInterface):
    """Enhanced Magi interface with conversation capabilities."""

    def __init__(self) -> None:
        super().__init__()
        self.conversation_history = []
        self.max_history = 10  # Limit history to prevent memory issues

    def analyze_query(self, user_input: str) -> tuple:
        """Analyze user input to determine relevant Magi capabilities."""
        relevant_caps = []

        # Check for capability keywords
        capability_checks = {
            "python_programming": [
                "python",
                "code",
                "programming",
                "script",
                "function",
                "class",
            ],
            "algorithm_design": [
                "algorithm",
                "complexity",
                "optimize",
                "efficient",
                "performance",
            ],
            "data_structures": [
                "data",
                "structure",
                "array",
                "tree",
                "graph",
                "list",
                "dict",
            ],
            "problem_solving": ["solve", "problem", "solution", "approach", "strategy"],
            "mathematical_analysis": [
                "math",
                "calculate",
                "equation",
                "formula",
                "compute",
            ],
            "technical_reasoning": [
                "technical",
                "engineering",
                "system",
                "design",
                "architecture",
            ],
        }

        input_lower = user_input.lower()
        for capability, keywords in capability_checks.items():
            if any(word in input_lower for word in keywords):
                score = self.capabilities.get(capability, 0.7)
                relevant_caps.append((capability, score))

        # Default to technical reasoning if no specific match
        if not relevant_caps:
            relevant_caps.append(
                (
                    "technical_reasoning",
                    self.capabilities.get("technical_reasoning", 0.7),
                )
            )

        # Return the strongest capability
        return max(relevant_caps, key=lambda x: x[1])

    def get_response(self, user_input: str) -> str:
        """Generate Magi response based on specialized capabilities."""
        capability_name, capability_score = self.analyze_query(user_input)

        # Determine expertise level
        if capability_score >= 0.90:
            level = "MASTERY"
            confidence = "With mastery-level expertise"
        elif capability_score >= 0.75:
            level = "EXPERT"
            confidence = "With expert-level knowledge"
        else:
            level = "ADVANCED"
            confidence = "With advanced understanding"

        # Build response
        response_parts = [
            f"\n[MAGI ANALYSIS - {capability_name.replace('_', ' ').upper()}]",
            f"[Capability Level: {capability_score:.3f} - {level}]",
            "-" * 50,
            f"\n{confidence}, I can help you with this.",
        ]

        # Add specific guidance based on capability
        if capability_name == "python_programming" and capability_score >= 0.90:
            response_parts.extend(
                [
                    "\nPYTHON MASTERY AREAS:",
                    "• Advanced algorithms and optimization",
                    "• Clean, maintainable code architecture",
                    "• Performance profiling and debugging",
                    "• Best practices and design patterns",
                ]
            )
        elif capability_name == "algorithm_design" and capability_score >= 0.90:
            response_parts.extend(
                [
                    "\nALGORITHM DESIGN MASTERY:",
                    "• Time/space complexity analysis",
                    "• Optimization strategy development",
                    "• Custom algorithm creation",
                    "• Performance benchmarking",
                ]
            )
        elif capability_name == "data_structures" and capability_score >= 0.90:
            response_parts.extend(
                [
                    "\nDATA STRUCTURES MASTERY:",
                    "• Optimal structure selection",
                    "• Memory-efficient implementations",
                    "• Concurrent data structures",
                    "• Performance analysis",
                ]
            )

        response_parts.append("\nHow can I assist you with this specific question?")

        # Store in history (with size limit)
        self.conversation_history.append(
            {
                "user": user_input,
                "capability": capability_name,
                "score": capability_score,
            }
        )

        # Trim history if too long
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history :]

        return "\n".join(response_parts)

    def show_capabilities(self) -> None:
        """Display Magi capabilities in a clean format."""
        print("\nMAGI SPECIALIZED CAPABILITIES:")
        print("=" * 50)
        for cap, score in sorted(
            self.capabilities.items(), key=lambda x: x[1], reverse=True
        ):
            level = (
                "MASTERY"
                if score >= 0.90
                else "EXPERT"
                if score >= 0.75
                else "ADVANCED"
            )
            cap_name = cap.replace("_", " ").title()
            print(f"  {cap_name:<25} {score:.3f}  [{level}]")
        print(f"\nOverall Specialization Score: {self.specialization_score:.3f}")
        print("=" * 50)

    def run(self) -> None:
        """Run the interactive Magi session."""
        print("\n" + "=" * 60)
        print("MAGI AI AGENT - SPECIALIZED TECHNICAL ASSISTANT")
        print("=" * 60)

        # Load capabilities safely
        if not self.load_magi_data_safely():
            print("Error: Could not load Magi capabilities")
            return

        print("\nCommands:")
        print("  'capabilities' - Show my specialized skills")
        print("  'history' - Show recent conversation topics")
        print("  'quit' - Exit the session")
        print("\nAsk me technical questions and I'll apply my specialized knowledge!")
        print("-" * 60)

        while True:
            try:
                user_input = input("\nYOU: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["quit", "exit", "bye"]:
                    print(
                        "\nMAGI: Thank you for our conversation. My capabilities remain at your service!"
                    )
                    break

                if user_input.lower() == "capabilities":
                    self.show_capabilities()
                    continue

                if user_input.lower() == "history":
                    if self.conversation_history:
                        print(
                            f"\nRECENT TOPICS ({len(self.conversation_history)} exchanges):"
                        )
                        for i, item in enumerate(self.conversation_history[-5:], 1):
                            print(
                                f"  {i}. {item['capability']} (score: {item['score']:.3f})"
                            )
                    else:
                        print("\nNo conversation history yet.")
                    continue

                # Get and display response
                response = self.get_response(user_input)
                print(f"\nMAGI: {response}")

            except KeyboardInterrupt:
                print("\n\nSession interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Let me try to continue...")


def main() -> None:
    """Launch the safe Magi interface."""
    print("Initializing Safe Magi Interface...")
    magi = ConversationalMagi()
    magi.run()


if __name__ == "__main__":
    main()
