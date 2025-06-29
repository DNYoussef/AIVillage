import re
from typing import Dict, Any

class UserIntentInterpreterAgent:
    """Basic interpreter returning a simple intent structure."""

    def interpret_intent(self, query: str) -> Dict[str, Any]:
        """Return a dictionary describing the user's intent.

        The implementation mirrors the simple keyword based patterns described
        in :doc:`docs/system_overview.md <system_overview.md>` so that the
        rest of the pipeline can rely on consistent intent labels.
        """

        lowered = query.lower()
        intent = "unknown"

        patterns = {
            "search": r"\b(search for|look up|find)\b",
            "summarize": r"\b(summarize|summary of)\b",
            "explanation": r"\bexplain\b",
            "analysis": r"\banalyze\b",
            "comparison": r"\bcompare\b",
            "generation": r"\b(generate|write|create)\b",
        }

        for name, pattern in patterns.items():
            if re.search(pattern, lowered):
                intent = name
                break

        confidence = 0.5 if intent == "unknown" else 0.9
        return {
            "type": intent,
            "topic": query,
            "primary_intent": intent,
            "confidence": confidence,
        }
