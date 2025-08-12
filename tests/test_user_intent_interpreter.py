import importlib.util
import unittest
from pathlib import Path

# Load the interpreter directly to avoid importing heavy dependencies from the
# parent `agents` package during test collection.
module_path = (
    Path(__file__).resolve().parents[1]
    / "agents"
    / "sage"
    / "user_intent_interpreter.py"
)
spec = importlib.util.spec_from_file_location("user_intent_interpreter", module_path)
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)  # type: ignore
UserIntentInterpreter = ui.UserIntentInterpreter


class TestUserIntentInterpreter(unittest.IsolatedAsyncioTestCase):
    async def test_search_intent(self):
        interpreter = UserIntentInterpreter()
        result = await interpreter.interpret_intent("search for python tutorials")
        assert result["type"] == "search"
        assert result["confidence"] >= 0.9

    async def test_summarize_intent(self):
        interpreter = UserIntentInterpreter()
        result = await interpreter.interpret_intent("summarize the rust book")
        assert result["type"] == "summarize"
        assert "rust book" in result["topic"]

    async def test_unknown_intent(self):
        interpreter = UserIntentInterpreter()
        result = await interpreter.interpret_intent("Tell me something interesting")
        assert result["type"] == "unknown"
        assert result["confidence"] < 0.9


if __name__ == "__main__":
    unittest.main()
