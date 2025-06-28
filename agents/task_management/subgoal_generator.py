class SubGoalGenerator:
    async def generate_subgoals(self, description: str, context: dict) -> list[str]:
        return [description]
