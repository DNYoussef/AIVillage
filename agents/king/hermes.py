from ai_village.core.king.main import King

def hermes(uuid: str):
    """The orchestrator that interacts with the user to understand goals, plan out how agents can meet the goal, assign tasks, and coordinate the activities agents."""
    king = King()
    return king.run(uuid)