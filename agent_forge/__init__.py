try:
    from . import evomerge
except Exception:  # pragma: no cover - optional heavy deps
    evomerge = None
try:
    from .training.training import TrainingTask
except Exception:  # pragma: no cover - optional heavy deps
    TrainingTask = None
try:
    from . import tool_baking
except Exception:  # pragma: no cover - optional heavy deps
    tool_baking = None
# from . import adas  # Module not found in project structure

__all__ = [
    "AgentForge",
]

class AgentForge:
    def __init__(self, model_name="gpt2"):
        if evomerge:
            config = evomerge.create_default_config()
            self.evolution_tournament = evomerge.EvolutionaryTournament(config)
        else:
            config = None
            self.evolution_tournament = None
        if TrainingTask:
            self.training_task = TrainingTask(None)  # Note: We're passing None as the agent, you might need to adjust this
        else:
            self.training_task = None
        self.prompt_baker = tool_baking.RAGPromptBaker(model_name) if tool_baking else None
        # self.adas_process = adas.ADASProcess()  # Commented out since module doesn't exist

    def run_evolution_tournament(self):
        if self.evolution_tournament:
            return self.evolution_tournament.evolve()
        return None

    def run_training(self):
        # You might need to adjust this method to work with TrainingTask
        pass

    def run_prompt_baking(self):
        if self.prompt_baker and tool_baking:
            self.prompt_baker.load_model()  # Explicitly load the model
            self.prompt_baker.bake_prompts(tool_baking.get_rag_prompts())

    # def run_adas_process(self):
    #     self.adas_process.run()

    def run_full_agent_forge_process(self):
        best_model = self.run_evolution_tournament()
        self.run_training()
        self.run_prompt_baking()
        # self.run_adas_process()
        print("Agent Forge process completed.")
        return best_model

# Don't create an instance here, let it be created when needed
# agent_forge = AgentForge()
