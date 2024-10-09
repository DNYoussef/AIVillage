from . import evomerge
from .training.training import TrainingTask
from . import tool_baking
# from . import adas  # Module not found in project structure

__all__ = [
    "AgentForge",
]

class AgentForge:
    def __init__(self, model_name="gpt2"):
        config = evomerge.create_default_config()
        self.evolution_tournament = evomerge.EvolutionaryTournament(config)
        self.training_task = TrainingTask(None)  # Note: We're passing None as the agent, you might need to adjust this
        self.prompt_baker = tool_baking.RAGPromptBaker(model_name)
        # self.adas_process = adas.ADASProcess()  # Commented out since module doesn't exist

    def run_evolution_tournament(self):
        return self.evolution_tournament.evolve()

    def run_training(self):
        # You might need to adjust this method to work with TrainingTask
        pass

    def run_prompt_baking(self):
        self.prompt_baker.bake_prompts()

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
