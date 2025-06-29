try:
    from . import evomerge
except Exception:  # pragma: no cover - optional heavy deps may be missing
    evomerge = None
# Re-export key subsystems so external modules can rely on them
from .training.training import TrainingTask
from .training import expert_vectors
from . import tool_baking, adas
# Import ADAS utilities if available. The ADAS module is optional and may
# require additional heavy dependencies.
from . import adas

__all__ = [
    "AgentForge",
    "adas",
    "expert_vectors",
]

class AgentForge:
    def __init__(self, model_name="gpt2"):
        config = evomerge.create_default_config()
        self.evolution_tournament = evomerge.EvolutionaryTournament(config)
        self.training_task = TrainingTask(None)  # Note: We're passing None as the agent, you might need to adjust this
        self.prompt_baker = tool_baking.RAGPromptBaker(model_name)
        # Optional: instantiate ADASProcess if the dependencies are installed.
        # self.adas_process = adas.ADASProcess()

    def run_evolution_tournament(self):
        return self.evolution_tournament.evolve()

    def run_training(self):
        """Invoke the training task if an agent is configured."""
        if hasattr(self.training_task, "run_training_loop"):
            self.training_task.run_training_loop()
        
    def run_prompt_baking(self):
        self.prompt_baker.load_model()  # Explicitly load the model
        self.prompt_baker.bake_prompts(tool_baking.get_rag_prompts())

    # def run_adas_process(self):
    #     """Invoke the ADAS processing pipeline if initialized."""
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
