from agent_forge.evomerge.evolutionary_tournament import EvolutionaryTournament
from agent_forge.training.training import Trainer
from agent_forge.tool_baking.rag_prompt_baker import RAGPromptBaker
# from agent_forge.adas.adas_process import ADASProcess  # Module not found in project structure

__all__ = [
    "AgentForge",
    "EvolutionaryTournament",
    "Trainer",
    "RAGPromptBaker",
    # "ADASProcess",
]

class AgentForge:
    def __init__(self):
        self.evolution_tournament = EvolutionaryTournament()
        self.trainer = Trainer()
        self.prompt_baker = RAGPromptBaker()
        # self.adas_process = ADASProcess()  # Commented out since module doesn't exist

    def run_evolution_tournament(self):
        self.evolution_tournament.run()

    def run_training(self):
        self.trainer.start_training()

    def run_prompt_baking(self):
        self.prompt_baker.bake_prompts()

    # def run_adas_process(self):
    #     self.adas_process.run()

    def run_full_agent_forge_process(self):
        self.run_evolution_tournament()
        self.run_training()
        self.run_prompt_baking()
        # self.run_adas_process()
        print("Agent Forge process completed.")

agent_forge = AgentForge()
