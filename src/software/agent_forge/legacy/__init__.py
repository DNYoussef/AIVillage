"""AIVillage Agent Forge - Production Components.

Stable agent_forge components that are production-ready:
- core: Core agent functionality
- evaluation: Agent evaluation systems
- deployment: Production deployment tools
- utils: Utility functions and helpers
- orchestration: Agent orchestration systems

Experimental components have been moved to: experimental/agent_forge_experimental/
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
__version__ = "1.0.0"
__status__ = "Production"

__all__ = [
    "AgentForge",
    "core",
    "deployment",
    "evaluation",
    "orchestration",
    "utils",
    "compression",
]


def __getattr__(name):
    """Lazy module attribute loading for production components."""
    if name == "adas":
        from . import adas

        return adas
    if name == "expert_vectors":
        from .utils import expert_vector

        return expert_vector
    if name == "tool_baking":
        from . import tool_baking

        return tool_baking
    msg = f"module '{__name__}' has no attribute '{name}'"
    raise AttributeError(msg)


class AgentForge:
    """Production-ready AgentForge with lazy loading and full functionality."""

    def __init__(self, model_name: str = "gpt2") -> None:
        """Initialize AgentForge with model configuration.

        Args:
            model_name: Name of the model to use for prompt baking.
        """
        self.model_name = model_name
        self._evolution_tournament = None
        self._training_task = None
        self._prompt_baker = None
        self._config = None

    @property
    def config(self):
        """Lazy-loaded config property."""
        if self._config is None:
            try:
                from .orchestration import config

                self._config = config.create_default_config()
            except ImportError:
                msg = "orchestration module not available"
                raise ImportError(msg)
        return self._config

    @property
    def evolution_tournament(self):
        """Lazy-loaded evolution tournament property."""
        if self._evolution_tournament is None:
            try:
                from .evolution import evolution_orchestrator

                self._evolution_tournament = evolution_orchestrator.EvolutionOrchestrator(self.config)
            except ImportError:
                msg = "evolution module not available"
                raise ImportError(msg)
        return self._evolution_tournament

    @property
    def training_task(self):
        """Lazy-loaded training task property."""
        if self._training_task is None:
            from .training.training import TrainingTask

            self._training_task = TrainingTask(None)
        return self._training_task

    @property
    def prompt_baker(self):
        """Lazy-loaded prompt baker property."""
        if self._prompt_baker is None:
            from .tool_baking import rag_prompt_baker

            self._prompt_baker = rag_prompt_baker.RAGPromptBaker(self.model_name)
        return self._prompt_baker

    def run_evolution_tournament(self) -> object:
        """Run the evolution tournament and return the best model."""
        return self.evolution_tournament.evolve()

    def run_training(self) -> None:
        """Invoke the training task if an agent is configured."""
        if hasattr(self.training_task, "run_training_loop"):
            self.training_task.run_training_loop()

    def run_prompt_baking(self) -> None:
        """Run prompt baking process."""
        self.prompt_baker.load_model()
        prompts = getattr(self.prompt_baker, "get_rag_prompts", lambda: [])()
        self.prompt_baker.bake_prompts(prompts)

    def run_full_agent_forge_process(self) -> object:
        """Run the complete AgentForge process."""
        best_model = self.run_evolution_tournament()
        self.run_training()
        self.run_prompt_baking()
        print("Agent Forge process completed.")
        return best_model
