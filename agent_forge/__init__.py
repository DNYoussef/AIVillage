# Lazy imports - moved to properties to avoid startup penalty
# Heavy imports are deferred until actually needed

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

# Import ADAS utilities if available. The ADAS module is optional and may
# require additional heavy dependencies.

# Expose main class and compression subpackage
__all__ = [
    "AgentForge",
    "compression",
]

# Lazy module-level properties for backward compatibility
def __getattr__(name):
    """Lazy module attribute loading."""
    if name == "adas":
        from . import adas
        return adas
    elif name == "expert_vectors":
        from .training import expert_vectors  
        return expert_vectors
    elif name == "tool_baking":
        from . import tool_baking
        return tool_baking
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


class AgentForge:
    def __init__(self, model_name: str = "gpt2") -> None:
        """Initialize AgentForge with model configuration.

        Args:
            model_name: Name of the model to use for prompt baking.
        """
        # Store configuration for lazy initialization
        self.model_name = model_name
        self._evolution_tournament = None
        self._training_task = None 
        self._prompt_baker = None
        self._config = None
        # Optional: instantiate ADASProcess if the dependencies are installed.

    @property
    def config(self):
        """Lazy-loaded config property."""
        if self._config is None:
            try:
                from . import evomerge
                self._config = evomerge.create_default_config()
            except ImportError:
                raise ImportError("evomerge module not available - install heavy dependencies")
        return self._config
    
    @property 
    def evolution_tournament(self):
        """Lazy-loaded evolution tournament property."""
        if self._evolution_tournament is None:
            try:
                from . import evomerge
                self._evolution_tournament = evomerge.EvolutionaryTournament(self.config)
            except ImportError:
                raise ImportError("evomerge module not available - install heavy dependencies")
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
            from . import tool_baking
            self._prompt_baker = tool_baking.RAGPromptBaker(self.model_name)
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
        self.prompt_baker.load_model()  # Explicitly load the model
        self.prompt_baker.bake_prompts(tool_baking.get_rag_prompts())

    def run_full_agent_forge_process(self) -> object:
        """Run the complete AgentForge process."""
        best_model = self.run_evolution_tournament()
        self.run_training()
        self.run_prompt_baking()
        print("Agent Forge process completed.")
        return best_model
