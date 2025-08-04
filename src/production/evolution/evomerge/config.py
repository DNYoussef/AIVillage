import os

from pydantic import BaseModel, Field, validator


class ModelDomain(BaseModel):
    name: str
    architecture: str
    task_type: str


class ModelReference(BaseModel):
    name: str
    path: str
    domain: ModelDomain | None = None


class MergeSettings(BaseModel):
    merge_method: str
    parameters: dict[str, float | list[float] | dict[str, float | list[float]]] = Field(default_factory=dict)
    custom_dir: str = Field(default="./merged_models")
    ps_techniques: list[str] = ["linear"]
    dfs_techniques: list[str] = []
    use_8bit: bool = Field(default=False)
    use_4bit: bool = Field(default=False)
    cross_domain_strategy: str = Field(default="adapter")
    instruction_tuning_preservation: str = Field(
        default="max",
        description="Strategy for preserving instruction tuning: 'max' or 'mean'",
    )
    weight_mask_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    use_weight_rescale: bool = Field(default=True)
    mask_strategy: str = Field(default="random")
    use_disk_based_merge: bool = Field(default=True)
    chunk_size: int = Field(default=1000000)

    @validator("merge_method")
    def validate_merge_method(self, v):
        valid_methods = ["ps", "dfs", "ps_dfs"]
        if v not in valid_methods:
            msg = f"Invalid merge method. Choose from: {', '.join(valid_methods)}"
            raise ValueError(msg)
        return v

    @validator("ps_techniques", "dfs_techniques")
    def validate_techniques(self, v):
        valid_techniques = [
            "linear",
            "slerp",
            "ties",
            "dare",
            "task_arithmetic",
            "frankenmerge",
            "dfs",
        ]
        for technique in v:
            if technique not in valid_techniques:
                msg = f"Invalid technique: {technique}. Choose from: {', '.join(valid_techniques)}"
                raise ValueError(msg)
        return v

    @validator("custom_dir")
    def validate_custom_dir(self, v):
        if not os.path.exists(v):
            os.makedirs(v, exist_ok=True)
        return v

    @validator("cross_domain_strategy")
    def validate_cross_domain_strategy(self, v):
        valid_strategies = ["adapter", "embedding_only", "full"]
        if v not in valid_strategies:
            msg = f"Invalid cross-domain strategy. Choose from: {', '.join(valid_strategies)}"
            raise ValueError(msg)
        return v

    @validator("mask_strategy")
    def validate_mask_strategy(self, v):
        valid_strategies = ["random", "magnitude"]
        if v not in valid_strategies:
            msg = f"Invalid mask strategy. Choose from: {', '.join(valid_strategies)}"
            raise ValueError(msg)
        return v


class EvolutionSettings(BaseModel):
    population_size: int = Field(default=8, ge=2)
    num_generations: int = Field(default=50, ge=1)
    mutation_rate: float = Field(default=0.1, ge=0, le=1)
    tournament_size: int = Field(default=3, ge=2)
    early_stopping_generations: int = Field(default=10, ge=1)
    use_cma_es: bool = Field(default=False)
    adaptive_mutation: bool = Field(default=True)
    objectives: list[str] = Field(default=["overall_score", "perplexity"])

    @validator("tournament_size")
    def validate_tournament_size(self, v, values):
        if "population_size" in values and v > values["population_size"]:
            msg = "Tournament size must be less than or equal to population size"
            raise ValueError(msg)
        return v

    @validator("objectives")
    def validate_objectives(self, v):
        valid_objectives = ["overall_score", "perplexity", "accuracy", "coherence"]
        for obj in v:
            if obj not in valid_objectives:
                msg = f"Invalid objective: {obj}. Choose from: {', '.join(valid_objectives)}"
                raise ValueError(msg)
        return v


class Configuration(BaseModel):
    models: list[ModelReference]
    merge_settings: MergeSettings
    evolution_settings: EvolutionSettings
    target_domain: ModelDomain | None = None
    enable_adas: bool = Field(default=True, description="Run ADAS optimization after training")


def create_default_config() -> Configuration:
    return Configuration(
        models=[
            ModelReference(name="Qwen2.5-1.5B-Instruct", path="Qwen/Qwen2.5-1.5B-Instruct"),
            ModelReference(
                name="Qwen2.5-Coder-1.5B-Instruct",
                path="Qwen/Qwen2.5-Coder-1.5B-Instruct",
            ),
            ModelReference(
                name="Qwen2.5-Math-1.5B-Instruct",
                path="Qwen/Qwen2.5-Math-1.5B-Instruct",
            ),
        ],
        merge_settings=MergeSettings(
            merge_method="ps_dfs",
            parameters={
                "linear": {"weights": [1 / 3, 1 / 3, 1 / 3]},
                "slerp": {"t": 0.5},
                "ties": {"threshold": 0.1},
                "dare": {"threshold": 0.1, "amplification": 2.0},
            },
            ps_techniques=["linear", "ties"],
            dfs_techniques=["frankenmerge"],
            use_8bit=False,
            use_4bit=False,
            cross_domain_strategy="adapter",
            weight_mask_rate=0.0,
            use_weight_rescale=True,
            mask_strategy="random",
            use_disk_based_merge=True,
            chunk_size=1000000,
        ),
        evolution_settings=EvolutionSettings(
            use_cma_es=False,
            adaptive_mutation=True,
            objectives=["overall_score", "perplexity"],
        ),
        enable_adas=True,
    )


if __name__ == "__main__":
    # Test the configuration
    config = create_default_config()
    print(config.json(indent=2))
