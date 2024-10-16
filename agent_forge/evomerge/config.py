from typing import List, Dict, Union
from pydantic import BaseModel, Field, validator

class ModelReference(BaseModel):
    name: str
    path: str  # Hugging Face model ID or local path

class MergeSettings(BaseModel):
    merge_method: str
    parameters: Dict[str, Union[float, List[float], Dict[str, Union[float, List[float]]]]] = Field(default_factory=dict)
    custom_dir: str = Field(default="./merged_models")
    ps_techniques: List[str] = ["linear"]
    dfs_techniques: List[str] = []

    @validator('merge_method')
    def validate_merge_method(cls, v):
        valid_methods = ["ps", "dfs", "ps_dfs"]
        if v not in valid_methods:
            raise ValueError(f"Invalid merge method. Choose from: {', '.join(valid_methods)}")
        return v

    @validator('ps_techniques', 'dfs_techniques')
    def validate_techniques(cls, v):
        valid_techniques = ["linear", "slerp", "ties", "dare", "task_arithmetic", "frankenmerge", "dfs"]
        for technique in v:
            if technique not in valid_techniques:
                raise ValueError(f"Invalid technique: {technique}. Choose from: {', '.join(valid_techniques)}")
        return v

class EvolutionSettings(BaseModel):
    population_size: int = Field(default=8, ge=2)
    num_generations: int = Field(default=50, ge=1)
    mutation_rate: float = Field(default=0.1, ge=0, le=1)
    tournament_size: int = Field(default=3, ge=2)
    early_stopping_generations: int = Field(default=10, ge=1)

    @validator('tournament_size')
    def validate_tournament_size(cls, v, values):
        if 'population_size' in values and v > values['population_size']:
            raise ValueError("Tournament size must be less than or equal to population size")
        return v

class Configuration(BaseModel):
    models: List[ModelReference]
    merge_settings: MergeSettings
    evolution_settings: EvolutionSettings

    class Config:
        extra = "allow"

def create_default_config() -> Configuration:
    return Configuration(
        models=[
            ModelReference(name="model1", path="placeholder_path_1"),
            ModelReference(name="model2", path="placeholder_path_2"),
            ModelReference(name="model3", path="placeholder_path_3")
        ],
        merge_settings=MergeSettings(
            merge_method="ps_dfs",
            parameters={
                "linear": {"weights": [1/3, 1/3, 1/3]},
                "slerp": {"t": 0.5},
                "ties": {"threshold": 0.1},
                "dare": {"threshold": 0.1, "amplification": 2.0},
            },
            ps_techniques=["linear", "ties"],
            dfs_techniques=["frankenmerge"]
        ),
        evolution_settings=EvolutionSettings()
    )

if __name__ == "__main__":
    # Test the configuration
    config = create_default_config()
    print(config.json(indent=2))
