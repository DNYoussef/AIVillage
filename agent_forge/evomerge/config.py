from typing import List, Dict, Union
from pydantic import BaseModel, Field, validator

class ModelReference(BaseModel):
    name: str
    path: str  # Hugging Face model ID or local path

class MergeConfig(BaseModel):
    merge_method: str
    models: List[ModelReference]
    parameters: Dict[str, Union[float, List[float], Dict[str, float]]] = Field(default_factory=dict)
    custom_dir: str = Field(default="./merged_models")
    ps_techniques: List[str] = ["linear"]
    dfs_techniques: List[str] = []

    class Config:
        extra = "allow"

    @property
    def merge_methods(self):
        return ["ps", "dfs", "ps_dfs"]

    @validator('merge_method')
    def validate_merge_method(cls, v):
        if v not in cls.merge_methods:
            raise ValueError(f"Invalid merge method. Choose from: {', '.join(cls.merge_methods)}")
        return v

    @validator('ps_techniques', 'dfs_techniques')
    def validate_techniques(cls, v):
        valid_techniques = ["linear", "slerp", "ties", "dare", "task_arithmetic", "frankenmerge", "dfs"]
        for technique in v:
            if technique not in valid_techniques:
                raise ValueError(f"Invalid technique: {technique}. Choose from: {', '.join(valid_techniques)}")
        return v

class EvolutionConfig(BaseModel):
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
