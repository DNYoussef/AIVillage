from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class ModelReference(BaseModel):
    name: str
    path: str  # Hugging Face model ID or local path
class MergeKitConfig(BaseModel):
    merge_method: str
    models: List[ModelReference]
    parameters: Dict[str, any] = Field(default_factory=dict)
    custom_dir: str = Field(default="./merged_models")
    ps_techniques: List[str] = ["linear"]
    dfs_techniques: List[str] = []
    class Config:
        extra = "allow"

    @property
    def merge_methods(self):
        return ["linear", "slerp"]  # Add more methods as they are implemented

    def validate_merge_method(self):
        if self.merge_method not in self.merge_methods:
            raise ValueError(f"Invalid merge method. Choose from: {', '.join(self.merge_methods)}")

class EvolutionConfig(BaseModel):
    population_size: int = 8
    num_generations: int = 50
    mutation_rate: float = 0.1
    tournament_size: int = 3