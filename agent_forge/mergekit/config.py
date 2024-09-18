from typing import List, Optional
from pydantic import BaseModel

class ModelReference(BaseModel):
    name: str
    path: Optional[str] = None

class MergeKitConfig(BaseModel):
    merge_method: str
    models: List[ModelReference]
    parameters: Optional[dict] = None

    class Config:
        extra = "allow"

    @property
    def merge_methods(self):
        return ["linear", "slerp"]  # Add more methods as they are implemented

    def validate_merge_method(self):
        if self.merge_method not in self.merge_methods:
            raise ValueError(f"Invalid merge method. Choose from: {', '.join(self.merge_methods)}")