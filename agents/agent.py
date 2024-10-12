from typing import List, Callable
from pydantic import BaseModel

class Agent(BaseModel):
    name: str
    model: str
    instructions: str
    tools: List[Callable]
