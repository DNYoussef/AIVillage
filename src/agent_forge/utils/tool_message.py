from pydantic import BaseModel


class ToolMessage(BaseModel):
    request: str
    purpose: str

    def handle(self):
        raise NotImplementedError("This method should be implemented by subclasses")
